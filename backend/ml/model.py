import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import shap
import numpy as np

class FraudDetectionPipeline:
    def __init__(self, model_type="random_forest", random_state=42):
        # features consistent with preprocessing.py feature engineering
        self.numeric_features = [
            "claim_amount",
            "days_to_submit",
            "customer_tenure",
            "previous_claims_count",
            "location_risk_score",
            "avg_claim_per_year",
            "claims_per_year"
        ]
        self.categorical_features = ["claim_type"]
        self.boolean_features = ["is_high_risk_region"]

        # Define transformers
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])

        # Combine transformers into a ColumnTransformer
        self.preprocessor = ColumnTransformer([
            ('num', numeric_transformer, self.numeric_features),
            ('cat', categorical_transformer, self.categorical_features),
            ('bool', 'passthrough', self.boolean_features)
        ])

        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=random_state
            )
        elif model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=random_state
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # Create full pipeline: preprocessing + classifier
        self.pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', self.model)
        ])

        self.is_trained = False
        self.model_type = model_type

    def train(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True

    def evaluate(self, X_test, y_test):
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")
        
        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test)[:, 1]

        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    def predict_proba(self, X):
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")
        return self.pipeline.predict_proba(X)
    
    def predict(self, X):
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")
        return self.pipeline.predict(X)

    def explain(self, X_train):
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")

        # SHAP explainer requires just the classifier model
        explainer = shap.TreeExplainer(self.pipeline.named_steps['classifier'])

        # preprocess data first
        X_processed = self.pipeline.named_steps['preprocessor'].transform(X_train)

        shap_values = explainer.shap_values(X_processed)

        feature_names = self.get_feature_names()

        print(f"SHAP values shape: {np.array(shap_values).shape}")
        print(f"Feature names ({len(feature_names)}): {feature_names}")

        shap.summary_plot(shap_values, features=X_processed, feature_names=feature_names, plot_type="bar")
        shap.summary_plot(shap_values, features=X_processed, feature_names=feature_names)

        return shap_values

    def get_feature_names(self):
        numeric = self.numeric_features
        boolean = self.boolean_features

        # The one-hot encoded categorical feature names changed and thus are retrieved from the pipeline preprocessor
        cat = list(
            self.pipeline.named_steps['preprocessor']
            .named_transformers_['cat']
            .named_steps['onehot']
            .get_feature_names_out(self.categorical_features)
        )

        return numeric + cat + boolean
