import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # One-hot encode claim_type
    # Not needed here as it is done within the model
    # df = pd.get_dummies(df, columns=["claim_type"], drop_first=True)

    # new features that are engineered from the existing ones
    df["avg_claim_per_year"] = df["claim_amount"] / df["customer_tenure"]
    df["claims_per_year"] = df["previous_claims_count"] / df["customer_tenure"]
    df["is_high_risk_region"] = (df["location_risk_score"] > 0.8).astype(int)

    # drop ID column as it is not relevant for training
    df = df.drop(columns=["claim_id"], errors='ignore')
    return df

def prepare_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    df = feature_engineering(df)

    # separate features and target
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


### deprectated as this step is now part within a pipeline defined within the model.py

# Define columns used in transformation
NUMERIC_FEATURES = [
    "claim_amount", "days_to_submit", "previous_claims_count",
    "customer_tenure", "location_risk_score", "avg_claim_per_year",
    "claims_per_year"
]

CATEGORICAL_FEATURES = ["claim_type"]
BOOLEAN_FEATURES = ["is_high_risk_region"]

def get_pipeline():
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, NUMERIC_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES),
        ('bool', 'passthrough', BOOLEAN_FEATURES)
    ])

    return preprocessor
    
