import pandas as pd
from backend.ml.preprocessing import prepare_data, get_pipeline
from backend.ml.model import FraudDetectionPipeline
import joblib
import os

def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(f"{curr_dir}/data/simulated_fraud_claims.csv") 

    # feature engineering + train-test split)
    X_train, X_test, y_train, y_test = prepare_data(df)

    # choose model type 'random_forest' or 'xgboost'
    model_type = 'xgboost'
    model_pipeline = FraudDetectionPipeline(model_type=model_type)

    model_pipeline.train(X_train, y_train)

    model_pipeline.evaluate(X_test, y_test)

    model_pipeline.explain(X_train)

    # save trained model
    joblib.dump(model_pipeline, f"{curr_dir}/models/fraud_model_{model_type}.pkl")


if __name__ == "__main__":
    main()
