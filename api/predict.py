import joblib
import pandas as pd
from backend.ml.preprocessing import feature_engineering
import os

def make_prediction(input_data: dict) -> dict:
    # Load the full pipeline (preprocessor + classifier) saved earlier
    model_type = input_data.pop('model_type')
    model_type = str(model_type).lower().replace(' ', '_')
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline = joblib.load(os.path.join(curr_dir, "..", "backend", "ml", "models", f"fraud_model_{model_type}.pkl"))

    df = pd.DataFrame([input_data])
    
    df = feature_engineering(df)
    
    proba = pipeline.predict_proba(df)[0, 1]
    
    # threshold to make binary prediction above 50%
    prediction = int(proba >= 0.5)
    
    return {
        "prediction": prediction,
        "fraud_probability": float(round(proba, 3))
    }

