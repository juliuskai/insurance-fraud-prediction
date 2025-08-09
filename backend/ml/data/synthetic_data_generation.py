import pandas as pd
import numpy as np
import os

# This is to create a synthetic data set which the project is based on
# This data set will be used as the training data for the model

def generate_synthetic_data(n_samples=10000, fraud_ratio=0.05, random_state=42):
    np.random.seed(random_state)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_non_fraud = n_samples - n_fraud

    # generate non-fraud data
    non_fraud = pd.DataFrame({
        "claim_id": np.arange(n_non_fraud),
        "claim_amount": np.random.normal(2000, 1200, n_non_fraud),
        "days_to_submit": np.random.randint(10, 60, n_non_fraud),
        "previous_claims_count": np.random.poisson(1.5, n_non_fraud),
        "customer_tenure": np.random.uniform(2, 10, n_non_fraud),
        "location_risk_score": np.random.uniform(0.2, 0.8, n_non_fraud),
        "claim_type": np.random.choice(["Health", "Property", "Auto", "Life"], n_non_fraud),
        "is_fraud": 0
    })

    # generate fraud data with some overlapping values
    fraud = pd.DataFrame({
        "claim_id": np.arange(n_non_fraud, n_samples),
        "claim_amount": np.random.normal(3000, 1500, n_fraud), 
        "days_to_submit": np.random.randint(20, 70, n_fraud),
        "previous_claims_count": np.random.poisson(2.5, n_fraud),
        "customer_tenure": np.random.uniform(1, 7, n_fraud),  
        "location_risk_score": np.random.uniform(0.4, 0.95, n_fraud),
        "claim_type": np.random.choice(["Health", "Property", "Auto", "Life"], n_fraud),
        "is_fraud": 1
    })

    # cmbine and shuffle
    data = pd.concat([non_fraud, fraud], ignore_index=True)
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # add noise to make features less clean
    data["claim_amount"] += np.random.normal(0, 300, size=n_samples)
    data["location_risk_score"] += np.random.normal(0, 0.05, size=n_samples)
    data["location_risk_score"] = data["location_risk_score"].clip(0, 1)

    return data

if __name__ == "__main__":
    df = generate_synthetic_data()
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    df.to_csv(f"{curr_dir}/simulated_fraud_claims.csv", index=False)
    print(f"Synthetic data saved to '{curr_dir}/simulated_fraud_claims.csv'")
