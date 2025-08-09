import streamlit as st
import requests

#API_URL = "http://127.0.0.1:8000/predict"  # FastAPI should be running here for local testing
API_URL = "http://backend:8000/predict"     # for Docker localhost or 127.0.0.1 refers to the container itself. thus a different container is accessed by its service name 

st.title("Insurance Fraud Detection")
st.subheader("🚨 Insurance Fraud Detection via Machine Learning API 🚨")
st.write("This app sends claim details to a Machine Learning-powered API that analyzes the data and predicts the likelihood of insurance fraud.")

# User Inputs
model_type = st.selectbox("Model Type", ['XGBoost', 'Random Forest'])
st.markdown("---")  
st.write("Enter claim details to predict fraud.")
claim_amount = st.number_input("Claim Amount in Euros", value=1200.0)
days_to_submit = st.number_input("Number Of Days Passed Between Incident And Claim", value=10)
previous_claims_count = st.number_input("Number Of Previous Claims", value=2)
customer_tenure = st.number_input("Customer Tenure In Years", value=5.0)
location_risk_score = st.slider("Location Risk Score (Based On Previous Fraudulent Activity In This Region)", 0.0, 1.0, value=0.3)
claim_type = st.selectbox("Claim Type", ["Health", "Property", "Auto", "Travel"])

if st.button("Predict"):
    input_data = {
        "model_type": model_type,
        "claim_amount": claim_amount,
        "days_to_submit": days_to_submit,
        "previous_claims_count": previous_claims_count,
        "customer_tenure": customer_tenure,
        "location_risk_score": location_risk_score,
        "claim_type": claim_type
    }

    try:
        response = requests.post(API_URL, json=input_data)
        result = response.json()
        st.success(f"Prediction: {'FRAUD' if result['prediction'] == 1 else 'LEGITIMATE'}")
        st.info(f"Fraud Probability: {result['fraud_probability']}")
    except Exception as e:
        st.error(f"Error: {e}")
