# 🛡️ Insurance Fraud Detection

This project is a machine learning pipeline that detects fraudulent insurance claims using synthetic data. It includes data preparation, feature engineering, model training, a REST API using FastAPI, and a frontend built with Streamlit.

DISCLAIMER: this README was automatically generated

---

## 📌 Problem Statement

The goal of this project is to build a predictive model that identifies potentially fraudulent insurance claims based on structured data.

---

## 🧪 Synthetic Dataset

This project uses **synthetically generated data** that mimics real-world insurance claim patterns. The following features are used.

### Features

- `claim_id`: Unique identifier
- `claim_amount`: Euro amount of the insurance claim
- `days_to_submit`: Days passed between the incident and the file of the claim
- `previous_claims_count`: Number of previous claims of the customer
- `customer_tenure`: Number of years the customer has been with the company
- `location_risk_score`: A score refering to the the area the customer is from and its past proneness for fraudulent insurance claims
- `claim_type`: Type of incident (e.g., "Health", "Property", "Auto", "Life")
- `is_fraud`: Target variable (0: Legit, 1: Fraud)

The data is saved as `simulated_fraud_claims.csv`.

---

## ⚙️ Feature Engineering & Preprocessing

The `preprocessing.py` file handles:

- One-hot encoding for categorical features (`claim_type`)
- Standard scaling of numerical features
- Feature Engineering of three new features:
   -`avg_claim_per_year` = `claim_amount` / `customer_tenure`  
   -`claims_per_year` = `previous_claims_count` / `customer_tenure`  
   -`is_high_risk_region` = `location_risk_score` > 0.8
- Train-test splitting

---

## 🧠 Models

Implemented via `model.py`:

- Random Forest
- XGBoost

A wrapper class `FraudDetectionPipeline` handles:

- Preprocessing of the data (one-hot encoding, etc)
- Training
- Evaluation (Accuracy, AUC)
- SHAP-based explainability
- Model saving/loading with `joblib`

---

## 🚀 FastAPI Backend

A FastAPI backend is built for quick testing.

### File: `api/api_main.py`

- `/predict`: Accepts JSON input and returns prediction result.

### File: `api/schemas.py`

Defines Pydantic models for request validation.

---

## 💻 Streamlit Frontend

### File: `frontend/app.py`

- A form-based UI to enter insurance claim details
- On submit, it sends data to the FastAPI backend
- Displays fraud prediction result (fraud or not fraud)

---

## 🐳 Dockerized Architecture

All components run in isolated containers using Docker Compose:

### Services

- `backend`: FastAPI + model
- `frontend`: Streamlit app

### 🔸 Network

No explicit network config needed — Docker Compose creates a default network where services can resolve each other by name.

---

## 🧪 Run the Project (With Docker)

1. Copy the docker-compose-dockerhub.yml

2. Run

   ```bash
   docker-compose -f docker-compose-dockerhub.yml up --build
   ```

Alternatively:

1. Clone the repo

2. Build and start containers:

   ```bash
   docker-compose up --build
   ```

3. Access frontend (Streamlit):\
   [http://localhost:8501](http://localhost:8501)

4. Backend (FastAPI docs):\
   [http://localhost:8000/docs](http://localhost:8000/docs)

---

## ✅ Functionality Summary

| Component | Functionality                            |
| --------- | ---------------------------------------- |
| Model     | Trains on synthetic fraud data           |
| API       | Receives claim input, returns prediction |
| Streamlit | User-friendly fraud detection UI         |
| Docker    | Full-stack deployment, portable          |

---
