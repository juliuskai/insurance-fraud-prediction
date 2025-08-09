# for quick testing with SwaggerUI within FastAPI
from fastapi import FastAPI
from api.schemas import ClaimData
from api.predict import make_prediction

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Fraud Detection API is live."}

@app.post("/predict")
def predict_claim(data: ClaimData):
    input_dict = data.dict()
    result = make_prediction(input_dict)
    return result
