# for quick testing with SwaggerUI within FastAPI
from pydantic import BaseModel

class ClaimData(BaseModel):
    model_type: str
    claim_amount: float
    days_to_submit: int
    previous_claims_count: int
    customer_tenure: float
    location_risk_score: float
    claim_type: str

