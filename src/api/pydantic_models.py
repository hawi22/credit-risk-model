from pydantic import BaseModel

class TransactionInput(BaseModel):
    Amount: float
    Value: float
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    PricingStrategy: int

class PredictionResponse(BaseModel):
    risk_probability: float
    is_high_risk: int