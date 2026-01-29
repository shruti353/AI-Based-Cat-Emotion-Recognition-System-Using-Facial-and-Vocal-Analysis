from pydantic import BaseModel
from typing import Optional, List

class PredictionResponse(BaseModel):
    predicted_emotion: str
    confidence: float
    all_predictions: Optional[dict] = None
    processing_time: Optional[float] = None

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    supported_formats: List[str]
