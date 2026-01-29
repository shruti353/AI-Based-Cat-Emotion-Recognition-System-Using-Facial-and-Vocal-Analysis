from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
from app.models.schemas import PredictionResponse, ErrorResponse, HealthResponse
from app.services.prediction_service import prediction_service
from app.core.config import settings

router = APIRouter()

@router.post("/audio", response_model=PredictionResponse)
async def predict_cat_emotion(audio_file: UploadFile = File(...)):
    """
    Predict cat emotion from uploaded audio file
    """
    try:
        # Validate file format
        if not audio_file.filename.lower().endswith(settings.SUPPORTED_FORMATS):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported formats: {settings.SUPPORTED_FORMATS}"
            )
        
        # Validate file size
        audio_data = await audio_file.read()
        if len(audio_data) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024):.1f}MB"
            )
        
        # Check if models are loaded
        if not prediction_service.is_model_loaded():
            raise HTTPException(
                status_code=500,
                detail="Models not loaded properly"
            )
        
        # Make prediction
        predicted_emotion, confidence, all_predictions, processing_time = prediction_service.predict_emotion(audio_data)
        
        return PredictionResponse(
            predicted_emotion=predicted_emotion,
            confidence=confidence,
            all_predictions=all_predictions,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@router.get("/health", response_model=HealthResponse)
async def prediction_health():
    """
    Check prediction service health
    """
    return HealthResponse(
        status="healthy" if prediction_service.is_model_loaded() else "unhealthy",
        model_loaded=prediction_service.is_model_loaded(),
        supported_formats=list(settings.SUPPORTED_FORMATS)
    )
