import os
import joblib
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# ===============================
# PATH CONFIG
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "yamnet_svm.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

TEST_AUDIO_DIR = os.path.join(BASE_DIR, "test_audio")
CSV_PATH = os.path.join(BASE_DIR, "test_labels.csv")

SUPPORTED_FORMATS = (".wav", ".mp3", ".mpeg")
SAMPLE_RATE = 16000  # YAMNet required

# ===============================
# LOAD MODEL & ENCODER
# ===============================
print("Loading SVM model and encoder...")
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

print("Loading YAMNet model...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

print("Models loaded successfully.\n")

# ===============================
# YAMNET FEATURE EXTRACTION
# ===============================
def extract_yamnet_features(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    waveform = tf.convert_to_tensor(y, dtype=tf.float32)

    scores, embeddings, spectrogram = yamnet_model(waveform)

    # Mean pooling → (1024,)
    return tf.reduce_mean(embeddings, axis=0).numpy()

# ===============================
# MAIN TESTING WITH CSV
# ===============================
if __name__ == "__main__":

    print("Evaluating unseen test audio using CSV labels...\n")

    df = pd.read_csv(CSV_PATH)

    y_true = []
    y_pred = []

    results = []

    for _, row in df.iterrows():
        file_name = row["file_name"]
        true_label = row["true_label"]

        audio_path = os.path.join(TEST_AUDIO_DIR, file_name)

        if not os.path.exists(audio_path):
            print(f"❌ File not found: {file_name}")
            continue

        features = extract_yamnet_features(audio_path).reshape(1, -1)

        pred_index = model.predict(features)[0]
        pred_label = encoder.inverse_transform([pred_index])[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = np.max(model.predict_proba(features))

        correct = pred_label == true_label

        true_label = true_label.strip().lower()
        pred_label = pred_label.strip().lower()

        y_true.append(true_label)
        y_pred.append(pred_label)

        results.append({
            "file": file_name,
            "true_label": true_label,
            "predicted_label": pred_label,
            "confidence": round(confidence, 3) if confidence else None,
            "correct": correct
        })

        print(
            f"{file_name} → Predicted: {pred_label}, "
            f"True: {true_label}, "
            f"Correct: {correct}, "
            f"Confidence: {confidence:.2f}"
        )

    # ===============================
    # SUMMARY METRICS
    # ===============================
    print("\n===============================")
    print("Overall Evaluation Results")
    print("===============================")


    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # ===============================
    # SAVE RESULTS TO CSV
    # ===============================
    results_df = pd.DataFrame(results)
    results_output_path = os.path.join(BASE_DIR, "test_results.csv")
    results_df.to_csv(results_output_path, index=False)
    print(f"\nResults saved to: {results_output_path}")
    
    # ===============================
    # DETAILED ANALYSIS
    # ===============================
    total_files = len(results)
    correct_predictions = sum(1 for r in results if r["correct"])
    
    print(f"\nTotal files tested: {total_files}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Incorrect predictions: {total_files - correct_predictions}")
    
    # Per-class accuracy
    class_stats = {}
    for result in results:
        true_label = result["true_label"]
        if true_label not in class_stats:
            class_stats[true_label] = {"total": 0, "correct": 0}
        class_stats[true_label]["total"] += 1
        if result["correct"]:
            class_stats[true_label]["correct"] += 1
    
    print("\nPer-class accuracy:")
    for class_name, stats in class_stats.items():
        class_accuracy = (stats["correct"] / stats["total"]) * 100
        print(f"  {class_name}: {class_accuracy:.2f}% ({stats['correct']}/{stats['total']})")
    
    # Confidence analysis
    if any(r["confidence"] for r in results):
        confidences = [r["confidence"] for r in results if r["confidence"] is not None]
        avg_confidence = np.mean(confidences)
        print(f"\nAverage confidence: {avg_confidence:.3f}")
        
        # High confidence correct vs incorrect
        high_conf_threshold = 0.8
        high_conf_correct = sum(1 for r in results if r["confidence"] and r["confidence"] > high_conf_threshold and r["correct"])
        high_conf_total = sum(1 for r in results if r["confidence"] and r["confidence"] > high_conf_threshold)
        
        if high_conf_total > 0:
            high_conf_accuracy = (high_conf_correct / high_conf_total) * 100
            print(f"High confidence (>{high_conf_threshold}) accuracy: {high_conf_accuracy:.2f}% ({high_conf_correct}/{high_conf_total})")
            
# ===============================
# FASTAPI BACKEND FOR CATCARE AI
# ===============================

# Create the following folder structure:
# catcare_ai/
# ├── main.py
# ├── app/
# │   ├── __init__.py
# │   ├── api/
# │   │   ├── __init__.py
# │   │   └── routes/
# │   │       ├── __init__.py
# │   │       └── prediction.py
# │   ├── core/
# │   │   ├── __init__.py
# │   │   └── config.py
# │   ├── models/
# │   │   ├── __init__.py
# │   │   └── schemas.py
# │   └── services/
# │       ├── __init__.py
# │       └── prediction_service.py
# └── models/
#     ├── yamnet_svm.pkl
#     └── label_encoder.pkl

# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes.prediction import router as prediction_router

app = FastAPI(
    title="CatCare AI",
    description="Cat emotion recognition API using YAMNet + SVM",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction_router, prefix="/predict", tags=["prediction"])

@app.get("/")
async def root():
    return {"message": "CatCare AI - Cat Emotion Recognition API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# app/__init__.py
# Empty file

# app/core/__init__.py
# Empty file

# app/core/config.py
import os
from pathlib import Path

class Settings:
    PROJECT_NAME: str = "CatCare AI"
    VERSION: str = "1.0.0"
    
    # Model paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    MODEL_DIR: Path = BASE_DIR / "models"
    SVM_MODEL_PATH: Path = MODEL_DIR / "yamnet_svm.pkl"
    ENCODER_PATH: Path = MODEL_DIR / "label_encoder.pkl"
    
    # Audio processing
    SAMPLE_RATE: int = 16000
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    SUPPORTED_FORMATS: tuple = (".wav", ".mp3", ".mpeg", ".m4a")

settings = Settings()

# app/models/__init__.py
# Empty file

# app/models/schemas.py
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

# app/services/__init__.py
# Empty file

# app/services/prediction_service.py
import os
import joblib
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import tempfile
import time
from typing import Tuple, Dict
from app.core.config import settings

class PredictionService:
    def __init__(self):
        self.svm_model = None
        self.label_encoder = None
        self.yamnet_model = None
        self._load_models()
    
    def _load_models(self):
        """Load SVM model, encoder, and YAMNet model"""
        try:
            # Load SVM model and encoder
            self.svm_model = joblib.load(settings.SVM_MODEL_PATH)
            self.label_encoder = joblib.load(settings.ENCODER_PATH)
            
            # Load YAMNet model
            self.yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
            
            print("✅ All models loaded successfully")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise e
    
    def extract_yamnet_features(self, audio_data: bytes) -> np.ndarray:
        """Extract YAMNet features from audio bytes"""
        # Save audio bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        
        try:
            # Load audio with librosa
            y, sr = librosa.load(temp_path, sr=settings.SAMPLE_RATE)
            waveform = tf.convert_to_tensor(y, dtype=tf.float32)
            
            # Extract YAMNet embeddings
            scores, embeddings, spectrogram = self.yamnet_model(waveform)
            
            # Mean pooling to get single feature vector
            features = tf.reduce_mean(embeddings, axis=0).numpy()
            
            return features
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
    
    def predict_emotion(self, audio_data: bytes) -> Tuple[str, float, Dict]:
        """Predict cat emotion from audio data"""
        start_time = time.time()
        
        # Extract features
        features = self.extract_yamnet_features(audio_data)
        features = features.reshape(1, -1)
        
        # Make prediction
        pred_index = self.svm_model.predict(features)[0]
        predicted_emotion = self.label_encoder.inverse_transform([pred_index])[0]
        
        # Get confidence scores
        confidence_scores = {}
        if hasattr(self.svm_model, "predict_proba"):
            probabilities = self.svm_model.predict_proba(features)[0]
            confidence = np.max(probabilities)
            
            # Map all class probabilities
            for i, prob in enumerate(probabilities):
                class_name = self.label_encoder.inverse_transform([i])[0]
                confidence_scores[class_name] = float(prob)
        else:
            confidence = 1.0  # If no probability available
        
        processing_time = time.time() - start_time
        
        return predicted_emotion, float(confidence), confidence_scores, processing_time
    
    def is_model_loaded(self) -> bool:
        """Check if all models are loaded"""
        return all([
            self.svm_model is not None,
            self.label_encoder is not None,
            self.yamnet_model is not None
        ])

# Create singleton instance
prediction_service = PredictionService()

# app/api/__init__.py
# Empty file

# app/api/routes/__init__.py
# Empty file

# app/api/routes/prediction.py
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

# To run the application:
# pip install fastapi uvicorn python-multipart librosa tensorflow tensorflow-hub scikit-learn joblib
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
