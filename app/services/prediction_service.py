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
