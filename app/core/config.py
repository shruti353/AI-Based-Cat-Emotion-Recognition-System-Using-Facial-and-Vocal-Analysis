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
