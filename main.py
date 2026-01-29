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


# To run the application:
# pip install fastapi uvicorn python-multipart librosa tensorflow tensorflow-hub scikit-learn joblib
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
