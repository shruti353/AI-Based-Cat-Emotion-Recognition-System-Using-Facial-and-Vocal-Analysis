import joblib
import numpy as np
from audio_features import extract_mfcc

model = joblib.load("cat_audio_emotion_svm.pkl")
labels = joblib.load("labels.pkl")

def predict_emotion(audio_path):
    features = extract_mfcc(audio_path)
    features = np.expand_dims(features, axis=0)

    probs = model.predict_proba(features)[0]
    idx = probs.argmax()

    return labels[idx], probs[idx]

emotion, confidence = predict_emotion(
    "audio_augmented/Happy/cat_8_aug1.mp3"
)

print("Predicted Emotion:", emotion)
print("Confidence:", round(confidence, 2))
