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

