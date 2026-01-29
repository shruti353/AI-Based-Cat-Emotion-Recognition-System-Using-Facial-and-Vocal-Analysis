import os
import joblib
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub

# ===============================
# PATH CONFIG
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "yamnet_svm.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
TEST_AUDIO_DIR = os.path.join(BASE_DIR, "test_audio")

SUPPORTED_FORMATS = (".wav", ".mp3", ".mpeg")

SAMPLE_RATE = 16000   # YAMNet REQUIRED sample rate

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

    # Convert to TensorFlow tensor
    waveform = tf.convert_to_tensor(y, dtype=tf.float32)

    scores, embeddings, spectrogram = yamnet_model(waveform)

    # Average embeddings over time → (1024,)
    embedding_mean = tf.reduce_mean(embeddings, axis=0)

    return embedding_mean.numpy()


# ===============================
# PREDICTION FUNCTION
# ===============================
def predict_emotion(audio_path):
    features = extract_yamnet_features(audio_path).reshape(1, -1)

    pred = model.predict(features)[0]
    emotion = encoder.inverse_transform([pred])[0]

    if hasattr(model, "predict_proba"):
        confidence = np.max(model.predict_proba(features))
    else:
        confidence = None

    return emotion, confidence


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":

    print("Testing unseen audio files...\n")
    print("Looking for audio files in:", TEST_AUDIO_DIR)

    audio_files = [
        f for f in os.listdir(TEST_AUDIO_DIR)
        if f.lower().endswith(SUPPORTED_FORMATS)
    ]

    if not audio_files:
        print("❌ No audio files found.")
        exit()

    for file in audio_files:
        path = os.path.join(TEST_AUDIO_DIR, file)
        emotion, confidence = predict_emotion(path)

        if confidence is not None:
            print(f"{file} → {emotion} (confidence: {confidence:.2f})")
        else:
            print(f"{file} → {emotion}")

    print("\nTesting completed.")
