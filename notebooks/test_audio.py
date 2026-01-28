import os
import joblib
import librosa
import numpy as np

# ===============================
# CONFIGURATION
# ===============================
MODEL_PATH = "models/yamnet_svm.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
TEST_AUDIO_DIR = "test_audio"

# folder with unseen .wav files

SAMPLE_RATE = 22050
N_MFCC = 40


# ===============================
# LOAD MODEL & ENCODER
# ===============================
print("Loading model and encoder...")
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
print("Model and encoder loaded successfully.\n")


# ===============================
# FEATURE EXTRACTION
# ===============================
def extract_audio_features(file_path):
    """
    Extract MFCC features from an audio file
    Must match training-time feature extraction
    """
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=N_MFCC
        )

        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# ===============================
# PREDICTION FUNCTION
# ===============================
def predict_emotion(audio_path):
    features = extract_audio_features(audio_path)

    if features is None:
        return None, None

    features = features.reshape(1, -1)

    # Predict class
    pred = model.predict(features)[0]
    emotion = encoder.inverse_transform([pred])[0]

    # Predict confidence if supported
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        confidence = np.max(probs)
    else:
        confidence = None

    return emotion, confidence


# ===============================
# TEST ON UNSEEN AUDIO FILES
# ===============================
if __name__ == "__main__":
    print("Testing unseen audio files...\n")

    if not os.path.exists(TEST_AUDIO_DIR):
        print(f"Test audio directory '{TEST_AUDIO_DIR}' not found!")
        exit()

    audio_files = [
    f for f in os.listdir(TEST_AUDIO_DIR)
    if f.lower().endswith((".wav", ".mp3"))
]

    if len(audio_files) == 0:
        print("No .wav files found in test_audio folder.")
        exit()

    for file in audio_files:
        file_path = os.path.join(TEST_AUDIO_DIR, file)
        emotion, confidence = predict_emotion(file_path)

        if emotion is None:
            continue

        if confidence is not None:
            print(f"{file}  →  {emotion}  (confidence: {confidence:.2f})")
        else:
            print(f"{file}  →  {emotion}")

    print("\nTesting completed.")
