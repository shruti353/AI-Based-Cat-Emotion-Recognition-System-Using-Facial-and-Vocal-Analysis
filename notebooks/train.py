import os
import numpy as np
import librosa
import joblib
import tensorflow as tf
import tensorflow_hub as hub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# CONFIG
# ===============================

DATA_PATH = r"C:\Users\HP\Downloads\audio_augmented-20260128T114635Z-3-001\audio_augmented"
SAMPLE_RATE = 16000

# ===============================
# LOAD YAMNET
# ===============================

print("🔹 Loading YAMNet model...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# ===============================
# FEATURE EXTRACTION
# ===============================

def extract_yamnet_embedding(audio_path):
    waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    scores, embeddings, spectrogram = yamnet_model(waveform)
    return np.mean(embeddings.numpy(), axis=0)  # (1024,)

# ===============================
# LOAD DATASET
# ===============================

def load_dataset():
    X, y = [], []

    print("\n🔍 Scanning dataset...")

    for label in os.listdir(DATA_PATH):
        folder = os.path.join(DATA_PATH, label)

        if not os.path.isdir(folder):
            continue

        files = [f for f in os.listdir(folder) if f.endswith(".mp3")]
        print(f"✅ {label}: {len(files)} files")

        for file in files:
            path = os.path.join(folder, file)
            try:
                embedding = extract_yamnet_embedding(path)
                X.append(embedding)
                y.append(label)
            except Exception as e:
                print(f"⚠️ Error processing {path}: {e}")

    return np.array(X), np.array(y)

# ===============================
# MAIN TRAINING PIPELINE
# ===============================

def main():
    print("\n🔹 Loading dataset...")
    X, y = load_dataset()

    print("\n📊 Dataset shape:", X.shape)

    if len(X) == 0:
        raise RuntimeError("Dataset is empty. Check DATA_PATH.")

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Train / Val / Test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded,
        test_size=0.3,
        random_state=42,
        stratify=y_encoded
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    # ===============================
    # MODEL
    # ===============================

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,
        class_weight="balanced"
    )

    print("\n🚀 Training SVM on YAMNet embeddings...")
    model.fit(X_train, y_train)

    # ===============================
    # EVALUATION
    # ===============================

    print("\n📈 Train Accuracy:", accuracy_score(y_train, model.predict(X_train)))
    print("📈 Val Accuracy:", accuracy_score(y_val, model.predict(X_val)))
    print("📈 Test Accuracy:", accuracy_score(y_test, model.predict(X_test)))

    print("\n📄 Classification Report:\n")
    print(
        classification_report(
            y_test,
            model.predict(X_test),
            target_names=encoder.classes_
        )
    )

    # ===============================
    # SAVE
    # ===============================

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/yamnet_svm.pkl")
    joblib.dump(encoder, "models/label_encoder.pkl")

    print("\n💾 Model & encoder saved successfully!")

if __name__ == "__main__":
    main()
