import os
import numpy as np
from audio_features import extract_mfcc

LABELS = [
    "Angry",
    "Defence",
    "Fighting",
    "Happy",
    "HuntingMind",
    "Mating",
    "MotherCall",
    "Painning",
    "Resting",
    "Warning"
]

LABEL_MAP = {label: idx for idx, label in enumerate(LABELS)}

def load_dataset(base_path):
    X, y = [], []

    for label in LABELS:
        folder_path = os.path.join(base_path, label)

        if not os.path.isdir(folder_path):
            print(f"Skipping missing folder: {folder_path}")
            continue

        for file in os.listdir(folder_path):
            if file.lower().endswith(".mp3"):
                file_path = os.path.join(folder_path, file)
                try:
                    features = extract_mfcc(file_path)
                    X.append(features)
                    y.append(LABEL_MAP[label])
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    return np.array(X), np.array(y), LABELS
