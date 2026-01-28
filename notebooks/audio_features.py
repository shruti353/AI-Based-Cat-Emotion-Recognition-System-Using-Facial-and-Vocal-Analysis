import librosa
import numpy as np

def extract_mfcc(file_path, duration=3, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=16000)
    y = librosa.util.fix_length(y, size=sr * duration)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc
    )

    return np.mean(mfcc.T, axis=0)
