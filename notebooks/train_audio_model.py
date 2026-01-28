import joblib
from dataset import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "audio_augmented"

# Load data
X, y, labels = load_dataset(DATA_PATH)

print("Samples:", X.shape)
print("Classes:", labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# SVM model (best classical choice for audio)
model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True,
    class_weight="balanced"
)

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=labels))

# Save model
joblib.dump(model, "cat_audio_emotion_svm.pkl")
joblib.dump(labels, "labels.pkl")

print("\n✅ Model saved successfully")
