import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/synthetic_dataset.csv")

# Preprocessing
df["spo2"] = df["spo2"].clip(70, 100)
X = df[["heart_rate", "spo2", "desaturation_events"]]
y = df["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature extraction
df["hr_variability"] = np.abs(df["heart_rate"] - df["heart_rate"].mean())
df["spo2_risk"] = (100 - df["spo2"]) + (df["desaturation_events"] * 0.1)

X = df[["heart_rate", "spo2", "desaturation_events", "hr_variability", "spo2_risk"]]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("✅ Training samples:", len(X_train))
print("✅ Testing samples:", len(X_test))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Train classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

# Evaluate
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Save trained model ---
import joblib
joblib.dump(clf, "synthetic_model.pkl")
print("✅ Model saved as synthetic_model.pkl")

# --- Prediction function ---
def predict_case(hr, spo2, desats):
    # Derived features
    hr_variability = abs(hr - df["heart_rate"].mean())
    spo2_risk = (100 - spo2) + (desats * 0.1)

    # Build feature vector
    X_new = [[hr, spo2, desats, hr_variability, spo2_risk]]

    # Predict using trained model
    return clf.predict(X_new)[0]

# Example usage
print("Prediction for HR=120, SpO2=91, desats=3 →", predict_case(120, 91, 3))
