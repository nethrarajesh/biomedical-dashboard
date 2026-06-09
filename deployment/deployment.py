import neurokit2 as nk
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.filters import butter_lowpass_filter, moving_average
from preprocessing.baseline_correction import mean_baseline_correction, linear_baseline_correction
from features.extraction import extract_ecg_features, extract_spo2_features
from fusion.fusion import fuse_features
from models.models import train_model

# --- Alert System ---
def check_alerts(fused_features):
    alerts = []
    if fused_features["heart_rate"] > 120:
        alerts.append("🚨 High Heart Rate (Tachycardia)")
    if fused_features["heart_rate"] < 50:
        alerts.append("⚠️ Low Heart Rate (Bradycardia)")
    if fused_features["min_spo2"] < 90:
        alerts.append("🚨 Low SpO₂ (Hypoxemia)")
    if fused_features["desaturation_events"] > 0:
        alerts.append("⚠️ Oxygen Desaturation Detected")
    return alerts

# --- Pipeline Function ---
def run_pipeline(duration=10, sampling_rate=250):
    # Simulate Healthy Signals
    ecg_signal = nk.ecg_simulate(duration=duration, heart_rate=70, sampling_rate=sampling_rate)
    time = np.linspace(0, duration, len(ecg_signal))
    spo2_signal = 98 + 0.5 * np.sin(0.5 * time) + np.random.normal(0, 0.2, len(time))

    # Preprocessing
    ecg_filtered = butter_lowpass_filter(ecg_signal)
    ecg_corrected = mean_baseline_correction(ecg_filtered)

    spo2_filtered = butter_lowpass_filter(spo2_signal)
    spo2_corrected = linear_baseline_correction(spo2_filtered)  # optional for visualization

    # Feature Extraction
    ecg_features = extract_ecg_features(ecg_corrected, sampling_rate=sampling_rate)
    spo2_features = extract_spo2_features(spo2_filtered)  # ✅ use filtered, not corrected

    # Fusion
    fused_features = fuse_features(ecg_features, spo2_features)
    return fused_features

# --- Main Section ---
if __name__ == "__main__":
    # Healthy Case
    healthy_features = run_pipeline()
    print("Healthy Features:", healthy_features)
    alerts = check_alerts(healthy_features)
    if alerts:
        print("ALERTS:", alerts)

    # Abnormal Case (Tachycardia + Hypoxemia)
    ecg_tachy = nk.ecg_simulate(duration=10, heart_rate=120, sampling_rate=250)
    spo2_hypoxemia = 92 + 2 * np.sin(0.5 * np.linspace(0, 10, len(ecg_tachy))) + np.random.normal(0, 0.5, len(ecg_tachy))

    ecg_features_tachy = extract_ecg_features(ecg_tachy, sampling_rate=250)
    spo2_features_hypoxemia = extract_spo2_features(spo2_hypoxemia)
    fused_abnormal = fuse_features(ecg_features_tachy, spo2_features_hypoxemia)

    print("Abnormal Features:", fused_abnormal)
    alerts = check_alerts(fused_abnormal)
    if alerts:
        print("ALERTS:", alerts)
            # Abnormal Case 2: Bradycardia
    ecg_brady = nk.ecg_simulate(duration=10, heart_rate=40, sampling_rate=250)
    spo2_normal = 98 + 0.3 * np.sin(0.5 * np.linspace(0, 10, len(ecg_brady))) + np.random.normal(0, 0.2, len(ecg_brady))
    ecg_features_brady = extract_ecg_features(ecg_brady, sampling_rate=250)
    spo2_features_normal = extract_spo2_features(spo2_normal)
    fused_brady = fuse_features(ecg_features_brady, spo2_features_normal)
    print("Bradycardia Features:", fused_brady)
    print("ALERTS:", check_alerts(fused_brady))

    # Abnormal Case 3: Severe Hypoxemia
    ecg_normal = nk.ecg_simulate(duration=10, heart_rate=70, sampling_rate=250)
    spo2_severe = 85 + 2 * np.sin(0.5 * np.linspace(0, 10, len(ecg_normal))) + np.random.normal(0, 0.5, len(ecg_normal))
    ecg_features_normal = extract_ecg_features(ecg_normal, sampling_rate=250)
    spo2_features_severe = extract_spo2_features(spo2_severe)
    fused_severe = fuse_features(ecg_features_normal, spo2_features_severe)
    print("Severe Hypoxemia Features:", fused_severe)
    print("ALERTS:", check_alerts(fused_severe))


    # Build Dataset
    features_list = [healthy_features] * 50 + [fused_abnormal] * 50
    labels = [0] * 50 + [1] * 50
        # Build Dataset with multiple classes
    features_list = [healthy_features] * 50 + [fused_abnormal] * 50 + [fused_brady] * 50 + [fused_severe] * 50
    labels = [0] * 50 + [1] * 50 + [2] * 50 + [3] * 50  # 0=Healthy, 1=Tachy+Hypoxemia, 2=Bradycardia, 3=Severe Hypoxemia


    # Train Model
    model, acc = train_model(features_list, labels)
    print("Model trained with accuracy:", acc)

    from sklearn.metrics import classification_report, confusion_matrix

import time

# --- Evaluate Model ---
X_eval = pd.DataFrame(features_list)
y_pred = model.predict(X_eval)

print("\nClassification Report:")
print(classification_report(
    labels, y_pred,
    target_names=["Healthy", "Tachy+Hypoxemia", "Bradycardia", "Severe Hypoxemia"]
))

# --- Confusion Matrix ---
cm = confusion_matrix(labels, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Healthy", "Tachy+Hypoxemia", "Bradycardia", "Severe Hypoxemia"],
            yticklabels=["Healthy", "Tachy+Hypoxemia", "Bradycardia", "Severe Hypoxemia"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

# Generate a timestamped filename
timestamp = time.strftime("%Y%m%d-%H%M%S")
filename = f"confusion_matrix_{timestamp}.png"

# Save plot as PNG
plt.savefig(filename, dpi=300, bbox_inches="tight")

# Show plot interactively
plt.show()

print(f"Confusion matrix saved as {filename}")

import csv
import time

# --- Save metrics to CSV ---
timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
metrics_file = "model_metrics_log.csv"

# Prepare metrics row
metrics_row = [timestamp, acc["accuracy"], acc["precision"], acc["recall"], acc["f1"], filename]

# Write header if file doesn’t exist yet
write_header = not os.path.exists(metrics_file)

with open(metrics_file, mode="a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["Timestamp", "Accuracy", "Precision", "Recall", "F1", "ConfusionMatrixFile"])
    writer.writerow(metrics_row)

print(f"Metrics logged to {metrics_file}")

import joblib

# --- Reload saved model ---
clf = joblib.load("synthetic_model.pkl")
print("✅ Model reloaded successfully")

# --- Prediction function ---
def predict_case(hr, spo2, desats):
    # Derived features (same as training)
    hr_variability = abs(hr - 70)   # approximate mean HR
    spo2_risk = (100 - spo2) + (desats * 0.1)

    # Build feature vector
    X_new = [[hr, spo2, desats, hr_variability, spo2_risk]]

    # Predict using reloaded model
    return clf.predict(X_new)[0]

# Example usage
print("Prediction for HR=120, SpO2=91, desats=3 →", predict_case(120, 91, 3))




