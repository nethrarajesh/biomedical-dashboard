import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.filters import butter_lowpass_filter, moving_average
from preprocessing.normalization import min_max_normalize, z_score_normalize
from preprocessing.segmentation import segment_signal
from preprocessing.baseline_correction import mean_baseline_correction, linear_baseline_correction
from features.extraction import extract_ecg_features, extract_spo2_features
from fusion.fusion import fuse_features
from models.models import train_model

# --- Healthy ECG Simulation ---
ecg_signal = nk.ecg_simulate(duration=10, heart_rate=70, sampling_rate=250)

# --- Healthy SpO₂ Simulation ---
time = np.linspace(0, 10, len(ecg_signal))
spo2_signal = 98 + 0.5 * np.sin(0.5 * time) + np.random.normal(0, 0.2, len(time))

# --- Abnormal ECG Cases ---
ecg_tachy = nk.ecg_simulate(duration=10, heart_rate=120, sampling_rate=250)   # Tachycardia
ecg_brady = nk.ecg_simulate(duration=10, heart_rate=40, sampling_rate=250)    # Bradycardia
ecg_arrhythmia = nk.ecg_simulate(duration=10, heart_rate=70, sampling_rate=250, noise=0.05)  # Irregular beats

# --- Abnormal SpO₂ Cases ---
spo2_hypoxemia = 92 + 2*np.sin(0.5*time) + np.random.normal(0, 0.5, len(time))   # Mild dips
spo2_severe = 85 + 3*np.sin(0.5*time) + np.random.normal(0, 0.7, len(time))      # Severe hypoxemia

# --- Apply filters ---
filtered_ecg = butter_lowpass_filter(ecg_signal)
smoothed_ecg = moving_average(filtered_ecg)

filtered_spo2 = butter_lowpass_filter(spo2_signal)
smoothed_spo2 = moving_average(filtered_spo2)

# --- Baseline correction ---
ecg_corrected = mean_baseline_correction(filtered_ecg)
spo2_corrected = linear_baseline_correction(filtered_spo2)

# --- Feature Extraction (Healthy) ---
ecg_features = extract_ecg_features(ecg_corrected, sampling_rate=250)
spo2_features = extract_spo2_features(spo2_corrected)
fused_features = fuse_features(ecg_features, spo2_features)

# --- Feature Extraction (Abnormal ECG) ---
fused_tachy = fuse_features(extract_ecg_features(ecg_tachy, sampling_rate=250),
                            extract_spo2_features(spo2_corrected))
fused_brady = fuse_features(extract_ecg_features(ecg_brady, sampling_rate=250),
                            extract_spo2_features(spo2_corrected))
fused_arrhythmia = fuse_features(extract_ecg_features(ecg_arrhythmia, sampling_rate=250),
                                 extract_spo2_features(spo2_corrected))

# --- Feature Extraction (Abnormal SpO₂) ---
fused_hypoxemia = fuse_features(extract_ecg_features(ecg_corrected, sampling_rate=250),
                                extract_spo2_features(spo2_hypoxemia))
fused_severe = fuse_features(extract_ecg_features(ecg_corrected, sampling_rate=250),
                             extract_spo2_features(spo2_severe))

# --- Build Dataset ---
healthy_features = [fused_features] * 50
healthy_labels = [0] * 50   # 0 = healthy

abnormal_features = (
    [fused_tachy] * 30 +
    [fused_brady] * 30 +
    [fused_arrhythmia] * 30 +
    [fused_hypoxemia] * 30 +
    [fused_severe] * 30
)
abnormal_labels = [1] * len(abnormal_features)   # 1 = abnormal

features_list = healthy_features + abnormal_features
labels = healthy_labels + abnormal_labels

# --- Train Model ---
model, acc = train_model(features_list, labels)
print("Model trained with accuracy:", acc)

from models.models import train_model

model, metrics = train_model(features_list, labels, model_type="random_forest")
print("Random Forest metrics:", metrics)

model, metrics = train_model(features_list, labels, model_type="logistic")
print("Logistic Regression metrics:", metrics)

model, metrics = train_model(features_list, labels, model_type="svm")
print("SVM metrics:", metrics)

