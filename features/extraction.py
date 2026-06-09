import numpy as np
import neurokit2 as nk

# --- ECG Features ---
def extract_ecg_features(ecg_signal, sampling_rate=250):
    _, info = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)
    r_peaks = info["ECG_R_Peaks"]
    rr_intervals = np.diff(r_peaks) / sampling_rate

    features = {
        "heart_rate": 60 / np.mean(rr_intervals),
        "mean_rr": np.mean(rr_intervals),
        "std_rr": np.std(rr_intervals),
    }
    return features

# --- SpO₂ Features ---
def extract_spo2_features(spo2_signal):
    features = {
        "mean_spo2": np.mean(spo2_signal),
        "min_spo2": np.min(spo2_signal),
        "std_spo2": np.std(spo2_signal),
        "desaturation_events": count_desaturation_events(spo2_signal)
    }
    return features

# --- Helper: Count Desaturation Events ---
def count_desaturation_events(spo2_signal, threshold=90, min_duration=5):
    below = spo2_signal < threshold
    events = 0
    count = 0
    for b in below:
        if b:
            count += 1
            if count >= min_duration:
                events += 1
                count = 0  # reset after event
        else:
            count = 0
    return events
