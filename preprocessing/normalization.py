import numpy as np

# --- Min-Max Normalization ---
def min_max_normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# --- Z-Score Normalization ---
def z_score_normalize(data):
    return (data - np.mean(data)) / np.std(data)
