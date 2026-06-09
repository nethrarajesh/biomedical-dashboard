import numpy as np
from scipy.signal import detrend

# --- Simple mean subtraction ---
def mean_baseline_correction(data):
    """
    Subtracts the average value from the signal
    so it is centered around zero.
    """
    return data - np.mean(data)

# --- Linear detrend (removes slow drift) ---
def linear_baseline_correction(data):
    """
    Removes linear drift using scipy's detrend.
    """
    return detrend(data)
