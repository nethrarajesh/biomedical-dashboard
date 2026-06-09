import numpy as np
from scipy.signal import butter, filtfilt

# --- Low-pass filter (removes high-frequency noise) ---
def butter_lowpass_filter(data, cutoff=40, fs=250, order=5):
    nyquist = 0.5 * fs                 # Nyquist frequency = half the sampling rate
    normal_cutoff = cutoff / nyquist   # Normalize cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)  # filter coefficients
    filtered = filtfilt(b, a, data)    # apply filter forward and backward
    return filtered

# --- Moving average smoother (smooths small jitters) ---
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
