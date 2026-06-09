import numpy as np

# --- Segment signal into fixed-size windows ---
def segment_signal(data, window_size, overlap=0):
    """
    Splits a signal into overlapping windows.
    
    Parameters:
    - data: input signal (list or numpy array)
    - window_size: number of samples per window
    - overlap: number of samples to overlap between windows
    
    Returns:
    - List of windows (numpy arrays)
    """
    segments = []
    step = window_size - overlap
    for start in range(0, len(data) - window_size + 1, step):
        end = start + window_size
        segments.append(data[start:end])
    return segments
