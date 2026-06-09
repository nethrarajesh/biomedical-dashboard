def fuse_features(ecg_features, spo2_features):
    """
    Merge ECG and SpO₂ features into one dictionary.
    """
    fused = {}
    fused.update(ecg_features)
    fused.update(spo2_features)
    return fused
