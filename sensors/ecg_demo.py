import neurokit2 as nk
import matplotlib.pyplot as plt

# Generate synthetic ECG signal
ecg_signal = nk.ecg_simulate(duration=10, heart_rate=70,sampling_rate=250)

# Plot the signal
plt.plot(ecg_signal)
plt.title("Synthetic ECG Signal")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.show()
