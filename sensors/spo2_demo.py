import matplotlib.pyplot as plt
import numpy as np

# --- Time axis (10 seconds) ---
time = np.linspace(0, 10, 1000)

# --- SpO₂ Simulation ---
spo2_signal = 98 + 0.5 * np.sin(0.5 * time) + np.random.normal(0, 0.2, len(time))

# --- Plot ---
plt.plot(time, spo2_signal, color="blue")
plt.title("Synthetic SpO₂ Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("SpO₂ (%)")
plt.show()
