import pandas as pd
import random
import numpy as np

def generate_case(case_type="healthy"):
    if case_type == "healthy":
        hr = np.random.normal(70, 5)
        spo2 = np.random.normal(98, 1)
        desats = 0
    elif case_type == "tachy_hypoxemia":
        hr = np.random.normal(120, 5)
        spo2 = np.random.normal(92, 2)
        desats = random.randint(1, 5)
    elif case_type == "bradycardia":
        hr = np.random.normal(40, 3)
        spo2 = np.random.normal(98, 1)
        desats = 0
    elif case_type == "severe_hypoxemia":
        hr = np.random.normal(70, 5)
        spo2 = np.random.normal(85, 3)
        desats = random.randint(100, 500)
    return {
        "heart_rate": hr,
        "spo2": spo2,
        "desaturation_events": desats,
        "label": case_type
    }

# Generate 200 synthetic cases
dataset = [
    generate_case(random.choice(["healthy", "tachy_hypoxemia", "bradycardia", "severe_hypoxemia"]))
    for _ in range(200)
]

# Save to CSV inside your data folder
df = pd.DataFrame(dataset)
df.to_csv("data/synthetic_dataset.csv", index=False)

print("✅ Synthetic dataset saved to data/synthetic_dataset.csv")
