import requests

url = "http://127.0.0.1:5000/predict"

# Test cases for each condition
test_cases = [
    {"heart_rate": 70, "spo2": 98, "desaturation_events": 0},      # Healthy
    {"heart_rate": 120, "spo2": 91, "desaturation_events": 3},     # Tachy+Hypoxemia
    {"heart_rate": 40, "spo2": 98, "desaturation_events": 0},      # Bradycardia
    {"heart_rate": 70, "spo2": 85, "desaturation_events": 500},    # Severe Hypoxemia
]

# Run each test case
for case in test_cases:
    response = requests.post(url, json=case)
    print(f"Input: {case} → Prediction: {response.json()['prediction']}")
