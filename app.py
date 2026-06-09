from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
clf = joblib.load("synthetic_model.pkl")

def predict_case(hr, spo2, desats):
    hr_variability = abs(hr - 70)
    spo2_risk = (100 - spo2) + (desats * 0.1)
    X_new = [[hr, spo2, desats, hr_variability, spo2_risk]]
    return clf.predict(X_new)[0]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    hr = data["heart_rate"]
    spo2 = data["spo2"]
    desats = data["desaturation_events"]
    result = predict_case(hr, spo2, desats)
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
