import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# --- Load model ---
clf = joblib.load("synthetic_model.pkl")

# --- Prediction function ---
def predict_case(hr, spo2, desats):
    hr_variability = abs(hr - 70)
    spo2_risk = (100 - spo2) + (desats * 0.1)
    X_new = [[hr, spo2, desats, hr_variability, spo2_risk]]
    return clf.predict(X_new)[0]

# --- Custom glowing alert ---
def styled_alert(message, color):
    st.markdown(
        f"""
        <div style="background-color:{color};
                    padding:15px;
                    border-radius:10px;
                    font-size:20px;
                    font-weight:bold;
                    text-align:center;
                    box-shadow:0 0 20px {color};
                    color:white;">
            {message}
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Streamlit UI ---
st.title("🩺 Biomedical Monitoring Dashboard")

st.sidebar.header("Enter Patient Data")
hr = st.sidebar.number_input("Heart Rate (bpm)", min_value=30, max_value=150, value=70)
spo2 = st.sidebar.number_input("SpO₂ (%)", min_value=70, max_value=100, value=98)
desats = st.sidebar.number_input("Desaturation Events", min_value=0, max_value=500, value=0)

# --- Prediction Output ---
result = predict_case(hr, spo2, desats)

st.subheader("🔎 Predicted Condition")
st.write(f"**Condition:** {result}")

# Glowing alerts
if "healthy" in result.lower():
    styled_alert("✅ Healthy", "#00FF00")
elif "brady" in result.lower():
    styled_alert("⚠️ Bradycardia", "#FFA500")
elif "tachy" in result.lower():
    styled_alert("🚨 Tachycardia + Hypoxemia", "#FF4B4B")
elif "hypoxemia" in result.lower():
    styled_alert("🚨 Severe Hypoxemia", "#FF0000")

# --- Patient Signal Trends ---
st.subheader("📊 Patient Signal Trends")
time = np.arange(0, 60, 1)
hr_trend = np.random.normal(hr, 2, len(time))
spo2_trend = np.random.normal(spo2, 0.5, len(time))

fig, ax = plt.subplots(2, 1, figsize=(8, 6))
ax[0].plot(time, hr_trend, color="red")
ax[0].set_title("Heart Rate Trend")
ax[0].set_ylabel("BPM")

ax[1].plot(time, spo2_trend, color="blue")
ax[1].set_title("SpO₂ Trend")
ax[1].set_ylabel("%")

st.pyplot(fig)

# --- Summary Metrics ---
st.subheader("📌 Summary Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Heart Rate", f"{hr} bpm", delta=hr-70)
col2.metric("SpO₂", f"{spo2}%", delta=spo2-98)
col3.metric("Desaturation Events", desats)

# --- Info Cards ---
st.info("ℹ️ Healthy range: HR 60–100 bpm, SpO₂ > 95%")
st.info("⚠️ Bradycardia: HR < 60 bpm")
st.info("🚨 Tachycardia + Hypoxemia: HR > 100 bpm with SpO₂ < 94%")
st.info("🚨 Severe Hypoxemia: SpO₂ < 90% with frequent desaturation events")

