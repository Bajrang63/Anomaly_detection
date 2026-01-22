import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("artifacts/model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🚨",
    layout="wide"
)

st.title("🚀 Real-Time Fraud Detection (Anomaly Detection)")
st.markdown("**Isolation Forest | Unsupervised ML | Production-Grade Project**")

# Sidebar
st.sidebar.header("⚙️ Options")
option = st.sidebar.selectbox(
    "Choose Input Method",
    ["Upload CSV", "Manual Transaction"]
)

# =============================
# 📤 CSV Upload Mode
# =============================
if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload Credit Card Transactions CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "Class" in df.columns:
            X = df.drop("Class", axis=1)
        else:
            X = df

        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        scores = model.decision_function(X_scaled)

        df["Prediction"] = predictions
        df["Prediction"] = df["Prediction"].map({1: "Normal", -1: "Fraud"})
        df["RiskScore"] = scores

        st.subheader("🔍 Prediction Results")
        st.dataframe(df.head(20))

        # Visualization
        st.subheader("📊 Fraud vs Normal Distribution")
        fig, ax = plt.subplots()
        df["Prediction"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

# =============================
# ✍️ Manual Entry Mode
# =============================
if option == "Manual Transaction":
    st.subheader("✍️ Enter Transaction Details")

    input_data = []
    cols = st.columns(4)

    for i in range(30):
        with cols[i % 4]:
            val = st.number_input(f"Feature V{i}" if i > 0 else "Time", value=0.0)
            input_data.append(val)

    amount = st.number_input("Amount", value=0.0)
    input_data.append(amount)

    if st.button("🔍 Predict Fraud"):
        X_scaled = scaler.transform([input_data])
        prediction = model.predict(X_scaled)[0]
        score = model.decision_function(X_scaled)[0]

        if prediction == -1:
            st.error(f"🚨 Fraudulent Transaction Detected (Risk Score: {score:.4f})")
        else:
            st.success(f"✅ Legitimate Transaction (Risk Score: {score:.4f})")
