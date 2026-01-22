🚀 Real-Time Credit Card Fraud Detection (Anomaly Detection)
📌 Project Overview

This project implements a production-grade anomaly detection system to identify fraudulent credit card transactions in real time using unsupervised machine learning.

Instead of learning known fraud patterns, the system learns normal transaction behavior and flags unusual (anomalous) transactions, which closely matches real-world fraud detection scenarios.

🧠 Why Anomaly Detection?

Fraud data is highly imbalanced (fraud < 0.2%)

New fraud patterns appear frequently

Supervised models fail on unseen fraud types

👉 Isolation Forest is used to detect anomalies without relying on labels.
CSV / Transaction Input
        ↓
Feature Scaling (StandardScaler)
        ↓
Isolation Forest Model
        ↓
Anomaly Score (Risk Score)
        ↓
Fraud / Normal Prediction
        ↓
Interactive Streamlit Dashboard
📊 Dataset

Dataset Used: Credit Card Fraud Detection Dataset (European Cardholders)

284,807 transactions

492 fraud cases (0.17%)

Features: Time, V1–V28 (PCA-transformed), Amount

Target column Class is used only for evaluation, not for training

⚠️ Dataset is NOT included in this repository due to GitHub’s 100 MB file size limit.

🔗 Download from Kaggle:
https://www.kaggle.com/mlg-ulb/creditcardfraud

📂 After download, place the file here:data/creditcard.csv
📁 Project Structure:
creditcard_anomaly_detection/
│
├── training/
│   └── train_model.py        # Model training (Isolation Forest)
│
├── app/
│   ├── predict.py            # Inference logic
│   └── dashboard.py          # Streamlit frontend
│
├── artifacts/
│   ├── model.pkl             # Trained model
│   └── scaler.pkl            # Feature scaler
│
├── data/
│   └── creditcard.csv        # Dataset (ignored in git)
│
├── requirements.txt
└── README.md

1️⃣ Clone Repository
git clone https://github.com/Bajrang63/Anomaly_detection.git
cd Anomaly_detection
2️⃣ Install Dependencies
python training/train_model.py
3️⃣ Train the Model
python training/train_model.py
🖥️ Run Interactive Dashboard
streamlit run app/dashboard.py

Features:

CSV upload for batch fraud detection

Manual transaction input

Real-time anomaly prediction

Risk score visualization

Fraud vs Normal distribution chart

🔍 Model Details

Algorithm: Isolation Forest

Learning Type: Unsupervised

Contamination: Set to real-world fraud ratio

Scaling: StandardScaler
Output:

Normal (1)

Fraud (-1)

Continuous risk score

🏆 Key Highlights

✔ Real-world imbalanced dataset
✔ Unsupervised anomaly detection
✔ Production-ready structure
✔ Interactive frontend
✔ Scalable & extensible design

🚀 Future Enhancements

SHAP-based explainability

Kafka real-time streaming

Autoencoder + Isolation Forest ensemble

FastAPI backend

Docker & cloud deployment
