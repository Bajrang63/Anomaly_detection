import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("artifacts/model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

# Load new transactions
data = pd.read_csv("data/creditcard.csv").sample(10)

X = data.drop("Class", axis=1)
X_scaled = scaler.transform(X)

# Predict anomalies
predictions = model.predict(X_scaled)

# Convert predictions
data["Prediction"] = predictions
data["Prediction"] = data["Prediction"].map({1: "Normal", -1: "Fraud"})

print(data[["Prediction"]])
