import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Drop target column for anomaly detection
X = df.drop("Class", axis=1)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest
model = IsolationForest(
    n_estimators=300,
    contamination=0.0017,  # real fraud rate
    random_state=42,
    n_jobs=-1
)

model.fit(X_scaled)

# Save model
joblib.dump(model, "artifacts/model.pkl")
joblib.dump(scaler, "artifacts/scaler.pkl")

print("✅ Anomaly detection model trained & saved")
