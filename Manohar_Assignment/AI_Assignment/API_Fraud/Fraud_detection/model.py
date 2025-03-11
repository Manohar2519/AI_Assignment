import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Load dataset
df = pd.read_csv("synthetic_transactions.csv")

# Select features and target
features = ["amount", "gas_fee", "transaction_count", "wallet_age"]
X = df[features]

# Train Isolation Forest Model
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
model.fit(X)

# Save the trained model
joblib.dump(model, "fraud_detection_model.pkl")

print("Model training complete and saved as fraud_detection_model.pkl")
