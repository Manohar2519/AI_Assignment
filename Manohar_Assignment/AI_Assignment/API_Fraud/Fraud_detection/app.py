from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load("fraud_detection_model.pkl")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

@app.route('/status', methods=['GET'])
def status():
    """Returns the API health status."""
    return jsonify({"status": "Model is running"})

@app.route('/detect_fraud', methods=['POST'])
def detect_fraud():
    """Detects fraud for a given transaction input."""
    try:
        data = request.get_json()
        features = ["amount", "gas_fee", "transaction_count", "wallet_age"]

        # Ensure the input has the correct fields
        if not all(key in data for key in features):
            return jsonify({"error": "Missing required fields"}), 400

        X_new = pd.DataFrame([data], columns=features)
        
        # Predict fraud probability
        prediction = model.predict(X_new)[0]
        fraud_probability = 1 if prediction == -1 else 0

        return jsonify({"fraud_probability": fraud_probability})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)