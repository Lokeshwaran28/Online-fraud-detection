
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from scipy.stats import zscore
import numpy as np

# Load your trained model
model = xgb.XGBClassifier()
model.load_model("fraud_detection_model.json")

# Hardcoded encoders for simplicity (replace with actual mapping if needed)
location_encoder = {'New York': 0, 'Delhi': 1, 'London': 2}
device_encoder = {'Android': 0, 'iOS': 1, 'Windows': 2}
transaction_type_encoder = {'payment': 0, 'transfer': 1, 'withdrawal': 2}
ip_encoder = {'192.168.0.1': 0, '10.0.0.1': 1, '172.16.0.1': 2}
user_encoder = {'U001': 0, 'U002': 1, 'U003': 2}
txn_id_encoder = {'T001': 0, 'T002': 1, 'T003': 2}

def encode(value, encoder):
    return encoder.get(value, 0)  # default to 0 if not found

def main():
    print("🔐 Real-time Fraud Detection CLI")

    # Get input from user
    transaction_id = input("Transaction ID (e.g., T001): ")
    user_id = input("User ID (e.g., U001): ")
    timestamp_str = input("Timestamp (DD-MM-YYYY HH:MM): ")
    amount = float(input("Transaction amount: "))
    location = input("Location (New York/Delhi/London): ")
    device_type = input("Device Type (Android/iOS/Windows): ")
    ip_address = input("IP Address (e.g., 192.168.0.1): ")
    transaction_type = input("Transaction Type (payment/transfer/withdrawal): ")
    is_new_device = int(input("Is New Device? (0 or 1): "))
    is_new_location = int(input("Is New Location? (0 or 1): "))
    num_txn_last_10min = int(input("Number of transactions in last 10 min: "))

    # Feature Engineering
    timestamp = datetime.strptime(timestamp_str, "%d-%m-%Y %H:%M")
    hour = timestamp.hour
    day = timestamp.day
    month = timestamp.month
    is_large_amount = 1 if amount > 100000 else 0
    amount_zscore = (amount - 5000) / 10000  # Simplified z-score example

    # Build DataFrame
    data = pd.DataFrame([{
        'transaction_id': encode(transaction_id, txn_id_encoder),
        'user_id': encode(user_id, user_encoder),
        'amount': amount,
        'location': encode(location, location_encoder),
        'device_type': encode(device_type, device_encoder),
        'ip_address': encode(ip_address, ip_encoder),
        'transaction_type': encode(transaction_type, transaction_type_encoder),
        'is_new_device': is_new_device,
        'is_new_location': is_new_location,
        'num_txn_last_10min': num_txn_last_10min,
        'hour': hour,
        'day': day,
        'month': month,
        'is_large_amount': is_large_amount,
        'amount_zscore': amount_zscore
    }])

    # Predict
    prediction = model.predict(data)[0]

    print("
🧾 Prediction Result:")
    print("⚠️ FRAUD" if prediction == 1 else "✅ NOT FRAUD")

if __name__ == "__main__":
    main()