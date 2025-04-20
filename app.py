from flask import Flask, request, render_template
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

app = Flask(__name__)

# Load the trained model
model = xgb.XGBClassifier()
model.load_model("fraud_detection_model.json")

# Hardcoded encoders (simplified)
location_encoder = {'New York': 0, 'Delhi': 1, 'London': 2}
device_encoder = {'Android': 0, 'iOS': 1, 'Windows': 2}
transaction_type_encoder = {'payment': 0, 'transfer': 1, 'withdrawal': 2}
ip_encoder = {'192.168.0.1': 0, '10.0.0.1': 1, '172.16.0.1': 2}
user_encoder = {'U001': 0, 'U002': 1, 'U003': 2}
txn_id_encoder = {'T001': 0, 'T002': 1, 'T003': 2}

def encode(value, encoder):
    return encoder.get(value, 0)

# Web form for users
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = request.form

        # Extract and preprocess data
        timestamp = datetime.strptime(data['timestamp'], "%Y-%m-%dT%H:%M")
        hour = timestamp.hour
        day = timestamp.day
        month = timestamp.month
        amount = float(data['amount'])

        # Prepare the data for prediction
        df = pd.DataFrame([{
            'transaction_id': encode(data['transaction_id'], txn_id_encoder),
            'user_id': encode(data['user_id'], user_encoder),
            'amount': amount,
            'location': encode(data['location'], location_encoder),
            'device_type': encode(data['device_type'], device_encoder),
            'ip_address': encode(data['ip_address'], ip_encoder),
            'transaction_type': encode(data['transaction_type'], transaction_type_encoder),
            'is_new_device': int(data['is_new_device']),
            'is_new_location': int(data['is_new_location']),
            'num_txn_last_10min': int(data['num_txn_last_10min']),
            'hour': hour,
            'day': day,
            'month': month,
            'is_large_amount': 1 if amount > 100000 else 0,
            'amount_zscore': (amount - 5000) / 10000
        }])

        # Get prediction
        prediction = model.predict(df)[0]
        result = "⚠️ FRAUD" if prediction == 1 else "✅ NOT FRAUD"
        return render_template("index.html", result=result)

    return render_template("index.html", result=None)

# API for external systems (e.g., webhooks, mobile apps, etc.)
@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()

        # Timestamp processing
        timestamp = datetime.strptime(data['timestamp'], "%Y-%m-%d %H:%M")
        hour = timestamp.hour
        day = timestamp.day
        month = timestamp.month
        amount = float(data['amount'])

        # Prepare the data for prediction
        df = pd.DataFrame([{
            'transaction_id': encode(data['transaction_id'], txn_id_encoder),
            'user_id': encode(data['user_id'], user_encoder),
            'amount': amount,
            'location': encode(data['location'], location_encoder),
            'device_type': encode(data['device_type'], device_encoder),
            'ip_address': encode(data['ip_address'], ip_encoder),
            'transaction_type': encode(data['transaction_type'], transaction_type_encoder),
            'is_new_device': int(data['is_new_device']),
            'is_new_location': int(data['is_new_location']),
            'num_txn_last_10min': int(data['num_txn_last_10min']),
            'hour': hour,
            'day': day,
            'month': month,
            'is_large_amount': 1 if amount > 100000 else 0,
            'amount_zscore': (amount - 5000) / 10000
        }])

        # Prediction result
        prediction = model.predict(df)[0]
        result = "FRAUD" if prediction == 1 else "NOT FRAUD"

        return {'prediction': result}, 200

    except Exception as e:
        return {'error': str(e)}, 400

if __name__ == '__main__':
    app.run(debug=True)
