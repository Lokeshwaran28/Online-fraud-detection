from flask import Flask, request, jsonify
import joblib
import pandas as pd
from scripts.preprocess import preprocess_input  # make sure this is your custom logic

app = Flask(__name__)

# Load the model
model = joblib.load('models/fraud_detection_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    
    # Preprocess input using your script
    processed = preprocess_input(df)
    
    prediction = model.predict(processed)[0]
    return jsonify({'fraud_prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
