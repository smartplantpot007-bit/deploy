from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your saved model
model = joblib.load("plant_model.pkl")

@app.route('/')
def home():
    return "🌿 Plant Health Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Extract feature values
        features = np.array([
            data['Soil_Moisture'],
            data['Ambient_Temperature'],
            data['Soil_Temperature'],
            data['Humidity']
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
