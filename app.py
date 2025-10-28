from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow requests from Android app

# Load your trained Random Forest model (.pkl)
try:
    model = joblib.load("plant_model.pkl")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Home route for testing
@app.route('/')
def home():
    return "🌿 Plant Health Prediction API is running!"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Extract and validate input features
        features = [
            data.get('Soil_Moisture'),
            data.get('Ambient_Temperature'),
            data.get('Soil_Temperature'),
            data.get('Humidity')
        ]

        if None in features:
            return jsonify({'error': 'Missing one or more input fields'}), 400

        # Convert to numpy array for model
        input_data = np.array(features).reshape(1, -1)

        # Get prediction
        prediction = model.predict(input_data)[0]

        return jsonify({'prediction': str(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Use Render’s dynamic port
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
