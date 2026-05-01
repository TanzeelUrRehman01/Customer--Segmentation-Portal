from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Global variables for models
scaler = None
kmeans = None
pca = None

def load_models():
    """Loads all ML artifacts at startup."""
    global scaler, kmeans, pca
    try:
        scaler = joblib.load("models/scaler.pkl")
        kmeans = joblib.load("models/kmeans.pkl")
        pca = joblib.load("models/pca.pkl")
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models. Run train.py first! Details: {e}")

# Load models upon initialization
load_models()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    if scaler and kmeans and pca:
        return jsonify({"status": "running", "models_loaded": True}), 200
    return jsonify({"status": "error", "message": "Models not loaded"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Predicts customer segment based on numeric features."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Validate inputs by casting to DataFrame (handles arbitrary number of numeric features)
        # In a strict production system, column names should be validated against scaler.feature_names_in_
        df_input = pd.DataFrame([data])
        
        # Check for missing values in input
        if df_input.isnull().values.any():
            return jsonify({"error": "Missing values in input features"}), 400
            
        # Transform and Predict
        scaled_features = scaler.transform(df_input)
        cluster = kmeans.predict(scaled_features)[0]
        
        return jsonify({
            "cluster": int(cluster),
            "interpretation": f"Customer belongs to Segment {int(cluster)}"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)