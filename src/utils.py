# ===== FILE: src/utils.py =====
import pandas as pd
import joblib
import os

def load_data(filepath: str) -> pd.DataFrame:
    """Loads a CSV dataset into a Pandas DataFrame."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def save_model(model, filepath: str):
    """Saves a trained model/object to disk using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath: str):
    """Loads a trained model/object from disk."""
    try:
        model = joblib.load(filepath)
        return model
    except Exception as e:
        print(f"Error loading model from {filepath}: {e}")
        raise