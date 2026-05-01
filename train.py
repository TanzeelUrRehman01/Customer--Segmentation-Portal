import os
import pandas as pd
import numpy as np
from src.utils import save_model
from src.preprocessing import clean_data, select_numeric_features, scale_data
from src.clustering import plot_elbow_curve, train_kmeans, assign_clusters
from src.pca import apply_pca, plot_pca_clusters

def generate_synthetic_data(filepath="data/dataset.csv"):
    """Generates a dataset if one does not exist for E2E testing."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not os.path.exists(filepath):
        print("Generating synthetic dataset...")
        np.random.seed(42)
        # Creating 3 distinct clusters artificially for demonstration
        data = np.vstack([
            np.random.normal(loc=[20, 50, 10], scale=5, size=(100, 3)),
            np.random.normal(loc=[60, 20, 80], scale=8, size=(100, 3)),
            np.random.normal(loc=[80, 80, 40], scale=6, size=(100, 3))
        ])
        df = pd.DataFrame(data, columns=["feature1", "feature2", "feature3"])
        # Add some dirty data to test preprocessing
        df.loc[0, 'feature1'] = np.nan
        df.loc[1, 'feature2'] = np.inf
        df = pd.concat([df, df.iloc[[2]]]) # Add a duplicate
        df.to_csv(filepath, index=False)
        print("Synthetic dataset created.")

def main():
    data_path = "data/dataset.csv"
    
    # 0. Setup and Data Loading
    generate_synthetic_data(data_path)
    df_raw = pd.read_csv(data_path)
    
    # 1. Data Processing & Feature Engineering
    df_clean = clean_data(df_raw)
    df_numeric = select_numeric_features(df_clean)
    df_scaled, scaler = scale_data(df_numeric)
    
    # 2. Clustering
    # Generate Elbow Curve to visualize optimal K (Saved to data/)
    plot_elbow_curve(df_scaled, max_k=10, save_path="data/elbow_curve.png")
    
    # For automation, we will set optimal K=3 based on our synthetic data profile
    OPTIMAL_K = 3 
    kmeans_model = train_kmeans(df_scaled, n_clusters=OPTIMAL_K)
    df_labeled = assign_clusters(df_scaled, kmeans_model)
    
    # 3. PCA & Visualization
    df_pca, pca_model = apply_pca(df_scaled, n_components=2)
    plot_pca_clusters(df_pca, df_labeled['Cluster'], save_path="data/pca_clusters.png")
    
    # 4. Save Models
    save_model(scaler, "models/scaler.pkl")
    save_model(kmeans_model, "models/kmeans.pkl")
    save_model(pca_model, "models/pca.pkl")
    
    print("Training Pipeline Completed Successfully! 🚀")

if __name__ == "__main__":
    main()