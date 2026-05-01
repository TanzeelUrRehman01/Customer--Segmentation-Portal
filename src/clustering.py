
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

def plot_elbow_curve(df_scaled, max_k=10, save_path="data/elbow_curve.png"):
    """Calculates inertia for 1 to max_k clusters and plots the Elbow Curve."""
    inertias = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        inertias.append(kmeans.inertia_)
        
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, marker='o', linestyle='--')
    plt.title('Elbow Method For Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Elbow curve saved to {save_path}")

def train_kmeans(df_scaled, n_clusters):
    """Trains a KMeans model with the specified number of clusters."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    return kmeans

def assign_clusters(df, model):
    """Assigns cluster labels to the dataset."""
    df_labeled = df.copy()
    df_labeled['Cluster'] = model.labels_
    return df_labeled