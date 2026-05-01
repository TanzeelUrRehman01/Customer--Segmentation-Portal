import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
import os

def apply_pca(df_scaled, n_components=2):
    """Reduces dataset dimensions using PCA."""
    pca = PCA(n_components=n_components, random_state=42)
    pca_array = pca.fit_transform(df_scaled)
    
    columns = [f'PCA_Component_{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(pca_array, columns=columns, index=df_scaled.index)
    
    return df_pca, pca

def plot_pca_clusters(df_pca, cluster_labels, save_path="data/pca_clusters.png"):
    """Visualizes the 2D PCA dataset colored by cluster labels."""
    df_plot = df_pca.copy()
    df_plot['Cluster'] = cluster_labels
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='PCA_Component_1', 
        y='PCA_Component_2', 
        hue='Cluster', 
        palette='viridis', 
        data=df_plot, 
        s=100, 
        alpha=0.7
    )
    plt.title('Customer Segments in 2D PCA Space')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Cluster')
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"PCA scatter plot saved to {save_path}")