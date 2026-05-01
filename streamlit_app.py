import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Setup Page configs
st.set_page_config(page_title="Customer Segmentation UI", layout="wide")

# ==========================================
# 🎨 CLEAN, NATIVE-FRIENDLY CSS
# ==========================================
st.markdown('<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">', unsafe_allow_html=True)

st.markdown("""
<style>
    /* Clean Centered Title */
    .main-title {
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 20px;
    }
    
    /* Subtle Icon Styling using Streamlit's native red accent */
    .header-icon {
        vertical-align: middle; 
        color: #ff4b4b; 
        font-size: 2rem; 
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Centered Title
st.markdown("<h1 class='main-title'>Customer Segmentation Portal</h1>", unsafe_allow_html=True)

@st.cache_resource
def load_ml_artifacts():
    try:
        scaler = joblib.load("models/scaler.pkl")
        kmeans = joblib.load("models/kmeans.pkl")
        pca = joblib.load("models/pca.pkl")
        df_raw = pd.read_csv("data/dataset.csv")
        return scaler, kmeans, pca, df_raw
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None, None

scaler, kmeans, pca, df_raw = load_ml_artifacts()

if not scaler:
    st.warning("Run `python train.py` first to generate models and data.")
    st.stop()

# Get clean numeric data for graphs
df_clean = df_raw.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
df_numeric = df_clean.select_dtypes(include=[np.number])

# Layout
col1, col2 = st.columns([1, 2.5])

with col1:
    st.markdown('<h2><span class="material-icons header-icon">online_prediction</span>Predict Segment</h2>', unsafe_allow_html=True)
    st.markdown("Enter customer feature values:")
    
    feature_names = scaler.feature_names_in_
    input_data = {}
    
    with st.form("prediction_form"):
        for feature in feature_names:
            min_val = float(df_numeric[feature].min())
            max_val = float(df_numeric[feature].max())
            mean_val = float(df_numeric[feature].mean())
            
            input_data[feature] = st.number_input(
                f"{feature.replace('_', ' ').title()}", 
                min_value=min_val, 
                max_value=max_val, 
                value=mean_val,
                step=1.0 
            )
            
        submitted = st.form_submit_button("Predict Cluster")

with col2:
    st.markdown('<h2><span class="material-icons header-icon">dashboard</span>Global Visualization</h2>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["Live PCA Map", "Data Distributions", "Correlation Heatmap", "Static Elbow"])
    
    with tab1:
        st.markdown("### 2D Projection of Customer Segments")
        st.info(
            "**How to read this map:**\n\n"
            "As you change the numbers on the left, click 'Predict Cluster'. You will see the Red Star "
            "jump around this map, showing exactly which cluster your new customer falls into based on their behaviors."
        )
        
        if submitted:
            # Transform background data
            scaled_bg_array = scaler.transform(df_numeric)
            scaled_background = pd.DataFrame(scaled_bg_array, columns=df_numeric.columns)
            pca_background = pca.transform(scaled_background)
            background_clusters = kmeans.predict(scaled_background)
            
            # Transform user input
            df_input = pd.DataFrame([input_data])
            scaled_input_array = scaler.transform(df_input)
            scaled_input = pd.DataFrame(scaled_input_array, columns=df_input.columns)
            pca_input = pca.transform(scaled_input)
            predicted_cluster = kmeans.predict(scaled_input)[0]
            
            st.success(f"### Predicted Segment: Cluster {predicted_cluster}")
            
            # Plot
            # Enforcing a white background for the plot so it looks good in Dark Mode
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
            ax.set_facecolor('white')
            
            sns.scatterplot(
                x=pca_background[:, 0], y=pca_background[:, 1], 
                hue=background_clusters, palette='viridis', alpha=0.3, ax=ax, s=60
            )
            
            ax.scatter(
                pca_input[:, 0], pca_input[:, 1], 
                color='red', marker='*', s=800, edgecolors='black', linewidths=2, label='You Are Here', zorder=5
            )

            plt.title('Customer Segments in PCA Space')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            st.pyplot(fig)
        else:
            st.warning("Enter values and click 'Predict Cluster' to generate the live map!")

    with tab2:
        st.markdown("### Feature Distributions (Histograms)")
        st.info(
            "**Where do you stand?**\n\n"
            "The blue mountains represent the overall database of customers. The **Dashed Red Line** represents the exact values you typed in. "
            "If your red line is in the middle of a tall peak, your customer is completely average. If it's on the far edges, they are an extreme outlier!"
        )
        
        if submitted:
            num_features = len(df_numeric.columns)
            fig_hist, axes = plt.subplots(1, num_features, figsize=(6 * num_features, 5), facecolor='white')
            
            if num_features == 1:
                axes = [axes]
                
            for i, col in enumerate(df_numeric.columns):
                axes[i].set_facecolor('white')
                sns.histplot(df_numeric[col], ax=axes[i], kde=True, color='skyblue', alpha=0.6)
                
                user_value = input_data[col]
                axes[i].axvline(x=user_value, color='red', linestyle='--', linewidth=3, label='Your Input')
                
                axes[i].set_title(f'{col} Distribution')
                axes[i].legend()
                
            plt.tight_layout()
            st.pyplot(fig_hist)
        else:
            st.warning("Click 'Predict Cluster' to see where your input lands on the histograms!")

    with tab3:
        st.markdown("### Correlation Heatmap")
        st.info("**Global Data Structure:** This shows how features are connected to each other across all customers.")
        
        fig_heat, ax_heat = plt.subplots(figsize=(8, 6), facecolor='white')
        ax_heat.set_facecolor('white')
        sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax_heat)
        plt.title("Feature Correlation Matrix")
        st.pyplot(fig_heat)

    with tab4:
        st.markdown("### Optimal K Determination (Static)")
        if os.path.exists("data/elbow_curve.png"):
            st.image(Image.open("data/elbow_curve.png"), use_column_width=True)
        else:
            st.warning("Elbow curve image not found.")