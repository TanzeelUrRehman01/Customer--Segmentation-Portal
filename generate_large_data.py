import pandas as pd
import numpy as np
import os

def generate_large_customer_data(filepath="data/dataset.csv", num_samples=5000):
    print(f"Generating {num_samples} rows of customer data...")
    np.random.seed(42)
    
    # Define 4 distinct customer segments to make the clustering interesting
    # Segment 1: Budget Shoppers (Low Income, Low Spend)
    seg1 = np.random.normal(loc=[30000, 20, 5], scale=[5000, 5, 2], size=(num_samples // 4, 3))
    
    # Segment 2: Premium Loyalists (High Income, High Spend)
    seg2 = np.random.normal(loc=[90000, 80, 25], scale=[10000, 10, 5], size=(num_samples // 4, 3))
    
    # Segment 3: Window Shoppers (Average Income, Low Spend)
    seg3 = np.random.normal(loc=[55000, 30, 8], scale=[8000, 8, 3], size=(num_samples // 4, 3))
    
    # Segment 4: Impulse Buyers (Average Income, High Spend)
    seg4 = np.random.normal(loc=[60000, 75, 15], scale=[9000, 12, 4], size=(num_samples // 4, 3))
    
    # Combine all segments
    data = np.vstack([seg1, seg2, seg3, seg4])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=["Annual_Income", "Spending_Score", "Purchases_Per_Year"])
    
    # Ensure no negative values
    df = df.clip(lower=0)
    
    # Shuffle the dataset so the clusters aren't perfectly ordered
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Save to CSV
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"✅ Success! Large dataset saved to {filepath}")

if __name__ == "__main__":
    generate_large_customer_data()