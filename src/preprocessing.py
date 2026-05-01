import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values, infinite values, and duplicates."""
    df_clean = df.copy()
    
    # Replace infinite values with NaN
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop rows with missing values
    df_clean.dropna(inplace=True)
    
    # Drop duplicates
    df_clean.drop_duplicates(inplace=True)
    
    return df_clean

def select_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Selects only numeric columns for modeling."""
    return df.select_dtypes(include=[np.number])

def scale_data(df: pd.DataFrame):
    """Applies StandardScaler to the dataframe."""
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
    return df_scaled, scaler