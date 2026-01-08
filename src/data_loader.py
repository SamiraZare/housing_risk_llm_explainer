import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_housing_data(path: str) -> pd.DataFrame:
    """
    Load raw housing dataset from CSV.
    """
    return pd.read_csv(path)

def train_test_split_xy(df, feature_cols, target_col, test_size=0.2, random_state=42):
    """
    Split dataframe into train/test NumPy arrays.
    """
    X = df[feature_cols].values
    y = df[target_col].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
