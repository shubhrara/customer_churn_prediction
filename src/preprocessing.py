import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Define column names
NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]
TARGET_COL = "Churn"

def clean_data(df):
    """
    Cleans the raw DataFrame:
    - Coerces TotalCharges to numeric, filling missing values (spaces) with 0.
    - Encodes Target variable (Churn) into binary integers.
    - Drops unnecessary columns.
    """
    df = df.copy()
    
    # Handle TotalCharges spaces
    df["TotalCharges"] = df["TotalCharges"].replace(r'^\s*$', np.nan, regex=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
    # If tenure is 0, TotalCharges is NaN, fill it with 0
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    
    # Encode Churn (Yes -> 1, No -> 0)
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
        
    # Drop customerID if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        
    return df

def get_preprocessor():
    """
    Creates and returns the ColumnTransformer for numerical and categorical features.
    """
    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, NUMERICAL_COLS),
            ("cat", cat_transformer, CATEGORICAL_COLS)
        ]
    )
    return preprocessor

def preprocess_and_split():
    """
    Loads raw data, cleans it, splits it, and saves train/test sets.
    """
    raw_path = os.path.join("data", "raw", "Telco-Customer-Churn.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_path}. Run data_loader.py first.")
        
    df = pd.read_csv(raw_path)
    df_clean = clean_data(df)
    
    # Split features and target
    X = df_clean.drop(columns=[TARGET_COL])
    y = df_clean[TARGET_COL]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save processed files
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
    
    print(f"Preprocessed train shape: {train_df.shape}")
    print(f"Preprocessed test shape: {test_df.shape}")
    print("Saved train.csv and test.csv to data/processed/")
    
    return train_df, test_df

if __name__ == "__main__":
    preprocess_and_split()
