import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.config import (
    DATA_PATH, TARGET_COL, DROP_COLS,
    CATEGORICAL_COLS, NUMERICAL_COLS, TEST_SIZE, RANDOM_STATE
)


def load_data(path=DATA_PATH):
    """Load the Adbot ad engagement dataset."""
    # TODO: Read the CSV file and return a DataFrame
    return pd.read_csv(path)


def clean_data(df):
    """
    Clean the dataset.
    Steps:
    1. Drop rows where clicks (target) is missing
    2. Drop columns in DROP_COLS
    3. Parse date column and extract day_of_week, month, year
    4. Fill missing numerical values with median
    5. Fill missing categorical values with 'Unknown'
    """
    df = df.copy()

    # TODO: Step 1 — drop rows where clicks is missing
    if TARGET_COL in df.columns:
        df = df.dropna(subset=[TARGET_COL])

    # TODO: Step 2 — parse date before dropping it
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
   

    # TODO: Step 3 — drop DROP_COLS (includes date)
    df = df.drop(columns=[col for col in DROP_COLS if col in df.columns], errors='ignore')

    # TODO: Step 4 — fill missing numerical with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] =df[col].fillna(df[col].median())

    # TODO: Step 5 — fill missing categorical with 'Unknown'
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')

    return df


def encode_features(df):
    """
    Encode categorical columns (ad_type, currency, ID) using LabelEncoder.
    """
    df = df.copy()
    le = LabelEncoder()

    # TODO: Loop through CATEGORICAL_COLS and encode each one
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
    return df


def prepare_features(df):
    """
    Full pipeline — returns X_train, X_test, y_train, y_test.
    This is a regression problem — no stratify needed.
    """
    # TODO: Step 1 — clean
    df = clean_data(df)

    
    # TODO: Step 2 — encode features
    df = encode_features(df)
    
    # TODO: Step 3 — X = df.drop(columns=[TARGET_COL])
    #                y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]  

    # TODO: Step 4 — train_test_split (no stratify for regression)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test






#########
if __name__ == "__main__":
    df = load_data()
    print("RAW DATA LOADED:", df.shape)

    X_train, X_test, y_train, y_test = prepare_features(df)

    print("\n=== PREPROCESSING SUCCESSFUL ===")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train mean:", y_train.mean())
    print("y_test mean:", y_test.mean())