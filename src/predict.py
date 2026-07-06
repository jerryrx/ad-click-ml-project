import pickle
import pandas as pd
import numpy as np

from src.preprocessing import clean_data, encode_features
from src.config import MODEL_PATH, TARGET_COL


def load_model(path=MODEL_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)


def align_columns(df, model):
    """
    Ensure prediction data matches training feature order.
    """
    if hasattr(model, "feature_names_in_"):
        expected_cols = model.feature_names_in_

        # Add missing columns as 0
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0

        # Keep correct order only
        df = df[expected_cols]

    return df


def predict_single(model, input_dict):
    """
    Predict number of clicks for a single ad record.
    Returns dict with predicted_clicks (int, rounded).
    """

    # TODO: DataFrame, clean, encode, align, predict

    df = pd.DataFrame([input_dict])

    # Clean + encode (same pipeline as training)
    df = clean_data(df)
    df = encode_features(df)

    # Remove target if accidentally present
    df = df.drop(columns=[TARGET_COL], errors='ignore')

    # Align columns with training data
    df = align_columns(df, model)

    # Predict
    prediction = model.predict(df)[0]

    # TODO: return {'predicted_clicks': int(round(model.predict(df)[0]))}
    return {
        "predicted_clicks": int(round(prediction))
    }


def predict_batch(model, df):
    """
    Predict clicks for a batch of ad records.
    Returns df with Predicted_Clicks column added.
    """

    # TODO: copy, clean, encode, align, predict

    df = df.copy()

    # Clean + encode
    df = clean_data(df)
    df = encode_features(df)

    # Remove target if present
    df = df.drop(columns=[TARGET_COL], errors='ignore')

    # Align columns
    df = align_columns(df, model)

    # Predict
    predictions = model.predict(df)

    # TODO: result['Predicted_Clicks'] = predictions.round().astype(int)
    df["Predicted_Clicks"] = np.round(predictions).astype(int)

    return df


if __name__ == "__main__":
    print("🚀 Running prediction test...")

    model = load_model()

    sample_input = {
        "impressions": 120,
        "cost": 2500,
        "ad_type": "banner",
        "currency": "USD",
        "date": "2024-01-01"
    }

    result = predict_single(model, sample_input)

    print("\n🔮 Single Prediction Result:")
    print(result)

    print("\n🚀 Batch Test:")

    df = pd.DataFrame([sample_input, sample_input])
    batch_result = predict_batch(model, df)

    print(batch_result.head())