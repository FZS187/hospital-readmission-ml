# Predict readmission using the saved pipeline and locked threshold.
# Run from the project root: python -m src.predict

import joblib
import pandas as pd

from src.config import DATA_PATH, DECISION_THRESHOLD, EXCLUDED_FEATURES, MODEL_PATH
from src.preprocessing import clean_data


def predict_readmission(df):
    cleaned_df = clean_data(df)
    X = cleaned_df.drop(columns=EXCLUDED_FEATURES, errors="ignore")

    # The saved pipeline handles preprocessing.
    model = joblib.load(MODEL_PATH)
    probability = model.predict_proba(X)[:, 1]
    predictions = (probability >= DECISION_THRESHOLD).astype(int)

    # Keep results aligned with the input rows.
    return pd.DataFrame({
        "readmission_probability": probability,
        "predicted_readmission_30d": predictions,
    }, index=df.index)


if __name__ == "__main__":
    df_raw = pd.read_csv(
        DATA_PATH,
        keep_default_na=False,
        na_values=["?"],
        low_memory=False,
    )
    sample = df_raw.head(5)
    result = predict_readmission(sample)
    print(result)
