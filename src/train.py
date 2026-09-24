# Train and save the selected Random Forest pipeline.
# Run from the project root: python -m src.train

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from src.config import DATA_PATH, MODEL_PATH, RANDOM_STATE
from src.preprocessing import build_preprocessor, clean_data, prepare_features


if __name__ == "__main__":
    df_raw = pd.read_csv(
        DATA_PATH,
        keep_default_na=False,
        na_values=["?"],
        low_memory=False,
    )
    df = clean_data(df_raw)

    # Reserve 40% of patients for validation and test.
    train_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.40,
        random_state=RANDOM_STATE,
    )
    train_idx, holdout_idx = next(
        train_splitter.split(df, groups=df["patient_nbr"])
    )
    train_df = df.iloc[train_idx].copy()
    holdout_df = df.iloc[holdout_idx].copy()

    # Split the remaining patients equally.
    holdout_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=RANDOM_STATE + 1,
    )
    validation_idx, test_idx = next(
        holdout_splitter.split(holdout_df, groups=holdout_df["patient_nbr"])
    )
    validation_df = holdout_df.iloc[validation_idx].copy()
    test_df = holdout_df.iloc[test_idx].copy()

    # A patient must belong to only one split.
    train_patients = set(train_df["patient_nbr"])
    validation_patients = set(validation_df["patient_nbr"])
    test_patients = set(test_df["patient_nbr"])

    assert train_patients.isdisjoint(validation_patients)
    assert train_patients.isdisjoint(test_patients)
    assert validation_patients.isdisjoint(test_patients)
    assert len(train_df) + len(validation_df) + len(test_df) == len(df)

    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    print(f"Test rows: {len(test_df)}")

    # Determine feature types from training data only.
    X_train, y_train, numeric_columns, categorical_columns = prepare_features(train_df)
    preprocessor = build_preprocessor(numeric_columns, categorical_columns)

    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    random_forest_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier),
    ])

    # Fit preprocessing and the model on training data.
    random_forest_pipeline.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(random_forest_pipeline, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
