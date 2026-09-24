# Clean the data, separate features and target, and build preprocessing.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.config import CATEGORICAL_ID_COLUMNS, EXCLUDED_FEATURES




def clean_data(df):
    # Keep the raw data unchanged.
    df = df.copy()
    # Too many missing values or constant values.
    df = df.drop(columns=["weight", "payer_code", "examide", "citoglipton"])

    unknown_columns = [
        "medical_specialty",
        "race",
        "diag_1",
        "diag_2",
        "diag_3",
    ]
    # Treat missing entries as a separate category.
    df[unknown_columns] = df[unknown_columns].fillna("Unknown")
    df["gender"] = df["gender"].replace("Unknown/Invalid", "Unknown")
    # 1 means readmitted within 30 days.
    if "readmitted" in df.columns:
        df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)

    return df


def prepare_features(df):
    # Keep IDs, race and target out of X.
    X = df.drop(columns=EXCLUDED_FEATURES).copy()
    y = df["readmitted_30d"].copy()

    # Category IDs are labels, not quantities.
    text_columns = X.select_dtypes(include=["object", "string", "category"]).columns
    categorical_columns = [
        column for column in X.columns
        if column in text_columns or column in CATEGORICAL_ID_COLUMNS
    ]
    numeric_columns = [
        column for column in X.columns if column not in categorical_columns
    ]

    return X, y, numeric_columns, categorical_columns


def build_preprocessor(numeric_columns, categorical_columns):
    numeric_transformer = StandardScaler()
    # Allow categories not seen during training.
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # Build it here; fit on training data later.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_columns),
            ("categorical", categorical_transformer, categorical_columns),
        ],
        remainder="drop",
    )

    return preprocessor
