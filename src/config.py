# Shared settings for preprocessing, training and prediction.
from pathlib import Path

# Keep random results reproducible.
RANDOM_STATE = 42
# Probability cutoff selected on validation data.
DECISION_THRESHOLD = 0.51

# Paths are relative to the project root.
DATA_PATH = Path("data/raw/diabetic_data.csv")


MODEL_PATH = Path("models/random_forest.joblib")

# Keep IDs, race and target out of model features.
EXCLUDED_FEATURES = [
    "encounter_id",
    "patient_nbr",
    "race",
    "readmitted",
    "readmitted_30d",
]

# These numbers represent categories, not quantities.
CATEGORICAL_ID_COLUMNS = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]
