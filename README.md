# 30-Day Hospital Readmission Prediction

Machine learning project for predicting whether a patient with diabetes will be readmitted to hospital within 30 days after discharge.

The project uses the **Diabetes 130-US Hospitals for Years 1999–2008** dataset and focuses on building an interpretable binary classification workflow with careful patient-level data splitting, leakage prevention, model comparison, validation-based threshold selection, and final evaluation on an independent test set.

The final model is a **class-balanced Random Forest** using a decision threshold of **0.51**.

> **Disclaimer:** This project is an educational and research prototype. It is not clinically validated and must not be used for medical diagnosis, treatment decisions, or patient-care recommendations.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/FZS187/hospital-readmission-ml.git
cd hospital-readmission-ml
```

Run all following commands from the `30-Day-Hospital-Readmission-Prediction/` project directory. The dataset is already included in the repository.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the final model

```bash
python -m src.train
```

This command:

- loads and cleans the dataset;
- performs the patient-level train/validation/test split;
- builds the preprocessing pipeline;
- trains the selected balanced Random Forest on the training set;
- saves the complete fitted pipeline to:

```text
models/random_forest.joblib
```

### 5. Generate predictions

```bash
python -m src.predict
```

The prediction script:

- loads the saved preprocessing + Random Forest pipeline;
- cleans the input data using the same preprocessing rules;
- calculates the probability of 30-day readmission;
- applies the locked decision threshold of **0.51**;
- returns both the predicted probability and binary prediction.

### 6. Reproduce the analysis

Open:

```text
notebooks/analysis.ipynb
```

and run the notebook from top to bottom to reproduce the full analysis, model comparison, threshold selection, final evaluation, feature interpretation, error analysis, and subgroup analysis.

## Project Structure

```text
hospital-readmission-ml/
├── README.md                      # Project overview, methodology, results and instructions
├── requirements.txt               # Python dependencies required to run the project
├── .gitignore                     # Files and folders excluded from Git
│
├── data/
│   ├── README.md                  # Information about the dataset and its source
│   └── raw/
│       ├── diabetic_data.csv      # Original Diabetes 130-US Hospitals dataset
│       └── IDS_mapping.csv        # Mapping for encoded categorical IDs
│
├── notebooks/
│   └── analysis.ipynb             # Full ML analysis, experiments and interpretation
│
├── src/
│   ├── __init__.py                # Marks src as a Python package
│   ├── config.py                  # Shared constants, paths, random seed and threshold
│   ├── preprocessing.py           # Data cleaning and preprocessing pipeline
│   ├── train.py                   # Trains and saves the final Random Forest pipeline
│   └── predict.py                 # Loads the saved model and generates predictions
│
├── models/
│   └── random_forest.joblib       # Serialized final preprocessing + model pipeline
│
└── reports/
    ├── figures/                   # Saved plots and visualizations
    └── metrics.json               # Final model metrics
```

The project separates **experimentation and analysis** from **reusable model code**.

The notebook documents how the final modelling decisions were reached, while the `src/` directory contains the simplified and reproducible implementation of the selected pipeline.

## Problem Definition

The goal is to predict **30-day hospital readmission** for patients with diabetes.

The original `readmitted` target contains three categories:

- `<30` — the patient was readmitted within 30 days
- `>30` — the patient was readmitted after more than 30 days
- `NO` — no recorded readmission

For this project, the target is converted into a binary variable:

```python
readmitted_30d = 1  # readmitted within 30 days
readmitted_30d = 0  # readmitted after 30 days or not readmitted
```

Because only about 11% of encounters belong to the positive class, accuracy is not used as the primary metric. Model comparison focuses mainly on **PR-AUC, ROC-AUC, precision, recall, and F1 score**.

## Dataset

The project uses the **Diabetes 130-US Hospitals for Years 1999–2008** dataset.

The dataset contains:

- **101,766 hospital encounters**
- **71,518 unique patients**
- data from **130 US hospitals**
- encounters recorded between **1999 and 2008**

A patient may appear in the dataset more than once. To prevent data leakage, encounters belonging to the same patient were kept within the same dataset split.

The data was divided at the patient level into approximately:

| Split      | Encounters | Patients | Positive rate |
| ---------- | ---------: | -------: | ------------: |
| Train      |     61,326 |   42,910 |        11.19% |
| Validation |     20,187 |   14,304 |        10.81% |
| Test       |     20,253 |   14,304 |        11.42% |

The training set was used to fit preprocessing and models, the validation set was used for model and threshold selection, and the test set was reserved for the final evaluation.

## Methodology

### Data Cleaning

The raw CSV was loaded so that only `"?"` values were interpreted as missing values. Text values such as `"None"` in `max_glu_serum` and `A1Cresult` were preserved because they indicate that the corresponding test was not performed.

The following columns were removed:

- `weight` — approximately 96.9% missing values
- `payer_code` — approximately 39.6% missing values and limited generalizability
- `examide` — constant feature
- `citoglipton` — constant feature

Missing values in `medical_specialty`, `race`, `diag_1`, `diag_2`, and `diag_3` were replaced with `"Unknown"`.

No rows were removed during cleaning.

### Leakage Prevention

Several precautions were used to reduce data leakage:

- `patient_nbr` was used only for patient-level splitting and was excluded from the model features.
- `encounter_id` was excluded because it is only an identifier.
- the original `readmitted` column was excluded because it directly defines the target.
- `readmitted_30d` was used only as the target.
- `race` was excluded from the model features and retained only for subgroup analysis.
- preprocessing was fitted only on the training set.
- model and threshold selection were performed only on the validation set.
- the test set was used only once for final evaluation.

### Preprocessing

Numerical features were standardized using `StandardScaler`.

Categorical features were encoded using:

```python
OneHotEncoder(handle_unknown="ignore")
```

The following numerically encoded identifier columns were also treated as categorical:

- `admission_type_id`
- `discharge_disposition_id`
- `admission_source_id`

Diagnostic codes (`diag_1`, `diag_2`, `diag_3`) were treated as categorical variables rather than continuous numbers.

All preprocessing steps were combined with the classifier using a Scikit-learn `Pipeline`, ensuring that preprocessing was learned only from the training data.

## Models and Model Selection

Four validation baselines/models were compared:

| Model                          | Accuracy | Precision |    Recall |        F1 |   ROC-AUC |    PR-AUC |
| ------------------------------ | -------: | --------: | --------: | --------: | --------: | --------: |
| Dummy baseline                 |    0.892 |     0.000 |     0.000 |     0.000 |     0.500 |     0.108 |
| Logistic Regression            |    0.892 |     0.494 |     0.018 |     0.034 |     0.664 |     0.209 |
| Logistic Regression (balanced) |    0.662 |     0.172 |     0.559 |     0.263 |     0.665 | **0.211** |
| Random Forest (balanced)       |    0.644 |     0.174 | **0.615** | **0.272** | **0.681** |     0.203 |

The Dummy baseline achieved high accuracy because approximately 89% of encounters belong to the negative class, but it failed to identify any 30-day readmissions.

Both Logistic Regression models learned meaningful predictive signal compared with the baseline. The class-balanced version substantially increased recall, while ROC-AUC and PR-AUC remained almost unchanged, showing that class weighting mainly changed the operating point rather than the underlying ranking performance.

The balanced Random Forest achieved the highest validation **recall, F1 score, and ROC-AUC**, although its PR-AUC was slightly lower than that of the balanced Logistic Regression.

Based on the overall validation trade-off, the balanced Random Forest was selected as the final model.

### Decision Threshold

The default classification threshold was not accepted automatically. Different thresholds were evaluated using the validation set only.

A threshold of **0.51** was selected for the final Random Forest:

- Precision: **0.184**
- Recall: **0.541**
- F1 score: **0.275**

This threshold provided the strongest F1 score among the evaluated Random Forest operating points while retaining a relatively high recall.

After the model and threshold were locked, the test set was used once for final evaluation.

## Final Test Results

After the Random Forest model and decision threshold were selected using the validation set, the final locked pipeline was evaluated once on the independent test set.

The final test results were:

| Metric    | Score |
| --------- | ----: |
| Accuracy  | 0.682 |
| Precision | 0.182 |
| Recall    | 0.510 |
| F1 score  | 0.268 |
| ROC-AUC   | 0.658 |
| PR-AUC    | 0.199 |

The confusion matrix was:

|          | Predicted 0 | Predicted 1 |
| -------- | ----------: | ----------: |
| Actual 0 |      12,639 |       5,302 |
| Actual 1 |       1,133 |       1,179 |

The model correctly identified **1,179 of 2,312 actual 30-day readmissions**, corresponding to a recall of approximately **51%**.

The cost of this higher sensitivity was a substantial number of false-positive predictions. Out of 6,481 encounters classified as potential 30-day readmissions, 1,179 were true positives.

Validation and test performance remained relatively similar. For example, validation F1 decreased only from **0.275 to 0.268**, while PR-AUC changed from **0.203 to 0.199**, suggesting that the selected model generalized reasonably to unseen patients.

### Class Imbalance

One of the main challenges in this dataset is the strong class imbalance. Only approximately **11% of encounters resulted in a readmission within 30 days**, while almost 89% belonged to the negative class.

This makes the positive readmission class substantially harder to identify and explains why accuracy alone is misleading: a model predicting almost every encounter as non-readmission can achieve high accuracy while providing almost no useful readmission detection.

The trained models nevertheless improved substantially over the Dummy baseline. Validation PR-AUC increased from approximately **0.108** for the baseline to around **0.20** for the trained models, showing that meaningful predictive signal was learned.

However, the model's modest PR-AUC and low precision show that the available features do not reliably distinguish readmitted from non-readmitted encounters. These results do not establish class imbalance as the main cause of the limited performance. Future work should first refine the eligible patient cohort, including the exclusion of encounters where readmission is impossible, and then evaluate whether richer clinical features improve prediction.

## Model Interpretation and Error Analysis

### Feature Importance

Random Forest feature importance showed that previous healthcare utilization was highly relevant to the model.

The most important individual transformed feature was:

- `number_inpatient` — previous inpatient admissions

Other highly ranked features included:

- `number_emergency`
- several `discharge_disposition_id` categories
- `number_diagnoses`
- `time_in_hospital`
- `num_medications`
- `number_outpatient`

These results suggest that previous hospital utilization and discharge information contributed strongly to the model's decisions.

Feature importance does not establish causality and does not indicate whether a feature increases or decreases readmission risk.

### Error Analysis

The final test predictions contained:

- **1,179 True Positives**
- **5,302 False Positives**
- **1,133 False Negatives**
- **12,639 True Negatives**

A notable pattern appeared when comparing previous inpatient admissions.

Patients in the True Positive group had, on average, approximately **1.93 previous inpatient admissions**, while patients in the False Negative group had approximately **0.41**.

This suggests that the model was better at identifying readmission among patients with a stronger history of previous hospital use, while actual readmissions among patients with less previous utilization were more difficult to detect.

### Subgroup Analysis by Race

Race was excluded from the model features and used only for post-hoc subgroup evaluation.

Performance was similar for the two largest race groups:

- Caucasian: recall **0.518**, F1 **0.269**
- African American: recall **0.505**, F1 **0.277**

Smaller groups showed greater variation, but their sample sizes were substantially lower, making those estimates less stable.

This descriptive subgroup analysis does not establish that the model is fair or unbiased.

## Limitations

Despite learning a meaningful predictive signal and generalizing reasonably from validation to the held-out test set, the project has several limitations that should be considered when interpreting the results.

### Historical Dataset

The dataset contains hospital encounters collected between **1999 and 2008** across US hospitals. Clinical practice, coding standards, treatment pathways, and patient populations may have changed since then. Therefore, the reported performance should be interpreted as performance on this historical dataset rather than as an estimate of performance in a modern healthcare system.

### Cohort Definition

Feature interpretation revealed that `discharge_disposition_id` was important to the Random Forest. One of these categories, `discharge_disposition_id = 11`, represents patients who expired during the hospital encounter.

These patients are not realistically at risk of subsequent 30-day readmission. A stronger future version of the project should define readmission eligibility before splitting the data and exclude encounters where future readmission is impossible or clinically inappropriate.

Identifying this issue through model interpretation is also an important result of the analysis, as it highlights how model explainability can reveal weaknesses in cohort construction.

### False-Positive Trade-off

The selected threshold was designed to retain useful recall rather than maximize precision. As a result, the model identifies approximately half of actual readmissions but also produces a substantial number of false-positive alerts.

For an educational decision-support prototype, this demonstrates the practical trade-off between missing high-risk patients and generating additional alerts. In a real clinical setting, the threshold would need to be selected according to the cost and clinical consequences of both types of error.

### Generalizability and Clinical Use

The model was evaluated using a patient-level held-out test set, which provides a more realistic estimate than evaluating on encounters from patients already seen during training.

However, the model has not been externally or prospectively validated on data from other hospitals or time periods.

This project should therefore be viewed as an **educational and research decision-support prototype**, not as a clinically validated system.

### Future Improvements

Potential improvements include:

- refining the eligible patient cohort;
- adding richer and more recent clinical data;
- improving feature engineering;
- evaluating probability calibration;
- using cross-validation for more robust model comparison;
- testing alternative interpretable models;
- validating performance on an external dataset.
