import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

import os


# ==============================================================
# STEP 1 - DEDICATED CVD PREDICTION MODEL
# ==============================================================

print("=" * 70)
print("STEP 1 - DEDICATED CVD PREDICTION MODEL")
print("=" * 70)


# ==============================================================
# 1. LOAD DATASET
# ==============================================================

input_file = "output/cleaned_dataset.csv"

df = pd.read_csv(input_file)

print("\nDataset loaded:")
print("Shape:", df.shape)

print("\nOriginal columns:")
print(df.columns.tolist())


# ==============================================================
# 2. DEFINE TARGET
# ==============================================================

target_col = "Cardiovascular_Disease"

if target_col not in df.columns:
    raise ValueError(
        f"Target column '{target_col}' was not found."
    )


# ==============================================================
# 3. DEFINE CVD PREDICTOR FEATURES
# ==============================================================

cvd_features = [

    # ----------------------------------------------------------
    # Hemodynamic & Lipid Profile
    # ----------------------------------------------------------

    "SYS_BP",
    "DIA_BP",
    "High_BP",
    "HDL",
    "Total_Cholesterol",
    "High_Cholesterol",
    "Low_HDL",

    # ----------------------------------------------------------
    # Metabolic Indicators
    # ----------------------------------------------------------

    "BMI",
    "Waist",
    "Weight",
    "Height",
    "Obesity",
    "Diabetes",

    # ----------------------------------------------------------
    # Lifestyle & Risk Composite
    # ----------------------------------------------------------

    "Smoking_100_Cigarettes",
    "Smoking_Risk",
    "Risk_Factor_Count"
]


print("\nCVD predictor features:")

for feature in cvd_features:
    print("-", feature)

print(
    "\nNumber of CVD predictor features:",
    len(cvd_features)
)


# ==============================================================
# 4. VERIFY REQUIRED COLUMNS
# ==============================================================

required_for_model = cvd_features + [target_col]

missing_columns = [
    col
    for col in required_for_model
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        "The following required columns are missing:\n"
        + "\n".join(missing_columns)
    )

print("\nAll required CVD model columns are present.")


# ==============================================================
# 5. VERIFY THYROID COLUMNS
# ==============================================================
#
# These columns are NOT used by the CVD model.
#
# They are preserved so that Step 3 can later investigate:
#
#       Thyroid status
#              ↓
#       CVD model probability
#              ↓
#       CVD model prediction
#
# ==============================================================

thyroid_columns = [
    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO",
    "Thyroid_Diagnosis",
    "Abnormal_TSH"
]


missing_thyroid_columns = [
    col
    for col in thyroid_columns
    if col not in df.columns
]


if missing_thyroid_columns:

    print("\nWARNING:")
    print(
        "The following thyroid columns are missing from "
        "the cleaned dataset:"
    )

    for col in missing_thyroid_columns:
        print("-", col)

    print(
        "\nStep 3 will NOT be possible until these columns "
        "exist in cleaned_dataset.csv."
    )

else:

    print("\nAll thyroid columns are available.")

    print(
        "These columns will NOT be used for CVD prediction."
    )

    print(
        "They will only be preserved for Step 3 analysis."
    )


# ==============================================================
# 6. CREATE MODEL DATASET
# ==============================================================

model_columns = cvd_features + [target_col]

model_df = df[model_columns].copy()

print("\n" + "=" * 70)
print("CVD MODEL DATASET")
print("=" * 70)

print("\nModel dataset shape:")
print(model_df.shape)

print("\nMissing values:")
print(model_df.isnull().sum())


# ==============================================================
# 7. TARGET DISTRIBUTION
# ==============================================================

y = model_df[target_col].astype(int)

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget proportion:")
print(y.value_counts(normalize=True))


# ==============================================================
# 8. IDENTIFY CATEGORICAL AND NUMERICAL FEATURES
# ==============================================================

categorical_features = [
    "Diabetes",
    "Smoking_100_Cigarettes"
]


numerical_features = [
    col
    for col in cvd_features
    if col not in categorical_features
]


print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


# ==============================================================
# 9. PREPROCESSING
# ==============================================================

numeric_transformer = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[

        (
            "num",
            numeric_transformer,
            numerical_features
        ),

        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ==============================================================
# 10. TRAIN / TEST SPLIT
# ==============================================================

X = model_df[cvd_features]

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# ==============================================================
# 11. LOGISTIC REGRESSION
# ==============================================================

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)


logistic_model = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


logistic_model.fit(
    X_train,
    y_train
)


print("Logistic Regression training complete.")


# ==============================================================
# 12. RANDOM FOREST
# ==============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)


random_forest_model = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            )
        )
    ]
)


random_forest_model.fit(
    X_train,
    y_train
)


print("Random Forest training complete.")


# ==============================================================
# 13. MODEL EVALUATION FUNCTION
# ==============================================================

def evaluate_model(
    model,
    model_name,
    X_test,
    y_test
):

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    return {

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "ROC_AUC": roc_auc
    }


# ==============================================================
# 14. EVALUATE MODELS
# ==============================================================

logistic_results = evaluate_model(
    logistic_model,
    "Logistic Regression",
    X_test,
    y_test
)


random_forest_results = evaluate_model(
    random_forest_model,
    "Random Forest",
    X_test,
    y_test
)


# ==============================================================
# 15. MODEL COMPARISON
# ==============================================================

comparison_df = pd.DataFrame(

    [
        logistic_results,
        random_forest_results
    ]
)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(comparison_df)


# ==============================================================
# 16. SELECT BEST MODEL
# ==============================================================

best_model_name = comparison_df.loc[
    comparison_df["ROC_AUC"].idxmax(),
    "Model"
]


if best_model_name == "Logistic Regression":

    best_model = logistic_model

else:

    best_model = random_forest_model


print(
    "\nBest model based on ROC-AUC:"
)

print(best_model_name)


# ==============================================================
# 17. GENERATE CVD PROBABILITIES
# ==============================================================

print("\n" + "=" * 70)
print("GENERATING CVD PROBABILITIES")
print("=" * 70)


test_probabilities = best_model.predict_proba(
    X_test
)[:, 1]


# Initial threshold only.
#
# Step 2 will perform threshold analysis.
#
initial_threshold = 0.50


test_predictions = (
    test_probabilities >= initial_threshold
).astype(int)


# ==============================================================
# 18. IMPORTANT:
#     PRESERVE ALL ORIGINAL TEST-SET INFORMATION
# ==============================================================

#
# This is the key correction.
#
# The model uses ONLY cvd_features.
#
# However, the output keeps:
#
#   SEQN
#   CVD features
#   thyroid features
#   original questionnaire CVD
#   all other variables
#
# This allows Step 3 to analyze thyroid status without
# putting thyroid variables into the CVD prediction model.
#


test_indices = X_test.index


test_output = df.loc[
    test_indices
].copy()


# ==============================================================
# 19. ADD MODEL OUTPUTS
# ==============================================================

test_output["CVD_Probability"] = (
    test_probabilities
)


test_output["CVD_Prediction_0.50"] = (
    test_predictions
)


# ==============================================================
# 20. SORT TEST OUTPUT BY SEQN IF AVAILABLE
# ==============================================================

if "SEQN" in test_output.columns:

    test_output = test_output.sort_values(
        by="SEQN"
    )


# ==============================================================
# 21. DISPLAY PREDICTION DISTRIBUTION
# ==============================================================

print("\nInitial threshold:", initial_threshold)


print("\nCVD prediction distribution:")

print(
    test_output[
        "CVD_Prediction_0.50"
    ].value_counts()
)


print("\nCVD prediction proportion:")

print(
    test_output[
        "CVD_Prediction_0.50"
    ].value_counts(
        normalize=True
    )
)


# ==============================================================
# 22. VERIFY THYROID COLUMNS IN OUTPUT
# ==============================================================

print("\n" + "=" * 70)
print("VERIFYING THYROID COLUMNS IN OUTPUT")
print("=" * 70)


for column in thyroid_columns:

    if column in test_output.columns:

        print(
            f"[OK] {column}"
        )

    else:

        print(
            f"[MISSING] {column}"
        )


# ==============================================================
# 23. CREATE OUTPUT DIRECTORY
# ==============================================================

os.makedirs(
    "output",
    exist_ok=True
)


# ==============================================================
# 24. SAVE MODEL COMPARISON
# ==============================================================

comparison_file = (
    "output/cvd_model_comparison.csv"
)


comparison_df.to_csv(
    comparison_file,
    index=False
)


# ==============================================================
# 25. SAVE TEST PREDICTIONS
# ==============================================================

prediction_file = (
    "output/cvd_test_predictions.csv"
)


test_output.to_csv(
    prediction_file,
    index=False
)


# ==============================================================
# 26. VERIFY SAVED FILE
# ==============================================================

print("\n" + "=" * 70)
print("VERIFYING SAVED OUTPUT")
print("=" * 70)


saved_df = pd.read_csv(
    prediction_file
)


print("\nSaved dataset shape:")
print(saved_df.shape)


print("\nSaved columns:")

print(
    saved_df.columns.tolist()
)


missing_after_save = [
    col
    for col in thyroid_columns
    if col not in saved_df.columns
]


if missing_after_save:

    print(
        "\nERROR: Thyroid columns are still missing:"
    )

    print(
        missing_after_save
    )

    raise ValueError(
        "Step 1 output does not contain all "
        "required thyroid columns."
    )


else:

    print(
        "\nSUCCESS: All thyroid columns were "
        "preserved in Step 1 output."
    )


# ==============================================================
# 27. FINAL SUMMARY
# ==============================================================

print("\n" + "=" * 70)
print("STEP 1 COMPLETE")
print("=" * 70)


print("\nDataset used:")
print(input_file)


print("\nOriginal target:")
print(target_col)


print("\nCVD predictor features:")
print(len(cvd_features))


print("\nBest model:")
print(best_model_name)


print(
    "\nImportant methodological point:"
)


print(
    "Thyroid variables were NOT used as predictors "
    "in the CVD model."
)


print(
    "Thyroid variables were preserved in the test "
    "output for Step 3 analysis."
)


print(
    "\nOriginal Cardiovascular_Disease column "
    "has NOT been modified."
)


print("\nFiles saved:")

print(
    "1.",
    prediction_file
)

print(
    "2.",
    comparison_file
)


print(
    "\nNext step:"
)

print(
    "Run step2_threshold_analysis.py"
)


print(
    "\nAfter Step 2 completes successfully, "
    "run step3_thyroid_cvd_analysis.py."
)


print("=" * 70)
