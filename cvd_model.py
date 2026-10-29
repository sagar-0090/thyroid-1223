import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

print("=" * 70)
print("STEP 1 - DEDICATED CVD PREDICTION MODEL")
print("=" * 70)


# ==============================================================
# 1. LOAD DATA
# ==============================================================

master = pd.read_csv(
    "output/cleaned_dataset.csv"
)

print("\nDataset shape:")
print(master.shape)


# ==============================================================
# 2. CVD PREDICTOR FEATURES
# ==============================================================

cvd_features = [
    "SYS_BP",
    "DIA_BP",
    "High_BP",
    "HDL",
    "Total_Cholesterol",
    "High_Cholesterol",
    "Low_HDL",
    "BMI",
    "Waist",
    "Weight",
    "Height",
    "Obesity",
    "Diabetes",
    "Smoking_100_Cigarettes",
    "Smoking_Risk",
    "Risk_Factor_Count"
]

target = "Cardiovascular_Disease"


print("\nCVD predictor features:")

for feature in cvd_features:
    print("-", feature)

print(
    "\nNumber of CVD predictor features:",
    len(cvd_features)
)


# ==============================================================
# 3. CHECK REQUIRED COLUMNS
# ==============================================================

required_columns = cvd_features + [target]

missing_columns = [
    col for col in required_columns
    if col not in master.columns
]

if missing_columns:

    print("\nERROR - Missing columns:")
    print(missing_columns)

    raise ValueError(
        "Required columns are missing from cleaned_dataset.csv"
    )

print("\nAll required columns are present.")


# ==============================================================
# 4. CREATE MODEL DATASET
# ==============================================================

model_df = master[
    cvd_features + [target]
].copy()

print("\nModel dataset shape:")
print(model_df.shape)

print("\nMissing values:")
print(model_df.isnull().sum())


# ==============================================================
# 5. TARGET DISTRIBUTION
# ==============================================================

print("\nTarget distribution:")
print(model_df[target].value_counts())

print("\nTarget proportion:")
print(model_df[target].value_counts(normalize=True))


# ==============================================================
# 6. SEPARATE X AND Y
# ==============================================================

X = model_df[cvd_features]

y = model_df[target].astype(int)


# ==============================================================
# 7. IDENTIFY FEATURE TYPES
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
# 8. TRAIN / TEST SPLIT
# ==============================================================

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
            "onehot",
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
# 10. LOGISTIC REGRESSION
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
                class_weight="balanced"
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
# 11. RANDOM FOREST
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
# 12. MODEL EVALUATION FUNCTION
# ==============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name
):

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

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
    }, y_pred, y_prob


# ==============================================================
# 13. EVALUATE BOTH MODELS
# ==============================================================

logistic_results, logistic_pred, logistic_prob = evaluate_model(
    logistic_model,
    X_test,
    y_test,
    "Logistic Regression"
)

rf_results, rf_pred, rf_prob = evaluate_model(
    random_forest_model,
    X_test,
    y_test,
    "Random Forest"
)


# ==============================================================
# 14. MODEL COMPARISON
# ==============================================================

comparison_df = pd.DataFrame(
    [
        logistic_results,
        rf_results
    ]
)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    comparison_df.to_string(
        index=False
    )
)


# ==============================================================
# 15. SELECT BEST MODEL USING ROC-AUC
# ==============================================================

if logistic_results["ROC_AUC"] >= rf_results["ROC_AUC"]:

    best_model = logistic_model
    best_model_name = "Logistic Regression"
    best_probability = logistic_prob

else:

    best_model = random_forest_model
    best_model_name = "Random Forest"
    best_probability = rf_prob


print(
    "\nBest model based on ROC-AUC:"
)

print(best_model_name)


# ==============================================================
# 16. GENERATE TEST-SET PREDICTIONS
# ==============================================================

print("\n" + "=" * 70)
print("GENERATING CVD PROBABILITIES")
print("=" * 70)


# Keep the ORIGINAL questionnaire label
test_predictions = X_test.copy()

test_predictions[
    "Cardiovascular_Disease"
] = y_test.values


# Model probability
test_predictions[
    "CVD_Probability"
] = best_probability


# Standard 0.50 prediction
test_predictions[
    "CVD_Prediction_0.50"
] = (
    best_probability >= 0.50
).astype(int)


print("\nInitial threshold: 0.50")

print("\nCVD prediction distribution:")

print(
    test_predictions[
        "CVD_Prediction_0.50"
    ].value_counts()
)

print("\nCVD prediction proportion:")

print(
    test_predictions[
        "CVD_Prediction_0.50"
    ].value_counts(
        normalize=True
    )
)


# ==============================================================
# 17. SAVE TEST PREDICTIONS
# ==============================================================

prediction_file = (
    "output/cvd_test_predictions.csv"
)

test_predictions.to_csv(
    prediction_file,
    index=False
)

print(
    "\nCVD test predictions saved to:"
)

print(prediction_file)


# ==============================================================
# 18. SAVE MODEL COMPARISON
# ==============================================================

comparison_file = (
    "output/cvd_model_comparison.csv"
)

comparison_df.to_csv(
    comparison_file,
    index=False
)

print(
    "Model comparison saved to:"
)

print(comparison_file)


# ==============================================================
# 19. FINAL
# ==============================================================

print("\n" + "=" * 70)
print("STEP 1 COMPLETE")
print("=" * 70)

print(
    "\nDataset used:"
    " output/cleaned_dataset.csv"
)

print(
    "\nOriginal target:"
    " Cardiovascular_Disease"
)

print(
    "\nProbability column:"
    " CVD_Probability"
)

print(
    "\nImportant:"
)

print(
    "The original Cardiovascular_Disease "
    "column has NOT been modified."
)

print(
    "CVD_Prediction_0.50 is only a "
    "model-generated prediction."
)

print(
    "\nStep 2 can now perform threshold analysis."
)

print("=" * 70)
