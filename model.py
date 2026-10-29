import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
    confusion_matrix,
    roc_curve
)

import matplotlib.pyplot as plt


# ============================================================
# EDA-3 - PREDICTIVE MODELING
# ============================================================

print("=" * 70)
print("EDA-3 - PREDICTIVE MODELING")
print("=" * 70)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("output/cleaned_dataset.csv")

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. DEFINE TARGET
# ============================================================

TARGET = "Cardiovascular_Disease"

X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].copy()


# SEQN is only an identification number.
# It must NOT be used as a predictive feature.
if "SEQN" in X.columns:
    X = X.drop(columns=["SEQN"])


print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget proportion:")
print(y.value_counts(normalize=True))


# ============================================================
# 3. IDENTIFY FEATURE TYPES
# ============================================================

categorical_features = [
    "Gender",
    "Race",
    "Smoking_100_Cigarettes",
    "Thyroid_Diagnosis",
    "Diabetes"
]

# Keep only columns that actually exist
categorical_features = [
    c for c in categorical_features
    if c in X.columns
]

numerical_features = [
    c for c in X.columns
    if c not in categorical_features
]


print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

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


# ============================================================
# 5. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# ============================================================
# 6. LOGISTIC REGRESSION
# ============================================================

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
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


# ============================================================
# 7. RANDOM FOREST
# ============================================================

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# 8. TRAIN MODELS
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)

logistic_model.fit(X_train, y_train)

print("Logistic Regression training complete.")


print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

random_forest_model.fit(X_train, y_train)

print("Random Forest training complete.")


# ============================================================
# 9. PREDICTIONS
# ============================================================

logistic_pred = logistic_model.predict(X_test)
logistic_prob = logistic_model.predict_proba(X_test)[:, 1]

rf_pred = random_forest_model.predict(X_test)
rf_prob = random_forest_model.predict_proba(X_test)[:, 1]


# ============================================================
# 10. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, y_true, predictions, probabilities):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_true,
        probabilities
    )

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            predictions,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_true,
            predictions
        )
    )

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc
    }


# ============================================================
# 11. EVALUATE MODELS
# ============================================================

logistic_results = evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_pred,
    logistic_prob
)

rf_results = evaluate_model(
    "Random Forest",
    y_test,
    rf_pred,
    rf_prob
)


# ============================================================
# 12. MODEL COMPARISON
# ============================================================

results = pd.DataFrame([
    logistic_results,
    rf_results
])

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results.to_string(index=False))


# ============================================================
# 13. SELECT BEST MODEL
# ============================================================

best_index = results["ROC_AUC"].idxmax()

best_model_name = results.loc[
    best_index,
    "Model"
]

print("\nBest model based on ROC-AUC:")
print(best_model_name)


# ============================================================
# 14. SAVE RESULTS
# ============================================================

results.to_csv(
    "output/model_comparison.csv",
    index=False
)

print("\nModel comparison saved:")
print("output/model_comparison.csv")


# ============================================================
# 15. ROC CURVE
# ============================================================

logistic_fpr, logistic_tpr, _ = roc_curve(
    y_test,
    logistic_prob
)

rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    rf_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    logistic_fpr,
    logistic_tpr,
    label=f"Logistic Regression (AUC = {logistic_results['ROC_AUC']:.3f})"
)

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest (AUC = {rf_results['ROC_AUC']:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Cardiovascular Disease Prediction")

plt.legend()

plt.tight_layout()

plt.savefig(
    "output/roc_curve.png",
    dpi=300
)

plt.show()


# ============================================================
# 16. SAVE TEST PREDICTIONS
# ============================================================

prediction_output = X_test.copy()

prediction_output["Actual_CVD"] = y_test.values
prediction_output["Logistic_Prediction"] = logistic_pred
prediction_output["Logistic_CVD_Probability"] = logistic_prob
prediction_output["RandomForest_Prediction"] = rf_pred
prediction_output["RandomForest_CVD_Probability"] = rf_prob

prediction_output.to_csv(
    "output/model_predictions.csv",
    index=False
)

print("\nPredictions saved:")
print("output/model_predictions.csv")


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("EDA-3 COMPLETE")
print("=" * 70)
