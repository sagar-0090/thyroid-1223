import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score
)

print("=" * 70)
print("STEP 2 - CVD THRESHOLD ANALYSIS")
print("=" * 70)

# ==============================================================
# 1. LOAD CVD TEST PREDICTIONS
# ==============================================================

input_file = "output/cvd_test_predictions.csv"

df = pd.read_csv(input_file)

print("\nDataset loaded:")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ==============================================================
# 2. IDENTIFY TARGET AND PROBABILITY COLUMN
# ==============================================================

target_col = "Cardiovascular_Disease"

# Find probability column automatically
possible_probability_columns = [
    "CVD_Probability",
    "CVD_Prob",
    "Probability",
    "Predicted_Probability",
    "CVD_probability"
]

prob_col = None

for col in possible_probability_columns:
    if col in df.columns:
        prob_col = col
        break

if prob_col is None:

    print("\nProbability column was not found automatically.")

    print("Available columns:")
    print(df.columns.tolist())

    raise ValueError(
        "Please check the probability column name in "
        "output/cvd_test_predictions.csv"
    )


print("\nTarget column:", target_col)
print("Probability column:", prob_col)


# ==============================================================
# 3. BASIC INFORMATION
# ==============================================================

y_true = df[target_col].astype(int)
y_prob = df[prob_col].astype(float)

print("\nObserved CVD distribution:")
print(y_true.value_counts())

print("\nObserved CVD proportion:")
print(y_true.value_counts(normalize=True))


# ==============================================================
# 4. BASELINE ROC-AUC
# ==============================================================

roc_auc = roc_auc_score(y_true, y_prob)

print("\n" + "=" * 70)
print("BASELINE MODEL PERFORMANCE")
print("=" * 70)

print(f"\nROC-AUC: {roc_auc:.4f}")


# ==============================================================
# 5. THRESHOLD ANALYSIS
# ==============================================================

thresholds = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80
]

results = []

print("\n" + "=" * 70)
print("THRESHOLD PERFORMANCE")
print("=" * 70)

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    sensitivity = recall

    balanced_acc = balanced_accuracy_score(
        y_true,
        y_pred
    )

    predicted_cvd_count = int(y_pred.sum())

    predicted_cvd_percentage = (
        predicted_cvd_count / len(y_pred)
    ) * 100

    results.append({
        "Threshold": threshold,
        "Predicted_CVD_Count": predicted_cvd_count,
        "Predicted_CVD_Percentage": predicted_cvd_percentage,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        "Balanced_Accuracy": balanced_acc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

    print("\n" + "-" * 60)

    print(f"Threshold       : {threshold:.2f}")
    print(f"Predicted CVD   : {predicted_cvd_count}")
    print(f"Predicted CVD % : {predicted_cvd_percentage:.2f}%")

    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Precision       : {precision:.4f}")
    print(f"Recall          : {recall:.4f}")
    print(f"Specificity     : {specificity:.4f}")
    print(f"F1 Score        : {f1:.4f}")
    print(f"Balanced Acc.   : {balanced_acc:.4f}")

    print(
        f"Confusion Matrix: "
        f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
    )


results_df = pd.DataFrame(results)


# ==============================================================
# 6. FIND BEST THRESHOLD BY F1
# ==============================================================

best_f1_row = results_df.loc[
    results_df["F1"].idxmax()
]

best_f1_threshold = best_f1_row["Threshold"]


print("\n" + "=" * 70)
print("BEST THRESHOLD BASED ON F1 SCORE")
print("=" * 70)

print(
    f"\nBest threshold: "
    f"{best_f1_threshold:.2f}"
)

print(
    f"F1 Score: "
    f"{best_f1_row['F1']:.4f}"
)

print(
    f"Precision: "
    f"{best_f1_row['Precision']:.4f}"
)

print(
    f"Recall: "
    f"{best_f1_row['Recall']:.4f}"
)

print(
    f"Specificity: "
    f"{best_f1_row['Specificity']:.4f}"
)

print(
    f"Predicted CVD: "
    f"{int(best_f1_row['Predicted_CVD_Count'])}"
)

print(
    f"Predicted CVD %: "
    f"{best_f1_row['Predicted_CVD_Percentage']:.2f}%"
)


# ==============================================================
# 7. FIND BEST THRESHOLD BY BALANCED ACCURACY
# ==============================================================

best_balanced_row = results_df.loc[
    results_df["Balanced_Accuracy"].idxmax()
]

best_balanced_threshold = best_balanced_row["Threshold"]


print("\n" + "=" * 70)
print("BEST THRESHOLD BASED ON BALANCED ACCURACY")
print("=" * 70)

print(
    f"\nBest threshold: "
    f"{best_balanced_threshold:.2f}"
)

print(
    f"Balanced Accuracy: "
    f"{best_balanced_row['Balanced_Accuracy']:.4f}"
)

print(
    f"Precision: "
    f"{best_balanced_row['Precision']:.4f}"
)

print(
    f"Recall: "
    f"{best_balanced_row['Recall']:.4f}"
)

print(
    f"Specificity: "
    f"{best_balanced_row['Specificity']:.4f}"
)


# ==============================================================
# 8. SELECT RESEARCH THRESHOLD
# ==============================================================

# IMPORTANT:
# We are NOT automatically changing labels.
#
# For this project, initially use the F1-optimal threshold
# as the candidate threshold.
#
# We will review the results before using it downstream.

selected_threshold = float(best_f1_threshold)

print("\n" + "=" * 70)
print("SELECTED CANDIDATE THRESHOLD")
print("=" * 70)

print(
    f"\nCandidate threshold: "
    f"{selected_threshold:.2f}"
)

print(
    "\nThis threshold is selected based on maximum F1 "
    "among the evaluated thresholds."
)


# ==============================================================
# 9. GENERATE CVD PREDICTION USING SELECTED THRESHOLD
# ==============================================================

df["CVD_Prediction"] = (
    df[prob_col] >= selected_threshold
).astype(int)

print("\n" + "=" * 70)
print("CVD PREDICTION USING SELECTED THRESHOLD")
print("=" * 70)

print("\nPrediction distribution:")
print(
    df["CVD_Prediction"].value_counts()
)

print("\nPrediction proportion:")
print(
    df["CVD_Prediction"].value_counts(
        normalize=True
    )
)


# ==============================================================
# 10. COMPARE QUESTIONNAIRE CVD WITH MODEL CVD
# ==============================================================

comparison = pd.crosstab(
    df[target_col],
    df["CVD_Prediction"],
    rownames=["Questionnaire_CVD"],
    colnames=["Model_CVD"]
)

print("\n" + "=" * 70)
print("QUESTIONNAIRE CVD VS MODEL CVD")
print("=" * 70)

print("\n")
print(comparison)


# ==============================================================
# 11. IDENTIFY POTENTIAL DISCORDANT GROUPS
# ==============================================================

df["CVD_Status_Group"] = "Agreement"

df.loc[
    (df[target_col] == 0) &
    (df["CVD_Prediction"] == 1),
    "CVD_Status_Group"
] = "Model_Positive_Questionnaire_Negative"

df.loc[
    (df[target_col] == 1) &
    (df["CVD_Prediction"] == 0),
    "CVD_Status_Group"
] = "Model_Negative_Questionnaire_Positive"


print("\n" + "=" * 70)
print("DISCORDANT CVD GROUPS")
print("=" * 70)

print(
    df["CVD_Status_Group"].value_counts()
)


# ==============================================================
# 12. SAVE THRESHOLD RESULTS
# ==============================================================

results_file = (
    "output/cvd_threshold_analysis.csv"
)

results_df.to_csv(
    results_file,
    index=False
)


# ==============================================================
# 13. SAVE TEST PREDICTIONS WITH SELECTED THRESHOLD
# ==============================================================

prediction_file = (
    "output/cvd_test_predictions_thresholded.csv"
)

df.to_csv(
    prediction_file,
    index=False
)


# ==============================================================
# 14. FINAL SUMMARY
# ==============================================================

print("\n" + "=" * 70)
print("STEP 2 COMPLETE")
print("=" * 70)

print(
    f"\nCandidate threshold: "
    f"{selected_threshold:.2f}"
)

print(
    f"ROC-AUC: "
    f"{roc_auc:.4f}"
)

print(
    f"F1 at candidate threshold: "
    f"{best_f1_row['F1']:.4f}"
)

print(
    f"Predicted CVD: "
    f"{int(df['CVD_Prediction'].sum())}"
)

print(
    f"Predicted CVD percentage: "
    f"{df['CVD_Prediction'].mean() * 100:.2f}%"
)

print("\nFiles saved:")
print("1.", results_file)
print("2.", prediction_file)

print("\nIMPORTANT:")
print(
    "The original Cardiovascular_Disease column "
    "has NOT been modified."
)

print(
    "CVD_Prediction remains a separate "
    "model-generated variable."
)

print(
    "\nNext step: review threshold results "
    "before proceeding to thyroid-specific CVD analysis."
)

print("=" * 70)
