import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu


# ============================================================================
# STEP 5 - INTEGRATED THYROID + CVD RISK ANALYSIS
# ============================================================================

print("=" * 75)
print("STEP 5 - INTEGRATED THYROID + CVD RISK ANALYSIS")
print("=" * 75)


# ============================================================================
# 1. LOAD STEP-4 DATA
# ============================================================================

input_file = "output/step4_thyroid_problem_population.csv"

df = pd.read_csv(input_file)

print("\nDataset loaded:")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================================

required_columns = [
    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO",
    "Thyroid_Diagnosis",
    "Abnormal_TSH",
    "CVD_Probability",
    "CVD_Prediction",
    "Cardiovascular_Disease"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\nAll required columns are present.")


# ============================================================================
# 3. BASIC DATA VALIDATION
# ============================================================================

print("\n" + "=" * 75)
print("1. DATA VALIDATION")
print("=" * 75)

print("\nTotal records:", len(df))

print("\nMissing values in important variables:")

important_columns = [
    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO",
    "Thyroid_Diagnosis",
    "Abnormal_TSH",
    "CVD_Probability",
    "CVD_Prediction"
]

print(
    df[important_columns]
    .isnull()
    .sum()
)


# ============================================================================
# 4. CREATE THYROID STATUS
# ============================================================================

print("\n" + "=" * 75)
print("2. THYROID STATUS CLASSIFICATION")
print("=" * 75)


# Based on the coding used in previous steps:
#
# 1 = Thyroid problem
# 2 = No thyroid problem
# 9 = Unknown / other
#
# We DO NOT reinterpret the underlying medical diagnosis.
# We simply preserve the existing dataset coding.

df["Thyroid_Status"] = "Unknown / Other"

df.loc[
    df["Thyroid_Diagnosis"] == 1,
    "Thyroid_Status"
] = "Thyroid Problem"

df.loc[
    df["Thyroid_Diagnosis"] == 2,
    "Thyroid_Status"
] = "No Thyroid Problem"


print("\nThyroid status distribution:")

print(
    df["Thyroid_Status"]
    .value_counts()
)


print("\nThyroid status percentage:")

print(
    (
        df["Thyroid_Status"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


# ============================================================================
# 5. CVD PROBABILITY GROUPS
# ============================================================================

print("\n" + "=" * 75)
print("3. CVD PROBABILITY ANALYSIS")
print("=" * 75)


# IMPORTANT:
#
# These are analytical probability bands.
# They are NOT clinical diagnostic categories.
#
# They are used only to make the model output easier to interpret.

def classify_cvd_probability(probability):

    if probability < 0.20:
        return "Low"

    elif probability < 0.40:
        return "Moderate"

    elif probability < 0.60:
        return "High"

    else:
        return "Very High"


df["CVD_Risk_Category"] = (
    df["CVD_Probability"]
    .apply(classify_cvd_probability)
)


print("\nCVD probability category distribution:")

print(
    df["CVD_Risk_Category"]
    .value_counts()
)


print("\nCVD probability category percentage:")

print(
    (
        df["CVD_Risk_Category"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


# ============================================================================
# 6. THYROID STATUS VS CVD PROBABILITY
# ============================================================================

print("\n" + "=" * 75)
print("4. THYROID STATUS VS CVD PROBABILITY")
print("=" * 75)


thyroid_cvd_summary = (
    df.groupby("Thyroid_Status")
    .agg(
        Number_of_Patients=("CVD_Probability", "count"),
        Mean_CVD_Probability=("CVD_Probability", "mean"),
        Median_CVD_Probability=("CVD_Probability", "median"),
        Std_CVD_Probability=("CVD_Probability", "std"),
        Minimum_CVD_Probability=("CVD_Probability", "min"),
        Maximum_CVD_Probability=("CVD_Probability", "max"),
        Predicted_CVD_Count=("CVD_Prediction", "sum")
    )
)

thyroid_cvd_summary["Predicted_CVD_Percentage"] = (
    thyroid_cvd_summary["Predicted_CVD_Count"]
    / thyroid_cvd_summary["Number_of_Patients"]
    * 100
)


print("\nSummary:")

print(
    thyroid_cvd_summary.round(4)
)


# ============================================================================
# 7. THYROID STATUS VS CVD PREDICTION
# ============================================================================

print("\n" + "=" * 75)
print("5. THYROID STATUS VS CVD PREDICTION")
print("=" * 75)


thyroid_cvd_table = pd.crosstab(
    df["Thyroid_Status"],
    df["CVD_Prediction"]
)

print("\nCount table:")

print(thyroid_cvd_table)


thyroid_cvd_percentage = pd.crosstab(
    df["Thyroid_Status"],
    df["CVD_Prediction"],
    normalize="index"
) * 100


print("\nPercentage within thyroid status:")

print(
    thyroid_cvd_percentage.round(2)
)


# ============================================================================
# 8. THYROID STATUS VS CVD RISK CATEGORY
# ============================================================================

print("\n" + "=" * 75)
print("6. THYROID STATUS VS CVD RISK CATEGORY")
print("=" * 75)


risk_category_table = pd.crosstab(
    df["Thyroid_Status"],
    df["CVD_Risk_Category"]
)

print("\nCount table:")

print(risk_category_table)


risk_category_percentage = pd.crosstab(
    df["Thyroid_Status"],
    df["CVD_Risk_Category"],
    normalize="index"
) * 100


print("\nPercentage within thyroid status:")

print(
    risk_category_percentage.round(2)
)


# ============================================================================
# 9. THYROID-PROBLEM PATIENTS ONLY
# ============================================================================

print("\n" + "=" * 75)
print("7. THYROID-PROBLEM PATIENTS ONLY")
print("=" * 75)


thyroid_patients = df[
    df["Thyroid_Status"] == "Thyroid Problem"
].copy()


print("\nNumber of thyroid-problem patients:")

print(len(thyroid_patients))


if len(thyroid_patients) > 0:

    print("\nCVD prediction distribution:")

    print(
        thyroid_patients["CVD_Prediction"]
        .value_counts()
    )


    print("\nCVD prediction percentage:")

    print(
        (
            thyroid_patients["CVD_Prediction"]
            .value_counts(normalize=True)
            * 100
        ).round(2)
    )


    print("\nMean CVD probability:")

    print(
        thyroid_patients["CVD_Probability"].mean()
    )


    print("\nMedian CVD probability:")

    print(
        thyroid_patients["CVD_Probability"].median()
    )


# ============================================================================
# 10. THYROID BIOMARKERS IN THYROID-PROBLEM PATIENTS
# ============================================================================

print("\n" + "=" * 75)
print("8. THYROID BIOMARKERS IN THYROID-PROBLEM PATIENTS")
print("=" * 75)


thyroid_biomarkers = [
    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO"
]


biomarker_summary = []


for feature in thyroid_biomarkers:

    print("\n" + "-" * 60)

    print(feature)


    cvd_zero = thyroid_patients.loc[
        thyroid_patients["CVD_Prediction"] == 0,
        feature
    ].dropna()


    cvd_one = thyroid_patients.loc[
        thyroid_patients["CVD_Prediction"] == 1,
        feature
    ].dropna()


    print(
        "Mean CVD = 0:",
        cvd_zero.mean()
    )

    print(
        "Mean CVD = 1:",
        cvd_one.mean()
    )


    print(
        "Median CVD = 0:",
        cvd_zero.median()
    )

    print(
        "Median CVD = 1:",
        cvd_one.median()
    )


    if len(cvd_zero) > 0 and len(cvd_one) > 0:

        u_stat, p_value = mannwhitneyu(
            cvd_zero,
            cvd_one,
            alternative="two-sided"
        )

    else:

        u_stat = np.nan
        p_value = np.nan


    print(
        "Mann-Whitney U:",
        u_stat
    )

    print(
        "P-value:",
        p_value
    )


    biomarker_summary.append({

        "Biomarker": feature,

        "Mean_CVD_0": cvd_zero.mean(),

        "Mean_CVD_1": cvd_one.mean(),

        "Median_CVD_0": cvd_zero.median(),

        "Median_CVD_1": cvd_one.median(),

        "Mann_Whitney_U": u_stat,

        "P_Value": p_value

    })


# ============================================================================
# 11. ABNORMAL TSH + CVD
# ============================================================================

print("\n" + "=" * 75)
print("9. ABNORMAL TSH VS CVD")
print("=" * 75)


abnormal_tsh_table = pd.crosstab(
    df["Abnormal_TSH"],
    df["CVD_Prediction"]
)

print("\nCount table:")

print(abnormal_tsh_table)


abnormal_tsh_percentage = pd.crosstab(
    df["Abnormal_TSH"],
    df["CVD_Prediction"],
    normalize="index"
) * 100


print("\nPercentage:")

print(
    abnormal_tsh_percentage.round(2)
)


# ============================================================================
# 12. INTEGRATED THYROID-CVD GROUP
# ============================================================================

print("\n" + "=" * 75)
print("10. INTEGRATED THYROID-CVD RISK GROUP")
print("=" * 75)


def integrated_risk(row):

    thyroid_problem = (
        row["Thyroid_Status"] == "Thyroid Problem"
    )

    cvd_positive = (
        row["CVD_Prediction"] == 1
    )

    probability = row["CVD_Probability"]


    if thyroid_problem and cvd_positive:

        if probability >= 0.60:
            return "Thyroid Problem + Very High CVD Probability"

        else:
            return "Thyroid Problem + Elevated CVD Probability"


    elif thyroid_problem and not cvd_positive:

        return "Thyroid Problem + Lower CVD Probability"


    elif not thyroid_problem and cvd_positive:

        return "No Thyroid Problem + Elevated CVD Probability"


    elif not thyroid_problem and not cvd_positive:

        return "No Thyroid Problem + Lower CVD Probability"


    else:

        return "Unknown"


df["Integrated_Thyroid_CVD_Group"] = (
    df.apply(
        integrated_risk,
        axis=1
    )
)


print("\nIntegrated group distribution:")

print(
    df["Integrated_Thyroid_CVD_Group"]
    .value_counts()
)


print("\nIntegrated group percentage:")

print(
    (
        df["Integrated_Thyroid_CVD_Group"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


# ============================================================================
# 13. THYROID PROBLEM + ELEVATED CVD PROBABILITY
# ============================================================================

print("\n" + "=" * 75)
print("11. THYROID PROBLEM + ELEVATED CVD PROBABILITY")
print("=" * 75)


thyroid_elevated = df[
    (
        df["Thyroid_Status"] == "Thyroid Problem"
    )
    &
    (
        df["CVD_Prediction"] == 1
    )
].copy()


print(
    "\nNumber of thyroid-problem participants "
    "with model-predicted CVD:",
    len(thyroid_elevated)
)


if len(thyroid_patients) > 0:

    percentage = (
        len(thyroid_elevated)
        /
        len(thyroid_patients)
        * 100
    )

else:

    percentage = np.nan


print(
    "Percentage of thyroid-problem participants:",
    round(percentage, 2),
    "%"
)


if len(thyroid_elevated) > 0:

    print(
        "\nMean CVD probability:",
        round(
            thyroid_elevated["CVD_Probability"].mean(),
            4
        )
    )

    print(
        "Median CVD probability:",
        round(
            thyroid_elevated["CVD_Probability"].median(),
            4
        )
    )


# ============================================================================
# 14. SAVE RESULTS
# ============================================================================

print("\n" + "=" * 75)
print("12. SAVING RESULTS")
print("=" * 75)


output_files = {

    "output/step5_thyroid_cvd_summary.csv":
        thyroid_cvd_summary,

    "output/step5_thyroid_cvd_percentage.csv":
        thyroid_cvd_percentage,

    "output/step5_thyroid_cvd_risk_categories.csv":
        risk_category_percentage,

    "output/step5_thyroid_biomarker_analysis.csv":
        pd.DataFrame(biomarker_summary),

    "output/step5_abnormal_tsh_cvd.csv":
        abnormal_tsh_percentage,

    "output/step5_integrated_risk_groups.csv":
        df["Integrated_Thyroid_CVD_Group"]
        .value_counts()
        .rename_axis("Integrated_Thyroid_CVD_Group")
        .reset_index(name="Count")

}


for filename, data in output_files.items():

    data.to_csv(
        filename,
        index=True
    )

    print("Saved:", filename)


# ============================================================================
# 15. SAVE COMPLETE STEP-5 DATASET
# ============================================================================

complete_output = (
    "output/step5_integrated_thyroid_cvd_dataset.csv"
)


df.to_csv(
    complete_output,
    index=False
)


print(
    "\nSaved complete integrated dataset:",
    complete_output
)


# ============================================================================
# 16. FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 75)
print("STEP 5 COMPLETE")
print("=" * 75)


print(
    "\nTotal analyzed records:",
    len(df)
)


print(
    "Thyroid-problem records:",
    len(thyroid_patients)
)


print(
    "Thyroid-problem + model-predicted CVD:",
    len(thyroid_elevated)
)


print(
    "Percentage of thyroid-problem participants "
    "with model-predicted CVD:",
    round(percentage, 2),
    "%"
)


print(
    "\nCVD probability is retained as a continuous model output."
)


print(
    "CVD_Prediction is retained as the threshold-based model output."
)


print(
    "\nOriginal Cardiovascular_Disease column:",
    "NOT MODIFIED"
)


print("\n" + "=" * 75)
