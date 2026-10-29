import pandas as pd

# ============================================================
# PHASE 2 + PHASE 2.5
# DATA CLEANING + FEATURE ENGINEERING
# ============================================================

# ------------------------------------------------------------
# 1. Load master dataset
# ------------------------------------------------------------

master = pd.read_csv("output/master_dataset.csv")

print("=" * 60)
print("ORIGINAL DATASET")
print("=" * 60)
print("Shape:", master.shape)


# ------------------------------------------------------------
# 2. Remove variables that are not required for the main model
# ------------------------------------------------------------
# LDL and Triglycerides have very high missingness.
# Current_Smoker is not used because we use the available
# Smoking_100_Cigarettes variable instead.
#
# MCQ160B-F were used to construct the combined
# Cardiovascular_Disease target, so they should not be
# used as independent predictors.

columns_to_drop = [
    "LDL",
    "Triglycerides",
    "Current_Smoker",
    "MCQ160B",
    "MCQ160C",
    "MCQ160D",
    "MCQ160E",
    "MCQ160F"
]

master_main = master.drop(
    columns=columns_to_drop,
    errors="ignore"
).copy()


# ------------------------------------------------------------
# 3. Require complete thyroid laboratory measurements
# ------------------------------------------------------------
# The primary project requires:
# TSH, Free T3 and Free T4

master_main = master_main.dropna(
    subset=[
        "TSH",
        "Free_T3",
        "Free_T4"
    ]
).copy()


# ------------------------------------------------------------
# 4. Handle missing numerical variables
# ------------------------------------------------------------

num_cols = [
    "BMI",
    "Height",
    "Weight",
    "Waist",
    "SYS_BP",
    "DIA_BP",
    "HDL",
    "Total_Cholesterol",
    "TPO"
]

for c in num_cols:
    master_main[c] = master_main[c].fillna(
        master_main[c].median()
    )


# ------------------------------------------------------------
# 5. Handle missing categorical variables
# ------------------------------------------------------------

cat_cols = [
    "Smoking_100_Cigarettes",
    "Diabetes",
    "Thyroid_Diagnosis"
]

for c in cat_cols:
    master_main[c] = master_main[c].fillna(
        master_main[c].mode()[0]
    )


# ------------------------------------------------------------
# 6. Create cardiovascular-related features
# ------------------------------------------------------------

# High blood pressure
master_main["High_BP"] = (
    (master_main["SYS_BP"] >= 130) |
    (master_main["DIA_BP"] >= 80)
).astype(int)


# Obesity
master_main["Obesity"] = (
    master_main["BMI"] >= 30
).astype(int)


# High total cholesterol
master_main["High_Cholesterol"] = (
    master_main["Total_Cholesterol"] >= 200
).astype(int)


# Low HDL
master_main["Low_HDL"] = (
    master_main["HDL"] < 40
).astype(int)


# Abnormal TSH
master_main["Abnormal_TSH"] = (
    (master_main["TSH"] < 0.4) |
    (master_main["TSH"] > 4.5)
).astype(int)


# Previous smoking exposure
master_main["Smoking_Risk"] = (
    master_main["Smoking_100_Cigarettes"] == 1
).astype(int)


# ------------------------------------------------------------
# 7. Create an overall risk-factor count
# ------------------------------------------------------------

master_main["Risk_Factor_Count"] = (
    master_main["High_BP"]
    + master_main["Obesity"]
    + master_main["High_Cholesterol"]
    + master_main["Low_HDL"]
    + master_main["Abnormal_TSH"]
    + master_main["Smoking_Risk"]
    + master_main["Diabetes"]
)


# ------------------------------------------------------------
# 8. Display final dataset information
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL DATASET")
print("=" * 60)

print("\nFinal shape:")
print(master_main.shape)


print("\nColumns:")
print(master_main.columns.tolist())


print("\nMissing values:")
print(master_main.isnull().sum())


print("\nCardiovascular disease distribution:")
print(
    master_main["Cardiovascular_Disease"].value_counts()
)


print("\nCardiovascular disease proportion:")
print(
    master_main["Cardiovascular_Disease"]
    .value_counts(normalize=True)
)


print("\nThyroid diagnosis distribution:")
print(
    master_main["Thyroid_Diagnosis"].value_counts()
)


print("\nRisk factor count:")
print(
    master_main["Risk_Factor_Count"].describe()
)


# ------------------------------------------------------------
# 9. Display first five records
# ------------------------------------------------------------

print("\nFirst five records:")
print(master_main.head())


# ------------------------------------------------------------
# 10. Save FINAL cleaned + engineered dataset
# ------------------------------------------------------------

master_main.to_csv(
    "output/cleaned_dataset.csv",
    index=False
)

print("\n" + "=" * 60)
print("SUCCESS")
print("=" * 60)
print(
    "Final cleaned dataset saved to:"
    " output/cleaned_dataset.csv"
)
# ------------------------------------------------------------
# 11. Temporary section
# ------------------------------------------------------------
print("\nThyroid Diagnosis vs Cardiovascular Disease:")
print(
    pd.crosstab(
        master_main["Thyroid_Diagnosis"],
        master_main["Cardiovascular_Disease"],
        margins=True
    )
)

print("\nThyroid Diagnosis proportions:")
print(
    master_main["Thyroid_Diagnosis"].value_counts(
        normalize=True
    )
)
print("\nAbnormal TSH vs Cardiovascular Disease:")
print(
    pd.crosstab(
        master_main["Abnormal_TSH"],
        master_main["Cardiovascular_Disease"],
        margins=True
    )
)
# ============================================================
# PHASE 3 - EDA 1
# BASIC THYROID / CVD ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("EDA 1 - THYROID AND CARDIOVASCULAR RELATIONSHIP")
print("=" * 60)

# ------------------------------------------------------------
# 1. CVD rate by thyroid diagnosis
# ------------------------------------------------------------

thyroid_cvd = pd.crosstab(
    master_main["Thyroid_Diagnosis"],
    master_main["Cardiovascular_Disease"],
    normalize="index"
) * 100

print("\nCVD percentage within each thyroid diagnosis:")
print(thyroid_cvd)


# ------------------------------------------------------------
# 2. CVD rate by abnormal TSH
# ------------------------------------------------------------

tsh_cvd = pd.crosstab(
    master_main["Abnormal_TSH"],
    master_main["Cardiovascular_Disease"],
    normalize="index"
) * 100

print("\nCVD percentage by TSH status:")
print(tsh_cvd)


# ------------------------------------------------------------
# 3. CVD rate by individual risk factors
# ------------------------------------------------------------

risk_features = [
    "High_BP",
    "Obesity",
    "High_Cholesterol",
    "Low_HDL",
    "Abnormal_TSH",
    "Smoking_Risk",
    "Diabetes"
]

print("\nCVD percentage by risk factor:")
print()

for feature in risk_features:

    result = pd.crosstab(
        master_main[feature],
        master_main["Cardiovascular_Disease"],
        normalize="index"
    ) * 100

    print("-" * 50)
    print(feature)
    print(result)


# ------------------------------------------------------------
# 4. Risk factor count vs CVD
# ------------------------------------------------------------

risk_count_cvd = pd.crosstab(
    master_main["Risk_Factor_Count"],
    master_main["Cardiovascular_Disease"],
    normalize="index"
) * 100

print("\nCVD percentage by number of risk factors:")
print(risk_count_cvd)


# ------------------------------------------------------------
# 5. Age comparison
# ------------------------------------------------------------

print("\nAge statistics by cardiovascular outcome:")
print(
    master_main.groupby(
        "Cardiovascular_Disease"
    )["Age"].describe()
)


# ------------------------------------------------------------
# 6. Thyroid laboratory comparison
# ------------------------------------------------------------

thyroid_labs = [
    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO"
]

print("\nThyroid laboratory statistics by cardiovascular outcome:")

for feature in thyroid_labs:

    print("\n" + "-" * 50)
    print(feature)

    print(
        master_main.groupby(
            "Cardiovascular_Disease"
        )[feature].describe()
    )
    
# ============================================================
# PHASE 3 - EDA 2
# STATISTICAL ANALYSIS
# ============================================================

import pandas as pd
import numpy as np

from scipy.stats import (
    chi2_contingency,
    fisher_exact,
    mannwhitneyu
)

from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests


print("\n" + "=" * 70)
print("EDA 2 - STATISTICAL ANALYSIS")
print("=" * 70)


# ============================================================
# 1. CHI-SQUARE TESTS
# ============================================================

print("\n")
print("=" * 70)
print("1. CHI-SQUARE TESTS")
print("=" * 70)

categorical_features = [
    "Thyroid_Diagnosis",
    "Abnormal_TSH",
    "High_BP",
    "Obesity",
    "High_Cholesterol",
    "Low_HDL",
    "Smoking_Risk",
    "Diabetes"
]

chi_results = []

for feature in categorical_features:

    # Remove unknown thyroid diagnosis code 9
    # from the thyroid diagnosis test
    if feature == "Thyroid_Diagnosis":
        temp = master_main[
            master_main["Thyroid_Diagnosis"] != 9
        ].copy()
    else:
        temp = master_main.copy()

    contingency = pd.crosstab(
        temp[feature],
        temp["Cardiovascular_Disease"]
    )

    chi2, p_value, dof, expected = chi2_contingency(
        contingency
    )

    chi_results.append({
        "Feature": feature,
        "Chi2": chi2,
        "p_value": p_value,
        "Degrees_of_Freedom": dof
    })

    print("\n" + "-" * 60)
    print(f"Feature: {feature}")

    print("\nContingency table:")
    print(contingency)

    print(f"\nChi-square statistic: {chi2:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p_value:.6f}")

    if p_value < 0.05:
        print("Result: STATISTICALLY SIGNIFICANT")
    else:
        print("Result: Not statistically significant")


chi_results_df = pd.DataFrame(chi_results)


# ============================================================
# 2. ODDS RATIO
# ============================================================

print("\n")
print("=" * 70)
print("2. ODDS RATIOS")
print("=" * 70)

print(
    "\nOdds Ratio > 1 indicates higher observed odds of CVD."
)
print(
    "Odds Ratio < 1 indicates lower observed odds of CVD."
)


def calculate_odds_ratio(data, feature):

    # For binary features only
    table = pd.crosstab(
        data[feature],
        data["Cardiovascular_Disease"]
    )

    # Make sure rows 0 and 1 exist
    if not set([0, 1]).issubset(table.index):
        return None

    if not set([0, 1]).issubset(table.columns):
        return None

    a = table.loc[1, 1]
    b = table.loc[1, 0]
    c = table.loc[0, 1]
    d = table.loc[0, 0]

    # Avoid division by zero
    if b == 0 or c == 0:
        return None

    odds_ratio = (a * d) / (b * c)

    # Standard error of log(OR)
    se = np.sqrt(
        (1 / a) +
        (1 / b) +
        (1 / c) +
        (1 / d)
    )

    log_or = np.log(odds_ratio)

    lower = np.exp(
        log_or - 1.96 * se
    )

    upper = np.exp(
        log_or + 1.96 * se
    )

    return odds_ratio, lower, upper


binary_features = [
    "Abnormal_TSH",
    "High_BP",
    "Obesity",
    "High_Cholesterol",
    "Low_HDL",
    "Smoking_Risk"
]

odds_results = []

for feature in binary_features:

    result = calculate_odds_ratio(
        master_main,
        feature
    )

    if result is not None:

        odds_ratio, lower, upper = result

        odds_results.append({
            "Feature": feature,
            "Odds_Ratio": odds_ratio,
            "CI_Lower": lower,
            "CI_Upper": upper
        })

        print("\n" + "-" * 60)
        print(feature)

        print(
            f"Odds Ratio: {odds_ratio:.3f}"
        )

        print(
            f"95% CI: ({lower:.3f}, {upper:.3f})"
        )

        if lower > 1:
            print(
                "Interpretation: Higher observed odds of CVD."
            )

        elif upper < 1:
            print(
                "Interpretation: Lower observed odds of CVD."
            )

        else:
            print(
                "Interpretation: Confidence interval includes 1."
            )


odds_results_df = pd.DataFrame(odds_results)


# ============================================================
# 3. THYROID DIAGNOSIS ODDS RATIO
# ============================================================

print("\n")
print("=" * 70)
print("3. THYROID DIAGNOSIS ODDS RATIO")
print("=" * 70)

# Convert thyroid diagnosis into:
# 1 = reported thyroid problem
# 0 = no reported thyroid problem
#
# Exclude code 9 (unknown)

thyroid_binary = master_main[
    master_main["Thyroid_Diagnosis"].isin([1, 2])
].copy()

thyroid_binary["Thyroid_Problem"] = (
    thyroid_binary["Thyroid_Diagnosis"] == 1
).astype(int)

thyroid_table = pd.crosstab(
    thyroid_binary["Thyroid_Problem"],
    thyroid_binary["Cardiovascular_Disease"]
)

print("\nContingency table:")
print(thyroid_table)

a = thyroid_table.loc[1, 1]
b = thyroid_table.loc[1, 0]
c = thyroid_table.loc[0, 1]
d = thyroid_table.loc[0, 0]

thyroid_or = (a * d) / (b * c)

se = np.sqrt(
    (1 / a) +
    (1 / b) +
    (1 / c) +
    (1 / d)
)

log_or = np.log(thyroid_or)

thyroid_lower = np.exp(
    log_or - 1.96 * se
)

thyroid_upper = np.exp(
    log_or + 1.96 * se
)

print(
    f"\nOdds Ratio: {thyroid_or:.3f}"
)

print(
    f"95% CI: ({thyroid_lower:.3f}, "
    f"{thyroid_upper:.3f})"
)


# Chi-square for thyroid diagnosis

chi2, p_value, dof, expected = chi2_contingency(
    thyroid_table
)

print(
    f"\nChi-square statistic: {chi2:.4f}"
)

print(
    f"P-value: {p_value:.6f}"
)

if p_value < 0.05:
    print(
        "Result: Thyroid diagnosis is "
        "statistically associated with CVD."
    )
else:
    print(
        "Result: No statistically significant "
        "association detected."
    )


# ============================================================
# 4. CONTINUOUS VARIABLES
#    MANN-WHITNEY U TEST
# ============================================================

print("\n")
print("=" * 70)
print("4. CONTINUOUS VARIABLE ANALYSIS")
print("=" * 70)

continuous_features = [
    "Age",
    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO",
    "BMI",
    "SYS_BP",
    "DIA_BP",
    "HDL",
    "Total_Cholesterol"
]

continuous_results = []

for feature in continuous_features:

    group_0 = master_main[
        master_main["Cardiovascular_Disease"] == 0
    ][feature].dropna()

    group_1 = master_main[
        master_main["Cardiovascular_Disease"] == 1
    ][feature].dropna()

    statistic, p_value = mannwhitneyu(
        group_0,
        group_1,
        alternative="two-sided"
    )

    median_0 = group_0.median()
    median_1 = group_1.median()

    mean_0 = group_0.mean()
    mean_1 = group_1.mean()

    continuous_results.append({
        "Feature": feature,
        "Mean_No_CVD": mean_0,
        "Mean_CVD": mean_1,
        "Median_No_CVD": median_0,
        "Median_CVD": median_1,
        "U_Statistic": statistic,
        "p_value": p_value
    })

    print("\n" + "-" * 60)
    print(feature)

    print(
        f"Mean (No CVD): {mean_0:.4f}"
    )

    print(
        f"Mean (CVD): {mean_1:.4f}"
    )

    print(
        f"Median (No CVD): {median_0:.4f}"
    )

    print(
        f"Median (CVD): {median_1:.4f}"
    )

    print(
        f"Mann-Whitney U: {statistic:.4f}"
    )

    print(
        f"P-value: {p_value:.6f}"
    )

    if p_value < 0.05:
        print(
            "Result: STATISTICALLY SIGNIFICANT"
        )
    else:
        print(
            "Result: Not statistically significant"
        )


continuous_results_df = pd.DataFrame(
    continuous_results
)


# ============================================================
# 5. MULTIPLE TESTING CORRECTION
# ============================================================

print("\n")
print("=" * 70)
print("5. MULTIPLE TESTING CORRECTION")
print("=" * 70)

# Combine all p-values

all_pvalues = []

for p in chi_results_df["p_value"]:
    all_pvalues.append(p)

for p in continuous_results_df["p_value"]:
    all_pvalues.append(p)


reject, corrected_pvalues, _, _ = multipletests(
    all_pvalues,
    alpha=0.05,
    method="fdr_bh"
)

print("\nBenjamini-Hochberg FDR correction applied.")

print(
    f"Number of statistically significant "
    f"tests after correction: {reject.sum()}"
)


# ============================================================
# 6. SAVE RESULTS
# ============================================================

chi_results_df.to_csv(
    "output/eda2_chi_square_results.csv",
    index=False
)

odds_results_df.to_csv(
    "output/eda2_odds_ratio_results.csv",
    index=False
)

continuous_results_df.to_csv(
    "output/eda2_continuous_results.csv",
    index=False
)


print("\n")
print("=" * 70)
print("EDA 2 COMPLETE")
print("=" * 70)

print(
    "\nResults saved:"
)

print(
    "1. output/eda2_chi_square_results.csv"
)

print(
    "2. output/eda2_odds_ratio_results.csv"
)

print(
    "3. output/eda2_continuous_results.csv"
)
