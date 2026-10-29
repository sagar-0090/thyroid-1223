import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, mannwhitneyu
from statsmodels.stats.multitest import multipletests


# ==============================================================
# STEP 3 - THYROID-SPECIFIC CVD ANALYSIS
# ==============================================================

print("=" * 70)
print("STEP 3 - THYROID-SPECIFIC CVD ANALYSIS")
print("=" * 70)


# --------------------------------------------------------------
# 1. LOAD STEP-2 DATA
# --------------------------------------------------------------

input_file = "output/cvd_test_predictions_thresholded.csv"

df = pd.read_csv(input_file)

print("\nDataset loaded:")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# --------------------------------------------------------------
# 2. CHECK REQUIRED COLUMNS
# --------------------------------------------------------------

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

print("\nAll required thyroid and CVD columns are present.")


# --------------------------------------------------------------
# 3. BASIC DISTRIBUTIONS
# --------------------------------------------------------------

print("\n" + "=" * 70)
print("1. BASIC DISTRIBUTIONS")
print("=" * 70)

print("\nQuestionnaire CVD:")
print(df["Cardiovascular_Disease"].value_counts())

print("\nModel CVD:")
print(df["CVD_Prediction"].value_counts())

print("\nThyroid Diagnosis:")
print(df["Thyroid_Diagnosis"].value_counts().sort_index())

print("\nAbnormal TSH:")
print(df["Abnormal_TSH"].value_counts())


# --------------------------------------------------------------
# 4. THYROID DIAGNOSIS VS MODEL CVD
# --------------------------------------------------------------

print("\n" + "=" * 70)
print("2. THYROID DIAGNOSIS VS MODEL CVD")
print("=" * 70)

thyroid_cvd_table = pd.crosstab(
    df["Thyroid_Diagnosis"],
    df["CVD_Prediction"],
    margins=True
)

print("\nContingency table:")
print(thyroid_cvd_table)


thyroid_cvd_percent = pd.crosstab(
    df["Thyroid_Diagnosis"],
    df["CVD_Prediction"],
    normalize="index"
) * 100

print("\nCVD prediction percentage within each thyroid diagnosis:")
print(thyroid_cvd_percent)


# --------------------------------------------------------------
# 5. THYROID DIAGNOSIS VS CVD PROBABILITY
# --------------------------------------------------------------

print("\n" + "=" * 70)
print("3. CVD PROBABILITY BY THYROID DIAGNOSIS")
print("=" * 70)

thyroid_probability = (
    df.groupby("Thyroid_Diagnosis")["CVD_Probability"]
    .agg(
        Count="count",
        Mean="mean",
        Median="median",
        Std="std",
        Min="min",
        Max="max"
    )
)

print(thyroid_probability)


# --------------------------------------------------------------
# 6. ABNORMAL TSH VS MODEL CVD
# --------------------------------------------------------------

print("\n" + "=" * 70)
print("4. ABNORMAL TSH VS MODEL CVD")
print("=" * 70)

abnormal_tsh_table = pd.crosstab(
    df["Abnormal_TSH"],
    df["CVD_Prediction"],
    margins=True
)

print("\nContingency table:")
print(abnormal_tsh_table)


abnormal_tsh_percent = pd.crosstab(
    df["Abnormal_TSH"],
    df["CVD_Prediction"],
    normalize="index"
) * 100

print("\nCVD prediction percentage by TSH status:")
print(abnormal_tsh_percent)


# --------------------------------------------------------------
# 7. CVD PROBABILITY BY TSH STATUS
# --------------------------------------------------------------

print("\n" + "=" * 70)
print("5. CVD PROBABILITY BY TSH STATUS")
print("=" * 70)

tsh_probability = (
    df.groupby("Abnormal_TSH")["CVD_Probability"]
    .agg(
        Count="count",
        Mean="mean",
        Median="median",
        Std="std",
        Min="min",
        Max="max"
    )
)

print(tsh_probability)


# --------------------------------------------------------------
# 8. THYROID BIOMARKERS BY MODEL CVD
# --------------------------------------------------------------

print("\n" + "=" * 70)
print("6. THYROID BIOMARKERS BY MODEL CVD")
print("=" * 70)

thyroid_biomarkers = [
    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO"
]

biomarker_results = []

for feature in thyroid_biomarkers:

    no_cvd = df.loc[
        df["CVD_Prediction"] == 0,
        feature
    ].dropna()

    predicted_cvd = df.loc[
        df["CVD_Prediction"] == 1,
        feature
    ].dropna()

    print("\n" + "-" * 60)
    print(feature)

    print("Mean (Predicted CVD = 0):", no_cvd.mean())
    print("Mean (Predicted CVD = 1):", predicted_cvd.mean())

    print("Median (Predicted CVD = 0):", no_cvd.median())
    print("Median (Predicted CVD = 1):", predicted_cvd.median())

    if len(no_cvd) > 0 and len(predicted_cvd) > 0:

        u_stat, p_value = mannwhitneyu(
            no_cvd,
            predicted_cvd,
            alternative="two-sided"
        )

        print("Mann-Whitney U:", u_stat)
        print("P-value:", p_value)

        if p_value < 0.05:
            result = "STATISTICALLY SIGNIFICANT"
        else:
            result = "NOT STATISTICALLY SIGNIFICANT"

        print("Result:", result)

    else:
        u_stat = np.nan
        p_value = np.nan
        result = "INSUFFICIENT DATA"

    biomarker_results.append({
        "Feature": feature,
        "Mean_No_Predicted_CVD": no_cvd.mean(),
        "Mean_Predicted_CVD": predicted_cvd.mean(),
        "Median_No_Predicted_CVD": no_cvd.median(),
        "Median_Predicted_CVD": predicted_cvd.median(),
        "U_Statistic": u_stat,
        "p_value": p_value,
        "Result": result
    })


# --------------------------------------------------------------
# 9. CORRELATION WITH CVD PROBABILITY
# --------------------------------------------------------------

print("\n" + "=" * 70)
print("7. THYROID BIOMARKER CORRELATION WITH CVD PROBABILITY")
print("=" * 70)

correlation_results = []

for feature in thyroid_biomarkers:

    temp = df[
        [feature, "CVD_Probability"]
    ].dropna()

    correlation = temp[feature].corr(
        temp["CVD_Probability"],
        method="spearman"
    )

    print("\n" + "-" * 60)
    print(feature)
    print("Spearman correlation:", correlation)

    correlation_results.append({
        "Feature": feature,
        "Spearman_Correlation": correlation,
        "N": len(temp)
    })


# --------------------------------------------------------------
# 10. CHI-SQUARE TESTS
# --------------------------------------------------------------

print("\n" + "=" * 70)
print("8. CHI-SQUARE TESTS")
print("=" * 70)

chi_square_results = []


# Thyroid diagnosis
categorical_features = [
    "Thyroid_Diagnosis",
    "Abnormal_TSH"
]

for feature in categorical_features:

    print("\n" + "-" * 60)
    print("Feature:", feature)

    table = pd.crosstab(
        df[feature],
        df["CVD_Prediction"]
    )

    print("\nContingency table:")
    print(table)

    chi2, p_value, dof, expected = chi2_contingency(table)

    print("\nChi-square statistic:", chi2)
    print("Degrees of freedom:", dof)
    print("P-value:", p_value)

    if p_value < 0.05:
        result = "STATISTICALLY SIGNIFICANT"
    else:
        result = "NOT STATISTICALLY SIGNIFICANT"

    print("Result:", result)

    chi_square_results.append({
        "Feature": feature,
        "Chi2": chi2,
        "Degrees_of_Freedom": dof,
        "p_value": p_value,
        "Result": result
    })


# --------------------------------------------------------------
# 11. ODDS RATIO FOR ABNORMAL TSH
# --------------------------------------------------------------

print("\n" + "=" * 70)
print("9. ODDS RATIO - ABNORMAL TSH")
print("=" * 70)

table = pd.crosstab(
    df["Abnormal_TSH"],
    df["CVD_Prediction"]
)

print("\nContingency table:")
print(table)


if table.shape == (2, 2):

    # Layout:
    #             CVD 0    CVD 1
    # TSH 0
    # TSH 1

    a = table.iloc[0, 0]
    b = table.iloc[0, 1]
    c = table.iloc[1, 0]
    d = table.iloc[1, 1]

    # Odds ratio:
    # (d/c) / (b/a)
    odds_ratio = (d * a) / (c * b)

    # Standard error of log OR
    se = np.sqrt(
        (1 / a) +
        (1 / b) +
        (1 / c) +
        (1 / d)
    )

    log_or = np.log(odds_ratio)

    ci_lower = np.exp(
        log_or - 1.96 * se
    )

    ci_upper = np.exp(
        log_or + 1.96 * se
    )

    print("\nOdds Ratio:", odds_ratio)
    print("95% CI:", (ci_lower, ci_upper))

    if odds_ratio > 1:
        print(
            "Interpretation: Abnormal TSH is associated "
            "with higher observed odds of model-predicted CVD."
        )
    else:
        print(
            "Interpretation: Abnormal TSH is associated "
            "with lower observed odds of model-predicted CVD."
        )

else:

    odds_ratio = np.nan
    ci_lower = np.nan
    ci_upper = np.nan

    print(
        "\nOdds ratio could not be calculated "
        "because the contingency table is not 2x2."
    )


# --------------------------------------------------------------
# 12. THYROID DIAGNOSIS - BINARY THYROID PROBLEM
# --------------------------------------------------------------

print("\n" + "=" * 70)
print("10. THYROID PROBLEM VS MODEL CVD")
print("=" * 70)


# Assuming:
# 1 = thyroid problem
# 2 = no thyroid problem
# 9 = unknown / other
#
# We exclude 9 from the binary comparison.

thyroid_binary = df[
    df["Thyroid_Diagnosis"].isin([1, 2])
].copy()

thyroid_binary["Thyroid_Problem"] = (
    thyroid_binary["Thyroid_Diagnosis"] == 1
).astype(int)


binary_table = pd.crosstab(
    thyroid_binary["Thyroid_Problem"],
    thyroid_binary["CVD_Prediction"]
)

print("\nContingency table:")
print(binary_table)


binary_percent = pd.crosstab(
    thyroid_binary["Thyroid_Problem"],
    thyroid_binary["CVD_Prediction"],
    normalize="index"
) * 100

print("\nCVD prediction percentage:")
print(binary_percent)


chi2, p_value, dof, expected = chi2_contingency(
    binary_table
)

print("\nChi-square statistic:", chi2)
print("P-value:", p_value)

if binary_table.shape == (2, 2):

    a = binary_table.iloc[0, 0]
    b = binary_table.iloc[0, 1]
    c = binary_table.iloc[1, 0]
    d = binary_table.iloc[1, 1]

    # Thyroid problem = 1
    # No thyroid problem = 0

    odds_ratio_thyroid = (b * c) / (a * d)

    se = np.sqrt(
        (1 / a) +
        (1 / b) +
        (1 / c) +
        (1 / d)
    )

    log_or = np.log(odds_ratio_thyroid)

    ci_lower_thyroid = np.exp(
        log_or - 1.96 * se
    )

    ci_upper_thyroid = np.exp(
        log_or + 1.96 * se
    )

    print("\nOdds Ratio:", odds_ratio_thyroid)
    print(
        "95% CI:",
        (ci_lower_thyroid, ci_upper_thyroid)
    )


# --------------------------------------------------------------
# 13. MULTIPLE TESTING CORRECTION
# --------------------------------------------------------------

print("\n" + "=" * 70)
print("11. MULTIPLE TESTING CORRECTION")
print("=" * 70)


# Mann-Whitney p-values
mw_pvalues = [
    result["p_value"]
    for result in biomarker_results
    if not pd.isna(result["p_value"])
]

if mw_pvalues:

    corrected = multipletests(
        mw_pvalues,
        method="fdr_bh"
    )

    corrected_pvalues = corrected[1]

    index = 0

    for result in biomarker_results:

        if not pd.isna(result["p_value"]):

            result["FDR_Adjusted_p"] = (
                corrected_pvalues[index]
            )

            result["FDR_Significant"] = (
                corrected_pvalues[index] < 0.05
            )

            index += 1


# Chi-square p-values
chi_pvalues = [
    result["p_value"]
    for result in chi_square_results
]

if chi_pvalues:

    corrected_chi = multipletests(
        chi_pvalues,
        method="fdr_bh"
    )

    corrected_chi_pvalues = corrected_chi[1]

    for i, result in enumerate(chi_square_results):

        result["FDR_Adjusted_p"] = (
            corrected_chi_pvalues[i]
        )

        result["FDR_Significant"] = (
            corrected_chi_pvalues[i] < 0.05
        )


print(
    "\nBenjamini-Hochberg FDR correction "
    "applied to statistical tests."
)


# --------------------------------------------------------------
# 14. SAVE RESULTS
# --------------------------------------------------------------

print("\n" + "=" * 70)
print("12. SAVING RESULTS")
print("=" * 70)


thyroid_probability.to_csv(
    "output/step3_thyroid_cvd_probability.csv"
)

thyroid_cvd_percent.to_csv(
    "output/step3_thyroid_cvd_percent.csv"
)

abnormal_tsh_percent.to_csv(
    "output/step3_abnormal_tsh_cvd_percent.csv"
)

pd.DataFrame(
    biomarker_results
).to_csv(
    "output/step3_thyroid_biomarker_tests.csv",
    index=False
)

pd.DataFrame(
    correlation_results
).to_csv(
    "output/step3_thyroid_probability_correlations.csv",
    index=False
)

pd.DataFrame(
    chi_square_results
).to_csv(
    "output/step3_thyroid_chi_square.csv",
    index=False
)


# --------------------------------------------------------------
# 15. FINAL SUMMARY
# --------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3 COMPLETE")
print("=" * 70)

print("\nThyroid variables analyzed:")
for feature in thyroid_biomarkers:
    print("-", feature)

print("\nCategorical thyroid variables analyzed:")
print("- Thyroid_Diagnosis")
print("- Abnormal_TSH")

print("\nCVD outcome used:")
print("- CVD_Prediction")

print("\nCVD probability used:")
print("- CVD_Probability")

print("\nOriginal Cardiovascular_Disease column was NOT modified.")

print("\nFiles saved:")
print("1. output/step3_thyroid_cvd_probability.csv")
print("2. output/step3_thyroid_cvd_percent.csv")
print("3. output/step3_abnormal_tsh_cvd_percent.csv")
print("4. output/step3_thyroid_biomarker_tests.csv")
print("5. output/step3_thyroid_probability_correlations.csv")
print("6. output/step3_thyroid_chi_square.csv")

print("\n" + "=" * 70)
