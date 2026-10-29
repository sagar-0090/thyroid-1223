import pandas as pd
import numpy as np

from scipy.stats import mannwhitneyu, chi2_contingency
from statsmodels.stats.multitest import multipletests


# ======================================================================
# STEP 4 - CVD RISK-FACTOR CONTRIBUTION IN THYROID PATIENTS
# ======================================================================

print("=" * 75)
print("STEP 4 - CVD RISK-FACTOR CONTRIBUTION IN THYROID PATIENTS")
print("=" * 75)


# ======================================================================
# 1. LOAD STEP-3 DATA
# ======================================================================

input_file = "output/cvd_test_predictions_thresholded.csv"

df = pd.read_csv(input_file)

print("\nDataset loaded:")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ======================================================================
# 2. REQUIRED COLUMNS
# ======================================================================

required_columns = [

    # Identification / demographics
    "SEQN",
    "Age",
    "Gender",
    "Race",

    # Thyroid
    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO",
    "Thyroid_Diagnosis",
    "Abnormal_TSH",

    # Cardiovascular / metabolic
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

    # Lifestyle
    "Smoking_100_Cigarettes",
    "Smoking_Risk",
    "Risk_Factor_Count",

    # CVD
    "Cardiovascular_Disease",
    "CVD_Probability",
    "CVD_Prediction"

]


missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        f"\nMissing required columns: {missing_columns}"
    )

print("\nAll required columns are present.")


# ======================================================================
# 3. CREATE THYROID-PROBLEM SUBSET
# ======================================================================

print("\n" + "=" * 75)
print("1. IDENTIFYING THYROID-PROBLEM POPULATION")
print("=" * 75)

# According to our dataset coding:
#
# 1 = Thyroid problem
# 2 = No thyroid problem
# 9 = Unknown / other
#
# We exclude 9 from the primary thyroid-vs-no-thyroid analysis.

thyroid_df = df[
    df["Thyroid_Diagnosis"].isin([1, 2])
].copy()

thyroid_df["Thyroid_Problem"] = (
    thyroid_df["Thyroid_Diagnosis"] == 1
).astype(int)


print("\nFull dataset:")
print("Records:", len(df))

print("\nThyroid analysis population:")
print("Records:", len(thyroid_df))

print("\nThyroid problem distribution:")
print(
    thyroid_df["Thyroid_Problem"].value_counts()
)

print("\nThyroid problem percentage:")
print(
    thyroid_df["Thyroid_Problem"]
    .value_counts(normalize=True)
    * 100
)


# ======================================================================
# 4. THYROID PROBLEM + CVD PREDICTION GROUPS
# ======================================================================

print("\n" + "=" * 75)
print("2. THYROID PROBLEM + CVD PREDICTION GROUPS")
print("=" * 75)


thyroid_problem_df = thyroid_df[
    thyroid_df["Thyroid_Problem"] == 1
].copy()


print("\nNumber of thyroid-problem patients:")
print(len(thyroid_problem_df))


print("\nCVD prediction distribution among thyroid-problem patients:")

print(
    thyroid_problem_df[
        "CVD_Prediction"
    ].value_counts()
)


print("\nCVD prediction percentage:")

print(
    thyroid_problem_df[
        "CVD_Prediction"
    ].value_counts(normalize=True) * 100
)


# ======================================================================
# 5. DEFINE CVD RISK FACTORS
# ======================================================================

risk_factor_features = [

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


print("\nRisk factors analyzed:")

for feature in risk_factor_features:
    print("-", feature)


# ======================================================================
# 6. NUMERICAL RISK-FACTOR COMPARISON
# ======================================================================

print("\n" + "=" * 75)
print("3. RISK-FACTOR COMPARISON WITHIN THYROID-PROBLEM PATIENTS")
print("=" * 75)


numerical_features = [

    "SYS_BP",
    "DIA_BP",
    "HDL",
    "Total_Cholesterol",
    "BMI",
    "Waist",
    "Weight",
    "Height",
    "Smoking_Risk",
    "Risk_Factor_Count"

]


numerical_results = []


for feature in numerical_features:

    group_0 = thyroid_problem_df.loc[
        thyroid_problem_df["CVD_Prediction"] == 0,
        feature
    ].dropna()

    group_1 = thyroid_problem_df.loc[
        thyroid_problem_df["CVD_Prediction"] == 1,
        feature
    ].dropna()


    print("\n" + "-" * 65)
    print(feature)

    print(
        "Predicted CVD = 0"
    )

    print(
        "  N:",
        len(group_0)
    )

    print(
        "  Mean:",
        group_0.mean()
    )

    print(
        "  Median:",
        group_0.median()
    )


    print(
        "Predicted CVD = 1"
    )

    print(
        "  N:",
        len(group_1)
    )

    print(
        "  Mean:",
        group_1.mean()
    )

    print(
        "  Median:",
        group_1.median()
    )


    if len(group_0) > 0 and len(group_1) > 0:

        statistic, p_value = mannwhitneyu(
            group_0,
            group_1,
            alternative="two-sided"
        )

    else:

        statistic = np.nan
        p_value = np.nan


    print(
        "  Mann-Whitney U:",
        statistic
    )

    print(
        "  P-value:",
        p_value
    )


    numerical_results.append({

        "Feature": feature,

        "N_CVD_0": len(group_0),
        "N_CVD_1": len(group_1),

        "Mean_CVD_0": group_0.mean(),
        "Mean_CVD_1": group_1.mean(),

        "Median_CVD_0": group_0.median(),
        "Median_CVD_1": group_1.median(),

        "Mann_Whitney_U": statistic,
        "p_value": p_value

    })


# ======================================================================
# 7. CATEGORICAL RISK-FACTOR ANALYSIS
# ======================================================================

print("\n" + "=" * 75)
print("4. CATEGORICAL RISK-FACTOR ANALYSIS")
print("=" * 75)


categorical_features = [

    "High_BP",
    "High_Cholesterol",
    "Low_HDL",
    "Obesity",
    "Diabetes",
    "Smoking_100_Cigarettes"

]


categorical_results = []


for feature in categorical_features:

    print("\n" + "-" * 65)
    print("Feature:", feature)


    table = pd.crosstab(
        thyroid_problem_df[feature],
        thyroid_problem_df["CVD_Prediction"]
    )


    print("\nContingency table:")
    print(table)


    if table.shape >= (2, 2):

        chi2, p_value, dof, expected = (
            chi2_contingency(table)
        )

    else:

        chi2 = np.nan
        p_value = np.nan
        dof = np.nan


    print(
        "\nChi-square:",
        chi2
    )

    print(
        "Degrees of freedom:",
        dof
    )

    print(
        "P-value:",
        p_value
    )


    percentage_table = pd.crosstab(
        thyroid_problem_df[feature],
        thyroid_problem_df["CVD_Prediction"],
        normalize="index"
    ) * 100


    print(
        "\nCVD prediction percentage:"
    )

    print(
        percentage_table
    )


    categorical_results.append({

        "Feature": feature,
        "Chi2": chi2,
        "Degrees_of_Freedom": dof,
        "p_value": p_value

    })


# ======================================================================
# 8. THYROID BIOMARKERS WITHIN THYROID-PROBLEM PATIENTS
# ======================================================================

print("\n" + "=" * 75)
print("5. THYROID BIOMARKERS VS MODEL CVD")
print("=" * 75)


thyroid_biomarkers = [

    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO"

]


thyroid_results = []


for feature in thyroid_biomarkers:

    group_0 = thyroid_problem_df.loc[
        thyroid_problem_df["CVD_Prediction"] == 0,
        feature
    ].dropna()

    group_1 = thyroid_problem_df.loc[
        thyroid_problem_df["CVD_Prediction"] == 1,
        feature
    ].dropna()


    print("\n" + "-" * 65)
    print(feature)


    print(
        "Mean CVD = 0:",
        group_0.mean()
    )

    print(
        "Mean CVD = 1:",
        group_1.mean()
    )

    print(
        "Median CVD = 0:",
        group_0.median()
    )

    print(
        "Median CVD = 1:",
        group_1.median()
    )


    if len(group_0) > 0 and len(group_1) > 0:

        statistic, p_value = mannwhitneyu(
            group_0,
            group_1,
            alternative="two-sided"
        )

    else:

        statistic = np.nan
        p_value = np.nan


    print(
        "Mann-Whitney U:",
        statistic
    )

    print(
        "P-value:",
        p_value
    )


    thyroid_results.append({

        "Feature": feature,

        "Mean_CVD_0": group_0.mean(),
        "Mean_CVD_1": group_1.mean(),

        "Median_CVD_0": group_0.median(),
        "Median_CVD_1": group_1.median(),

        "U_Statistic": statistic,
        "p_value": p_value

    })


# ======================================================================
# 9. FT3 / FT4 RATIO
# ======================================================================

print("\n" + "=" * 75)
print("6. FT3 / FT4 RATIO ANALYSIS")
print("=" * 75)


thyroid_problem_df["FT3_FT4_Ratio"] = (
    thyroid_problem_df["Free_T3"]
    / thyroid_problem_df["Free_T4"]
)


print("\nFT3/FT4 ratio summary:")

print(
    thyroid_problem_df[
        "FT3_FT4_Ratio"
    ].describe()
)


ratio_0 = thyroid_problem_df.loc[
    thyroid_problem_df["CVD_Prediction"] == 0,
    "FT3_FT4_Ratio"
].dropna()


ratio_1 = thyroid_problem_df.loc[
    thyroid_problem_df["CVD_Prediction"] == 1,
    "FT3_FT4_Ratio"
].dropna()


if len(ratio_0) > 0 and len(ratio_1) > 0:

    ratio_u, ratio_p = mannwhitneyu(
        ratio_0,
        ratio_1,
        alternative="two-sided"
    )

else:

    ratio_u = np.nan
    ratio_p = np.nan


print("\nMean FT3/FT4 ratio:")
print("CVD = 0:", ratio_0.mean())
print("CVD = 1:", ratio_1.mean())

print("\nMedian FT3/FT4 ratio:")
print("CVD = 0:", ratio_0.median())
print("CVD = 1:", ratio_1.median())

print("\nMann-Whitney U:", ratio_u)
print("P-value:", ratio_p)


thyroid_results.append({

    "Feature": "FT3_FT4_Ratio",

    "Mean_CVD_0": ratio_0.mean(),
    "Mean_CVD_1": ratio_1.mean(),

    "Median_CVD_0": ratio_0.median(),
    "Median_CVD_1": ratio_1.median(),

    "U_Statistic": ratio_u,
    "p_value": ratio_p

})


# ======================================================================
# 10. SPEARMAN CORRELATION WITH CVD PROBABILITY
# ======================================================================

print("\n" + "=" * 75)
print("7. CORRELATION WITH CVD PROBABILITY")
print("=" * 75)


correlation_features = [

    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO",
    "FT3_FT4_Ratio"

]


correlation_results = []


for feature in correlation_features:

    temp = thyroid_problem_df[
        [
            feature,
            "CVD_Probability"
        ]
    ].dropna()


    correlation = temp[
        feature
    ].corr(
        temp["CVD_Probability"],
        method="spearman"
    )


    print("\n" + "-" * 65)

    print(
        feature,
        "Spearman correlation:",
        correlation
    )


    correlation_results.append({

        "Feature": feature,

        "Spearman_Correlation": correlation,

        "N": len(temp)

    })


# ======================================================================
# 11. RISK FACTOR COUNT VS CVD PROBABILITY
# ======================================================================

print("\n" + "=" * 75)
print("8. RISK FACTOR COUNT VS CVD PROBABILITY")
print("=" * 75)


risk_count_correlation = thyroid_problem_df[
    [
        "Risk_Factor_Count",
        "CVD_Probability"
    ]
].dropna()


risk_count_corr = risk_count_correlation[
    "Risk_Factor_Count"
].corr(
    risk_count_correlation[
        "CVD_Probability"
    ],
    method="spearman"
)


print(
    "\nSpearman correlation:",
    risk_count_corr
)


# ======================================================================
# 12. THYROID PROBLEM VS NO THYROID PROBLEM
# ======================================================================

print("\n" + "=" * 75)
print("9. THYROID PROBLEM VS NO THYROID PROBLEM")
print("=" * 75)


thyroid_comparison = pd.crosstab(

    thyroid_df["Thyroid_Problem"],

    thyroid_df["CVD_Prediction"]

)


print("\nContingency table:")
print(thyroid_comparison)


thyroid_comparison_percent = pd.crosstab(

    thyroid_df["Thyroid_Problem"],

    thyroid_df["CVD_Prediction"],

    normalize="index"

) * 100


print("\nCVD prediction percentage:")
print(thyroid_comparison_percent)


# ======================================================================
# 13. ODDS RATIO
# ======================================================================

print("\n" + "=" * 75)
print("10. ODDS RATIO - THYROID PROBLEM AND MODEL CVD")
print("=" * 75)


table = thyroid_comparison


if table.shape == (2, 2):

    # Rows:
    # 0 = No thyroid problem
    # 1 = Thyroid problem
    #
    # Columns:
    # 0 = CVD prediction negative
    # 1 = CVD prediction positive

    a = table.loc[0, 0]
    b = table.loc[0, 1]
    c = table.loc[1, 0]
    d = table.loc[1, 1]


    if a > 0 and b > 0 and c > 0 and d > 0:

        odds_ratio = (d * a) / (c * b)

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

    else:

        odds_ratio = np.nan
        ci_lower = np.nan
        ci_upper = np.nan


else:

    odds_ratio = np.nan
    ci_lower = np.nan
    ci_upper = np.nan


print("\nOdds Ratio:", odds_ratio)

print(
    "95% CI:",
    (ci_lower, ci_upper)
)


# ======================================================================
# 14. MULTIPLE TESTING CORRECTION
# ======================================================================

print("\n" + "=" * 75)
print("11. MULTIPLE TESTING CORRECTION")
print("=" * 75)


# ------------------------------
# Numerical risk factors
# ------------------------------

mw_risk_pvalues = [

    result["p_value"]

    for result in numerical_results

    if not pd.isna(
        result["p_value"]
    )

]


if mw_risk_pvalues:

    corrected = multipletests(

        mw_risk_pvalues,

        method="fdr_bh"

    )

    corrected_pvalues = corrected[1]

    index = 0


    for result in numerical_results:

        if not pd.isna(
            result["p_value"]
        ):

            result["FDR_Adjusted_p"] = (
                corrected_pvalues[index]
            )

            result["FDR_Significant"] = (
                corrected_pvalues[index] < 0.05
            )

            index += 1


# ------------------------------
# Categorical risk factors
# ------------------------------

chi_pvalues = [

    result["p_value"]

    for result in categorical_results

    if not pd.isna(
        result["p_value"]
    )

]


if chi_pvalues:

    corrected = multipletests(

        chi_pvalues,

        method="fdr_bh"

    )

    corrected_pvalues = corrected[1]


    index = 0

    for result in categorical_results:

        if not pd.isna(
            result["p_value"]
        ):

            result["FDR_Adjusted_p"] = (
                corrected_pvalues[index]
            )

            result["FDR_Significant"] = (
                corrected_pvalues[index] < 0.05
            )

            index += 1


# ------------------------------
# Thyroid biomarkers
# ------------------------------

thyroid_pvalues = [

    result["p_value"]

    for result in thyroid_results

    if not pd.isna(
        result["p_value"]
    )

]


if thyroid_pvalues:

    corrected = multipletests(

        thyroid_pvalues,

        method="fdr_bh"

    )

    corrected_pvalues = corrected[1]


    index = 0

    for result in thyroid_results:

        if not pd.isna(
            result["p_value"]
        ):

            result["FDR_Adjusted_p"] = (
                corrected_pvalues[index]
            )

            result["FDR_Significant"] = (
                corrected_pvalues[index] < 0.05
            )

            index += 1


print(
    "\nBenjamini-Hochberg FDR correction applied."
)


# ======================================================================
# 15. SAVE RESULTS
# ======================================================================

print("\n" + "=" * 75)
print("12. SAVING RESULTS")
print("=" * 75)


output_1 = (
    "output/step4_numerical_risk_factors.csv"
)

output_2 = (
    "output/step4_categorical_risk_factors.csv"
)

output_3 = (
    "output/step4_thyroid_biomarkers.csv"
)

output_4 = (
    "output/step4_thyroid_probability_correlations.csv"
)

output_5 = (
    "output/step4_thyroid_cvd_comparison.csv"
)

output_6 = (
    "output/step4_thyroid_problem_population.csv"
)


pd.DataFrame(
    numerical_results
).to_csv(
    output_1,
    index=False
)


pd.DataFrame(
    categorical_results
).to_csv(
    output_2,
    index=False
)


pd.DataFrame(
    thyroid_results
).to_csv(
    output_3,
    index=False
)


pd.DataFrame(
    correlation_results
).to_csv(
    output_4,
    index=False
)


thyroid_comparison_percent.to_csv(
    output_5
)


thyroid_problem_df.to_csv(
    output_6,
    index=False
)


# ======================================================================
# 16. FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 75)
print("STEP 4 COMPLETE")
print("=" * 75)


print("\nThyroid-problem patients analyzed:")
print(
    len(thyroid_problem_df)
)


print("\nPredicted CVD among thyroid-problem patients:")

print(
    thyroid_problem_df[
        "CVD_Prediction"
    ].value_counts()
)


print("\nPredicted CVD percentage:")

print(
    thyroid_problem_df[
        "CVD_Prediction"
    ].mean() * 100
)


print("\nThyroid biomarkers analyzed:")

for feature in correlation_features:

    print("-", feature)


print("\nRisk factors analyzed:")

for feature in risk_factor_features:

    print("-", feature)


print("\nOriginal Cardiovascular_Disease column:")
print("NOT MODIFIED")


print("\nFiles saved:")

print("1.", output_1)
print("2.", output_2)
print("3.", output_3)
print("4.", output_4)
print("5.", output_5)
print("6.", output_6)


print("\n" + "=" * 75)
