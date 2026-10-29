# ==============================================================
# STEP 7 - THYROID FACTORS AND CVD ASSOCIATION ANALYSIS
# ==============================================================

import pandas as pd
import numpy as np

from scipy.stats import (
    mannwhitneyu,
    chi2_contingency,
    spearmanr
)

from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm


print("=" * 75)
print("STEP 7 - THYROID FACTORS AND CVD ASSOCIATION ANALYSIS")
print("=" * 75)


# --------------------------------------------------------------
# 1. LOAD STEP-5 INTEGRATED DATA
# --------------------------------------------------------------

input_file = "output/step5_integrated_thyroid_cvd_dataset.csv"

df = pd.read_csv(input_file)

print("\nDataset loaded:")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# --------------------------------------------------------------
# 2. CHECK REQUIRED COLUMNS
# --------------------------------------------------------------

required_columns = [
    "SEQN",

    # Thyroid variables
    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO",
    "Thyroid_Diagnosis",
    "Abnormal_TSH",
    "Thyroid_Problem",
    "FT3_FT4_Ratio",

    # CVD outcome
    "CVD_Probability",
    "CVD_Prediction",

    # Conventional CVD variables
    "Age",
    "BMI",
    "SYS_BP",
    "HDL",
    "Total_Cholesterol",
    "Diabetes",
    "Smoking_100_Cigarettes",

    # Original questionnaire CVD
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


# --------------------------------------------------------------
# 3. SELECT THYROID-PROBLEM PARTICIPANTS
# --------------------------------------------------------------

print("\n" + "=" * 75)
print("1. THYROID-PROBLEM POPULATION")
print("=" * 75)


thyroid_df = df[
    df["Thyroid_Problem"] == 1
].copy()


print("\nTotal records in input dataset:")
print(len(df))


print("\nThyroid-problem participants:")
print(len(thyroid_df))


print("\nCVD prediction distribution:")
print(
    thyroid_df["CVD_Prediction"]
    .value_counts()
    .sort_index()
)


cvd_percent = (
    thyroid_df["CVD_Prediction"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)


print("\nCVD prediction percentage:")
print(cvd_percent)


# --------------------------------------------------------------
# 4. DEFINE THYROID VARIABLES
# --------------------------------------------------------------

print("\n" + "=" * 75)
print("2. THYROID VARIABLES")
print("=" * 75)


thyroid_biomarkers = [
    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO",
    "FT3_FT4_Ratio"
]


print("\nThyroid biomarkers analyzed:")

for feature in thyroid_biomarkers:
    print("-", feature)


# --------------------------------------------------------------
# 5. DESCRIPTIVE COMPARISON
# --------------------------------------------------------------

print("\n" + "=" * 75)
print("3. THYROID BIOMARKERS BY MODEL-PREDICTED CVD")
print("=" * 75)


biomarker_results = []


for feature in thyroid_biomarkers:

    no_cvd = thyroid_df.loc[
        thyroid_df["CVD_Prediction"] == 0,
        feature
    ].dropna()

    predicted_cvd = thyroid_df.loc[
        thyroid_df["CVD_Prediction"] == 1,
        feature
    ].dropna()


    print("\n" + "-" * 65)
    print(feature)

    print("\nPredicted CVD = 0")
    print("N:", len(no_cvd))
    print("Mean:", no_cvd.mean())
    print("Median:", no_cvd.median())
    print("Std:", no_cvd.std())

    print("\nPredicted CVD = 1")
    print("N:", len(predicted_cvd))
    print("Mean:", predicted_cvd.mean())
    print("Median:", predicted_cvd.median())
    print("Std:", predicted_cvd.std())


    if len(no_cvd) > 0 and len(predicted_cvd) > 0:

        u_stat, p_value = mannwhitneyu(
            no_cvd,
            predicted_cvd,
            alternative="two-sided"
        )

    else:

        u_stat = np.nan
        p_value = np.nan


    biomarker_results.append({

        "Feature": feature,

        "N_No_CVD": len(no_cvd),
        "N_Predicted_CVD": len(predicted_cvd),

        "Mean_No_CVD": no_cvd.mean(),
        "Mean_Predicted_CVD": predicted_cvd.mean(),

        "Median_No_CVD": no_cvd.median(),
        "Median_Predicted_CVD": predicted_cvd.median(),

        "Std_No_CVD": no_cvd.std(),
        "Std_Predicted_CVD": predicted_cvd.std(),

        "Mean_Difference":
            predicted_cvd.mean() - no_cvd.mean(),

        "Mann_Whitney_U": u_stat,
        "p_value": p_value
    })


biomarker_results_df = pd.DataFrame(
    biomarker_results
)


# --------------------------------------------------------------
# 6. FDR CORRECTION
# --------------------------------------------------------------

print("\n" + "=" * 75)
print("4. MULTIPLE TESTING CORRECTION")
print("=" * 75)


valid_p = biomarker_results_df[
    biomarker_results_df["p_value"].notna()
]["p_value"]


if len(valid_p) > 0:

    rejected, corrected_p, _, _ = multipletests(
        valid_p,
        method="fdr_bh"
    )


    biomarker_results_df.loc[
        biomarker_results_df["p_value"].notna(),
        "FDR_Adjusted_p"
    ] = corrected_p


    biomarker_results_df.loc[
        biomarker_results_df["p_value"].notna(),
        "FDR_Significant"
    ] = rejected


print("\nBenjamini-Hochberg FDR correction applied.")


# --------------------------------------------------------------
# 7. SPEARMAN CORRELATION WITH CVD PROBABILITY
# --------------------------------------------------------------

print("\n" + "=" * 75)
print("5. THYROID BIOMARKERS VS CVD PROBABILITY")
print("=" * 75)


correlation_results = []


for feature in thyroid_biomarkers:

    temp = thyroid_df[
        [feature, "CVD_Probability"]
    ].dropna()


    if len(temp) >= 3:

        correlation, p_value = spearmanr(
            temp[feature],
            temp["CVD_Probability"]
        )

    else:

        correlation = np.nan
        p_value = np.nan


    print("\n" + "-" * 65)
    print(feature)

    print("N:", len(temp))
    print("Spearman correlation:", correlation)
    print("P-value:", p_value)


    correlation_results.append({

        "Feature": feature,

        "N": len(temp),

        "Spearman_Correlation":
            correlation,

        "p_value":
            p_value
    })


correlation_results_df = pd.DataFrame(
    correlation_results
)


# --------------------------------------------------------------
# 8. FDR CORRECTION FOR CORRELATIONS
# --------------------------------------------------------------

valid_corr_p = correlation_results_df[
    correlation_results_df["p_value"].notna()
]["p_value"]


if len(valid_corr_p) > 0:

    rejected_corr, corrected_corr_p, _, _ = multipletests(
        valid_corr_p,
        method="fdr_bh"
    )


    correlation_results_df.loc[
        correlation_results_df["p_value"].notna(),
        "FDR_Adjusted_p"
    ] = corrected_corr_p


    correlation_results_df.loc[
        correlation_results_df["p_value"].notna(),
        "FDR_Significant"
    ] = rejected_corr


# --------------------------------------------------------------
# 9. ABNORMAL TSH VS CVD
# --------------------------------------------------------------

print("\n" + "=" * 75)
print("6. ABNORMAL TSH VS MODEL-PREDICTED CVD")
print("=" * 75)


tsh_table = pd.crosstab(
    thyroid_df["Abnormal_TSH"],
    thyroid_df["CVD_Prediction"]
)


print("\nContingency table:")
print(tsh_table)


tsh_percent = (
    pd.crosstab(
        thyroid_df["Abnormal_TSH"],
        thyroid_df["CVD_Prediction"],
        normalize="index"
    ) * 100
)


print("\nCVD percentage within TSH groups:")
print(tsh_percent)


if tsh_table.shape == (2, 2):

    chi2, chi_p, dof, expected = chi2_contingency(
        tsh_table
    )

    print("\nChi-square:", chi2)
    print("Degrees of freedom:", dof)
    print("P-value:", chi_p)

else:

    chi2 = np.nan
    chi_p = np.nan
    dof = np.nan


# --------------------------------------------------------------
# 10. THYROID DIAGNOSIS VS CVD
# --------------------------------------------------------------

print("\n" + "=" * 75)
print("7. THYROID DIAGNOSIS VS MODEL-PREDICTED CVD")
print("=" * 75)


diagnosis_table = pd.crosstab(
    thyroid_df["Thyroid_Diagnosis"],
    thyroid_df["CVD_Prediction"]
)


print("\nContingency table:")
print(diagnosis_table)


if diagnosis_table.shape[0] >= 2:

    diagnosis_chi2, diagnosis_p, diagnosis_dof, diagnosis_expected = (
        chi2_contingency(diagnosis_table)
    )

    print("\nChi-square:", diagnosis_chi2)
    print("Degrees of freedom:", diagnosis_dof)
    print("P-value:", diagnosis_p)

else:

    diagnosis_chi2 = np.nan
    diagnosis_p = np.nan
    diagnosis_dof = np.nan


# --------------------------------------------------------------
# 11. THYROID BIOMARKERS + CONVENTIONAL CVD FACTORS
#     MULTIVARIABLE ASSOCIATION MODEL
# --------------------------------------------------------------

print("\n" + "=" * 75)
print("8. ADJUSTED THYROID-CVD ASSOCIATION")
print("=" * 75)


print(
    "\nPurpose:"
)

print(
    "Determine whether thyroid biomarkers are associated "
    "with model-predicted CVD after accounting for "
    "major conventional CVD factors."
)


# --------------------------------------------------------------
# IMPORTANT:
#
# CVD_Probability is NOT included.
# CVD_Prediction is derived from CVD_Probability.
#
# Diabetes and Smoking are treated as categorical variables.
# --------------------------------------------------------------


continuous_variables = [
    "Age",
    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO",
    "FT3_FT4_Ratio",
    "BMI",
    "SYS_BP",
    "HDL",
    "Total_Cholesterol"
]


categorical_variables = [
    "Diabetes",
    "Smoking_100_Cigarettes"
]


model_columns = (
    continuous_variables +
    categorical_variables +
    ["CVD_Prediction"]
)


model_df = thyroid_df[
    model_columns
].dropna().copy()


print("\nRecords available for adjusted analysis:")
print(len(model_df))


# --------------------------------------------------------------
# STANDARDIZE CONTINUOUS VARIABLES
# --------------------------------------------------------------

for feature in continuous_variables:

    mean_value = model_df[feature].mean()
    std_value = model_df[feature].std()

    if std_value > 0:

        model_df[
            feature + "_z"
        ] = (
            model_df[feature] - mean_value
        ) / std_value

    else:

        model_df[
            feature + "_z"
        ] = 0


# --------------------------------------------------------------
# CREATE DUMMY VARIABLES FOR CATEGORICAL VARIABLES
# --------------------------------------------------------------

model_df = pd.get_dummies(
    model_df,
    columns=categorical_variables,
    drop_first=True,
    dtype=float
)


predictor_columns = [
    feature + "_z"
    for feature in continuous_variables
]


dummy_columns = [
    col
    for col in model_df.columns
    if (
        col.startswith("Diabetes_")
        or
        col.startswith("Smoking_100_Cigarettes_")
    )
]


predictor_columns += dummy_columns


print("\nPredictors included:")
for feature in predictor_columns:
    print("-", feature)


# --------------------------------------------------------------
# REMOVE CONSTANT / ZERO-VARIANCE PREDICTORS
# --------------------------------------------------------------

valid_predictors = []


for feature in predictor_columns:

    if model_df[feature].nunique() > 1:
        valid_predictors.append(feature)


predictor_columns = valid_predictors


# --------------------------------------------------------------
# BUILD LOGISTIC REGRESSION
# --------------------------------------------------------------

X = model_df[
    predictor_columns
]

X = sm.add_constant(X)

y = model_df[
    "CVD_Prediction"
]


print("\nOutcome distribution:")
print(y.value_counts())


logit_model = sm.Logit(
    y,
    X
)


try:

    fitted_model = logit_model.fit(
        disp=False,
        maxiter=200
    )

    model_success = True

except Exception as e:

    print("\nWARNING: Logistic regression failed.")
    print("Error:", e)

    fitted_model = None
    model_success = False


# --------------------------------------------------------------
# 12. ADJUSTED ODDS RATIOS
# --------------------------------------------------------------

adjusted_results = []


if model_success:

    params = fitted_model.params
    pvalues = fitted_model.pvalues
    conf = fitted_model.conf_int()


    for feature in predictor_columns:

        coefficient = params[feature]

        odds_ratio = np.exp(
            coefficient
        )

        ci_lower = np.exp(
            conf.loc[feature, 0]
        )

        ci_upper = np.exp(
            conf.loc[feature, 1]
        )

        p_value = pvalues[feature]


        adjusted_results.append({

            "Feature": feature,

            "Coefficient":
                coefficient,

            "Adjusted_Odds_Ratio":
                odds_ratio,

            "CI_Lower_95":
                ci_lower,

            "CI_Upper_95":
                ci_upper,

            "p_value":
                p_value
        })


adjusted_results_df = pd.DataFrame(
    adjusted_results
)


# --------------------------------------------------------------
# 13. FDR CORRECTION FOR ADJUSTED MODEL
# --------------------------------------------------------------

if len(adjusted_results_df) > 0:

    rejected_adj, corrected_adj_p, _, _ = multipletests(
        adjusted_results_df["p_value"],
        method="fdr_bh"
    )


    adjusted_results_df[
        "FDR_Adjusted_p"
    ] = corrected_adj_p


    adjusted_results_df[
        "FDR_Significant"
    ] = rejected_adj


# --------------------------------------------------------------
# 14. SEPARATE THYROID-SPECIFIC RESULTS
# --------------------------------------------------------------

thyroid_adjusted_results = adjusted_results_df[
    adjusted_results_df["Feature"].isin(
        [
            "TSH_z",
            "Free_T3_z",
            "Free_T4_z",
            "TPO_z",
            "FT3_FT4_Ratio_z"
        ]
    )
].copy()


print("\n" + "=" * 75)
print("9. THYROID-SPECIFIC ADJUSTED ASSOCIATIONS")
print("=" * 75)


if len(thyroid_adjusted_results) > 0:

    print(
        thyroid_adjusted_results.to_string(
            index=False
        )
    )

else:

    print(
        "\nNo thyroid-specific adjusted results available."
    )


# --------------------------------------------------------------
# 15. CONVENTIONAL CVD FACTORS
# --------------------------------------------------------------

conventional_results = adjusted_results_df[
    ~adjusted_results_df["Feature"].isin(
        [
            "TSH_z",
            "Free_T3_z",
            "Free_T4_z",
            "TPO_z",
            "FT3_FT4_Ratio_z"
        ]
    )
].copy()


print("\n" + "=" * 75)
print("10. CONVENTIONAL CVD FACTOR RESULTS")
print("=" * 75)


if len(conventional_results) > 0:

    print(
        conventional_results.to_string(
            index=False
        )
    )


# --------------------------------------------------------------
# 16. MODEL FIT
# --------------------------------------------------------------

print("\n" + "=" * 75)
print("11. MODEL PERFORMANCE")
print("=" * 75)


if model_success:

    print(
        "\nMcFadden pseudo R-squared:",
        fitted_model.prsquared
    )

    print(
        "AIC:",
        fitted_model.aic
    )

    print(
        "BIC:",
        fitted_model.bic
    )

    print(
        "Log-likelihood:",
        fitted_model.llf
    )

    print(
        "LLR p-value:",
        fitted_model.llr_pvalue
    )


# --------------------------------------------------------------
# 17. SUMMARY OF SIGNIFICANT THYROID FINDINGS
# --------------------------------------------------------------

print("\n" + "=" * 75)
print("12. SIGNIFICANT THYROID ASSOCIATIONS")
print("=" * 75)


significant_unadjusted = biomarker_results_df[
    biomarker_results_df["FDR_Significant"] == True
]


if len(significant_unadjusted) > 0:

    print(
        "\nSignificant thyroid biomarkers after FDR:"
    )

    print(
        significant_unadjusted[
            [
                "Feature",
                "Mean_No_CVD",
                "Mean_Predicted_CVD",
                "p_value",
                "FDR_Adjusted_p"
            ]
        ].to_string(index=False)
    )

else:

    print(
        "\nNo thyroid biomarkers remained "
        "significant after FDR correction."
    )


# --------------------------------------------------------------
# 18. FINAL INTERPRETATION SUMMARY
# --------------------------------------------------------------

print("\n" + "=" * 75)
print("13. FINAL ANALYSIS SUMMARY")
print("=" * 75)


total_thyroid = len(thyroid_df)

predicted_cvd = (
    thyroid_df["CVD_Prediction"] == 1
).sum()


predicted_cvd_percentage = (
    predicted_cvd /
    total_thyroid
) * 100


print(
    "\nThyroid-problem participants:",
    total_thyroid
)


print(
    "Model-predicted CVD:",
    predicted_cvd
)


print(
    "Percentage with model-predicted CVD:",
    round(
        predicted_cvd_percentage,
        2
    ),
    "%"
)


print("\nResearch interpretation framework:")

print(
    "1. Thyroid-problem participants were identified first."
)

print(
    "2. Existing CVD model predictions were then evaluated "
    "within this thyroid population."
)

print(
    "3. Thyroid biomarkers were compared between "
    "predicted-CVD and non-CVD groups."
)

print(
    "4. Associations with continuous CVD probability "
    "were evaluated using Spearman correlation."
)

print(
    "5. Multivariable analysis adjusted thyroid biomarkers "
    "for conventional CVD factors."
)

print(
    "6. CVD_Probability was NOT used as a predictor."
)

print(
    "7. Original Cardiovascular_Disease was NOT modified."
)


# --------------------------------------------------------------
# 19. SAVE RESULTS
# --------------------------------------------------------------

print("\n" + "=" * 75)
print("14. SAVING RESULTS")
print("=" * 75)


biomarker_results_df.to_csv(
    "output/step7_thyroid_biomarker_cvd_association.csv",
    index=False
)


correlation_results_df.to_csv(
    "output/step7_thyroid_cvd_probability_correlation.csv",
    index=False
)


tsh_percent.to_csv(
    "output/step7_abnormal_tsh_cvd_percentage.csv"
)


adjusted_results_df.to_csv(
    "output/step7_adjusted_thyroid_cvd_model.csv",
    index=False
)


thyroid_adjusted_results.to_csv(
    "output/step7_thyroid_specific_adjusted_results.csv",
    index=False
)


conventional_results.to_csv(
    "output/step7_conventional_cvd_adjusted_results.csv",
    index=False
)


print(
    "\nSaved:"
)

print(
    "1. output/step7_thyroid_biomarker_cvd_association.csv"
)

print(
    "2. output/step7_thyroid_cvd_probability_correlation.csv"
)

print(
    "3. output/step7_abnormal_tsh_cvd_percentage.csv"
)

print(
    "4. output/step7_adjusted_thyroid_cvd_model.csv"
)

print(
    "5. output/step7_thyroid_specific_adjusted_results.csv"
)

print(
    "6. output/step7_conventional_cvd_adjusted_results.csv"
)


print("\n" + "=" * 75)
print("STEP 7 COMPLETE")
print("=" * 75)
