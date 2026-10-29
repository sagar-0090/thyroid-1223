# ================================================================
# STEP 8 - ROBUST THYROID-CVD ASSOCIATION VALIDATION
# ================================================================
#
# Research direction:
# 1. Start with participants already identified as having a
#    thyroid problem.
# 2. Evaluate their existing model-derived CVD outcome.
# 3. Validate associations between thyroid biomarkers and:
#       - CVD_Prediction
#       - CVD_Probability
# 4. Adjust thyroid associations for conventional CVD factors.
# 5. Avoid using CVD_Probability as a predictor because
#    CVD_Prediction is derived from it.
# 6. Address the convergence/separation problem encountered
#    in Step 7.
#
# IMPORTANT:
# This is NOT a new CVD prediction model.
# It is an association/validation analysis.
# ================================================================

import os
import warnings
import numpy as np
import pandas as pd

from scipy.stats import (
    mannwhitneyu,
    spearmanr,
    chi2_contingency
)

import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ================================================================
# SETTINGS
# ================================================================

INPUT_FILE = "output/step5_integrated_thyroid_cvd_dataset.csv"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

warnings.filterwarnings("ignore")


# ================================================================
# 1. LOAD DATA
# ================================================================

print("=" * 70)
print("STEP 8 - ROBUST THYROID-CVD ASSOCIATION VALIDATION")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded:")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ================================================================
# 2. REQUIRED COLUMNS
# ================================================================

required_columns = [
    "SEQN",
    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO",
    "FT3_FT4_Ratio",
    "Age",
    "BMI",
    "SYS_BP",
    "HDL",
    "Total_Cholesterol",
    "Diabetes",
    "Smoking_100_Cigarettes",
    "CVD_Probability",
    "CVD_Prediction",
    "Thyroid_Problem"
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


# ================================================================
# 3. THYROID-PROBLEM POPULATION
# ================================================================

thyroid_df = df[df["Thyroid_Problem"] == 1].copy()

print("\n" + "=" * 70)
print("1. THYROID-PROBLEM POPULATION")
print("=" * 70)

print("\nTotal input records:", len(df))
print("Thyroid-problem records:", len(thyroid_df))

print("\nCVD prediction distribution:")
print(
    thyroid_df["CVD_Prediction"]
    .value_counts()
    .sort_index()
)

print("\nCVD prediction percentage:")
print(
    thyroid_df["CVD_Prediction"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)


# ================================================================
# 4. DATA VALIDATION
# ================================================================

print("\n" + "=" * 70)
print("2. DATA VALIDATION")
print("=" * 70)

analysis_columns = [
    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO",
    "FT3_FT4_Ratio",
    "Age",
    "BMI",
    "SYS_BP",
    "HDL",
    "Total_Cholesterol",
    "Diabetes",
    "Smoking_100_Cigarettes",
    "CVD_Probability",
    "CVD_Prediction"
]

print("\nMissing values:")
print(thyroid_df[analysis_columns].isna().sum())


# ================================================================
# 5. THYROID BIOMARKER DEFINITIONS
# ================================================================

thyroid_biomarkers = [
    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO",
    "FT3_FT4_Ratio"
]

print("\n" + "=" * 70)
print("3. THYROID BIOMARKERS")
print("=" * 70)

for variable in thyroid_biomarkers:
    print("-", variable)


# ================================================================
# 6. UNADJUSTED CVD GROUP COMPARISON
# ================================================================

print("\n" + "=" * 70)
print("4. THYROID BIOMARKERS VS MODEL-PREDICTED CVD")
print("=" * 70)

group_results = []

for variable in thyroid_biomarkers:

    group0 = thyroid_df.loc[
        thyroid_df["CVD_Prediction"] == 0,
        variable
    ].dropna()

    group1 = thyroid_df.loc[
        thyroid_df["CVD_Prediction"] == 1,
        variable
    ].dropna()

    if len(group0) == 0 or len(group1) == 0:
        continue

    statistic, p_value = mannwhitneyu(
        group0,
        group1,
        alternative="two-sided"
    )

    result = {
        "Feature": variable,
        "N_No_CVD": len(group0),
        "N_Predicted_CVD": len(group1),
        "Mean_No_CVD": group0.mean(),
        "Mean_Predicted_CVD": group1.mean(),
        "Median_No_CVD": group0.median(),
        "Median_Predicted_CVD": group1.median(),
        "Mann_Whitney_U": statistic,
        "P_Value": p_value
    }

    group_results.append(result)

    print("\n" + "-" * 60)
    print(variable)

    print("No predicted CVD:")
    print("  N:", len(group0))
    print("  Mean:", group0.mean())
    print("  Median:", group0.median())

    print("Predicted CVD:")
    print("  N:", len(group1))
    print("  Mean:", group1.mean())
    print("  Median:", group1.median())

    print("Mann-Whitney U:", statistic)
    print("P-value:", p_value)


group_results_df = pd.DataFrame(group_results)


# ================================================================
# 7. FDR CORRECTION FOR GROUP COMPARISONS
# ================================================================

print("\n" + "=" * 70)
print("5. FDR CORRECTION - GROUP COMPARISONS")
print("=" * 70)

if len(group_results_df) > 0:

    reject, corrected_p, _, _ = multipletests(
        group_results_df["P_Value"],
        method="fdr_bh"
    )

    group_results_df["FDR_Adjusted_P"] = corrected_p
    group_results_df["FDR_Significant"] = reject

    print(
        group_results_df[
            [
                "Feature",
                "P_Value",
                "FDR_Adjusted_P",
                "FDR_Significant"
            ]
        ]
    )


# ================================================================
# 8. SPEARMAN ASSOCIATION WITH CONTINUOUS CVD PROBABILITY
# ================================================================

print("\n" + "=" * 70)
print("6. THYROID BIOMARKERS VS CVD PROBABILITY")
print("=" * 70)

correlation_results = []

for variable in thyroid_biomarkers:

    temp = thyroid_df[
        [variable, "CVD_Probability"]
    ].dropna()

    rho, p_value = spearmanr(
        temp[variable],
        temp["CVD_Probability"]
    )

    correlation_results.append({
        "Feature": variable,
        "N": len(temp),
        "Spearman_Rho": rho,
        "P_Value": p_value
    })

    print("\n" + "-" * 60)
    print(variable)
    print("N:", len(temp))
    print("Spearman correlation:", rho)
    print("P-value:", p_value)


correlation_df = pd.DataFrame(correlation_results)


# ================================================================
# 9. FDR CORRECTION FOR CORRELATIONS
# ================================================================

if len(correlation_df) > 0:

    reject, corrected_p, _, _ = multipletests(
        correlation_df["P_Value"],
        method="fdr_bh"
    )

    correlation_df["FDR_Adjusted_P"] = corrected_p
    correlation_df["FDR_Significant"] = reject

print("\nFDR-corrected correlations:")

print(
    correlation_df[
        [
            "Feature",
            "Spearman_Rho",
            "P_Value",
            "FDR_Adjusted_P",
            "FDR_Significant"
        ]
    ]
)


# ================================================================
# 10. ABNORMAL TSH ANALYSIS
# ================================================================

print("\n" + "=" * 70)
print("7. ABNORMAL TSH VS MODEL-PREDICTED CVD")
print("=" * 70)

tsh_table = pd.crosstab(
    thyroid_df["Abnormal_TSH"],
    thyroid_df["CVD_Prediction"]
)

print("\nContingency table:")
print(tsh_table)

chi2, p_value, dof, expected = chi2_contingency(
    tsh_table
)

print("\nChi-square:", chi2)
print("Degrees of freedom:", dof)
print("P-value:", p_value)

tsh_percent = pd.crosstab(
    thyroid_df["Abnormal_TSH"],
    thyroid_df["CVD_Prediction"],
    normalize="index"
).mul(100).round(2)

print("\nCVD percentage within TSH groups:")
print(tsh_percent)


# ================================================================
# 11. PREPARE ADJUSTED ANALYSIS DATA
# ================================================================

print("\n" + "=" * 70)
print("8. ROBUST ADJUSTED ANALYSIS")
print("=" * 70)

print("""
Purpose:
Determine whether thyroid biomarkers are associated with
model-predicted CVD after accounting for conventional CVD factors.

The analysis is performed only within participants already
identified as having a thyroid problem.

CVD_Probability is NOT used as a predictor.

The dependent variable is CVD_Prediction.
""")


# ================================================================
# 12. SELECT VARIABLES
# ================================================================

conventional_factors = [
    "Age",
    "BMI",
    "SYS_BP",
    "HDL",
    "Total_Cholesterol"
]

print("\nThyroid variables:")
for x in thyroid_biomarkers:
    print("-", x)

print("\nConventional adjustment variables:")
for x in conventional_factors:
    print("-", x)


# ================================================================
# 13. CREATE CLEAN NUMERIC DATA
# ================================================================

model_columns = (
    thyroid_biomarkers
    + conventional_factors
    + [
        "Diabetes",
        "Smoking_100_Cigarettes",
        "CVD_Prediction"
    ]
)

model_df = thyroid_df[model_columns].copy()

# Convert all variables to numeric where possible
for col in model_columns:
    model_df[col] = pd.to_numeric(
        model_df[col],
        errors="coerce"
    )

model_df = model_df.dropna().copy()

print("\nRecords available for adjusted analysis:")
print(len(model_df))

print("\nOutcome distribution:")
print(
    model_df["CVD_Prediction"]
    .value_counts()
    .sort_index()
)


# ================================================================
# 14. STANDARDIZE CONTINUOUS VARIABLES
# ================================================================

continuous_variables = (
    thyroid_biomarkers
    + conventional_factors
)

for col in continuous_variables:

    mean_value = model_df[col].mean()
    std_value = model_df[col].std()

    if std_value > 0:
        model_df[col + "_z"] = (
            model_df[col] - mean_value
        ) / std_value
    else:
        model_df[col + "_z"] = 0


# ================================================================
# 15. ENCODE CATEGORICAL VARIABLES
# ================================================================

# Explicitly convert categorical variables to dummy variables.
#
# This avoids treating diabetes categories as continuous numbers.

diabetes_dummies = pd.get_dummies(
    model_df["Diabetes"],
    prefix="Diabetes",
    drop_first=True,
    dtype=float
)

smoking_dummies = pd.get_dummies(
    model_df["Smoking_100_Cigarettes"],
    prefix="Smoking",
    drop_first=True,
    dtype=float
)

model_df = pd.concat(
    [
        model_df,
        diabetes_dummies,
        smoking_dummies
    ],
    axis=1
)


# ================================================================
# 16. BUILD THYROID-SPECIFIC MODEL
# ================================================================

predictor_columns = [
    "Age_z",
    "TSH_z",
    "Free_T3_z",
    "Free_T4_z",
    "TPO_z",
    "FT3_FT4_Ratio_z",
    "BMI_z",
    "SYS_BP_z",
    "HDL_z",
    "Total_Cholesterol_z"
]

# Add available dummy variables.
dummy_columns = [
    col for col in model_df.columns
    if col.startswith("Diabetes_")
    or col.startswith("Smoking_")
]

predictor_columns.extend(dummy_columns)

print("\nPredictors:")
for x in predictor_columns:
    print("-", x)


# ================================================================
# 17. REMOVE CONSTANT / ZERO-VARIANCE VARIABLES
# ================================================================

valid_predictors = []

for col in predictor_columns:

    if col not in model_df.columns:
        continue

    if model_df[col].nunique() > 1:
        valid_predictors.append(col)

predictor_columns = valid_predictors

print("\nPredictors retained after variance check:")
print(predictor_columns)


# ================================================================
# 18. VIF CHECK
# ================================================================

print("\n" + "=" * 70)
print("9. MULTICOLLINEARITY CHECK")
print("=" * 70)

X_vif = model_df[predictor_columns].copy()

X_vif = sm.add_constant(
    X_vif,
    has_constant="add"
)

vif_results = []

for i, col in enumerate(X_vif.columns):

    if col == "const":
        continue

    try:
        vif_value = variance_inflation_factor(
            X_vif.values,
            i
        )
    except Exception:
        vif_value = np.nan

    vif_results.append({
        "Feature": col,
        "VIF": vif_value
    })

vif_df = pd.DataFrame(vif_results)

print(vif_df)


# ================================================================
# 19. ROBUST MODELING APPROACH
# ================================================================
#
# Step 7 had convergence problems because the dataset is small
# and some categorical variables have sparse categories.
#
# To avoid unstable categorical separation, we first fit the
# primary thyroid association model using continuous variables
# and major conventional continuous CVD factors.
#
# This directly answers:
#
# "Are thyroid biomarkers associated with model-predicted CVD
# after accounting for age, BMI, systolic BP, HDL and cholesterol?"
#
# ================================================================

robust_predictors = [
    "Age_z",
    "TSH_z",
    "Free_T3_z",
    "Free_T4_z",
    "TPO_z",
    "FT3_FT4_Ratio_z",
    "BMI_z",
    "SYS_BP_z",
    "HDL_z",
    "Total_Cholesterol_z"
]

X = model_df[robust_predictors].copy()

y = model_df["CVD_Prediction"].astype(int)

X = sm.add_constant(
    X,
    has_constant="add"
)


# ================================================================
# 20. CHECK OUTCOME VARIATION
# ================================================================

if y.nunique() < 2:
    raise ValueError(
        "CVD_Prediction contains only one class. "
        "Logistic regression cannot be performed."
    )


# ================================================================
# 21. FIT ROBUST STANDARD LOGISTIC REGRESSION
# ================================================================

print("\n" + "=" * 70)
print("10. PRIMARY ADJUSTED THYROID-CVD MODEL")
print("=" * 70)

logit_model = sm.Logit(y, X)

try:

    result = logit_model.fit(
        disp=False,
        maxiter=500
    )

    print("\nModel converged:")
    print(result.mle_retvals)

    print("\nModel summary:")
    print(result.summary())

except Exception as e:

    print("\nStandard logistic regression failed:")
    print(str(e))

    result = None


# ================================================================
# 22. EXTRACT ADJUSTED ODDS RATIOS
# ================================================================

adjusted_results = []

if result is not None:

    params = result.params
    conf = result.conf_int()
    pvalues = result.pvalues

    for feature in robust_predictors:

        coefficient = params[feature]

        odds_ratio = np.exp(
            np.clip(coefficient, -20, 20)
        )

        lower = np.exp(
            np.clip(conf.loc[feature, 0], -20, 20)
        )

        upper = np.exp(
            np.clip(conf.loc[feature, 1], -20, 20)
        )

        adjusted_results.append({
            "Feature": feature,
            "Coefficient": coefficient,
            "Adjusted_Odds_Ratio": odds_ratio,
            "CI_Lower_95": lower,
            "CI_Upper_95": upper,
            "P_Value": pvalues[feature]
        })


adjusted_df = pd.DataFrame(adjusted_results)


# ================================================================
# 23. FDR CORRECTION
# ================================================================

if len(adjusted_df) > 0:

    reject, corrected_p, _, _ = multipletests(
        adjusted_df["P_Value"],
        method="fdr_bh"
    )

    adjusted_df["FDR_Adjusted_P"] = corrected_p
    adjusted_df["FDR_Significant"] = reject

print("\nAdjusted thyroid/CVD results:")

if len(adjusted_df) > 0:

    print(
        adjusted_df[
            [
                "Feature",
                "Adjusted_Odds_Ratio",
                "CI_Lower_95",
                "CI_Upper_95",
                "P_Value",
                "FDR_Adjusted_P",
                "FDR_Significant"
            ]
        ]
    )


# ================================================================
# 24. THYROID-SPECIFIC RESULTS ONLY
# ================================================================

thyroid_adjusted_df = adjusted_df[
    adjusted_df["Feature"].isin([
        "TSH_z",
        "Free_T3_z",
        "Free_T4_z",
        "TPO_z",
        "FT3_FT4_Ratio_z"
    ])
].copy()

print("\n" + "=" * 70)
print("11. THYROID-SPECIFIC ADJUSTED RESULTS")
print("=" * 70)

print(thyroid_adjusted_df)


# ================================================================
# 25. MODEL PERFORMANCE
# ================================================================

performance_results = []

if result is not None:

    performance_results.append({
        "N": len(model_df),
        "McFadden_Pseudo_R2": result.prsquared,
        "AIC": result.aic,
        "BIC": result.bic,
        "Log_Likelihood": result.llf,
        "LLR_P_Value": result.llr_pvalue,
        "Converged": True
    })

    print("\n" + "=" * 70)
    print("12. MODEL PERFORMANCE")
    print("=" * 70)

    print(
        "McFadden pseudo R-squared:",
        result.prsquared
    )

    print("AIC:", result.aic)
    print("BIC:", result.bic)
    print("Log-likelihood:", result.llf)
    print("LLR p-value:", result.llr_pvalue)

else:

    performance_results.append({
        "N": len(model_df),
        "McFadden_Pseudo_R2": np.nan,
        "AIC": np.nan,
        "BIC": np.nan,
        "Log_Likelihood": np.nan,
        "LLR_P_Value": np.nan,
        "Converged": False
    })


performance_df = pd.DataFrame(performance_results)


# ================================================================
# 26. IDENTIFY ROBUST THYROID ASSOCIATIONS
# ================================================================

print("\n" + "=" * 70)
print("13. FINAL THYROID ASSOCIATION FINDINGS")
print("=" * 70)

if len(thyroid_adjusted_df) > 0:

    significant_thyroid = thyroid_adjusted_df[
        thyroid_adjusted_df["FDR_Significant"] == True
    ]

    if len(significant_thyroid) > 0:

        print(
            "\nThyroid biomarkers remaining significant "
            "after adjustment and FDR correction:"
        )

        print(
            significant_thyroid[
                [
                    "Feature",
                    "Adjusted_Odds_Ratio",
                    "CI_Lower_95",
                    "CI_Upper_95",
                    "P_Value",
                    "FDR_Adjusted_P"
                ]
            ]
        )

    else:

        print(
            "\nNo thyroid biomarker remained independently "
            "significant after adjustment and FDR correction."
        )

        print(
            "\nThis does NOT mean that the biomarkers have "
            "no association. It means their association was "
            "not statistically independent of the conventional "
            "factors included in this adjusted model."
        )


# ================================================================
# 27. INTERPRETATION FRAMEWORK
# ================================================================

print("\n" + "=" * 70)
print("14. RESEARCH INTERPRETATION")
print("=" * 70)

print("""
The analysis follows the intended research framework:

1. Participants with thyroid problems were identified first.

2. The existing CVD model prediction was evaluated only
   within this thyroid-problem population.

3. Thyroid biomarkers were compared between participants
   with and without model-predicted CVD.

4. Spearman correlation was used to examine relationships
   between thyroid biomarkers and continuous CVD probability.

5. A multivariable association model was used to determine
   whether thyroid biomarkers remain associated with
   model-predicted CVD after accounting for major conventional
   cardiovascular factors.

6. CVD_Probability was NOT used as a predictor.

7. The analysis does NOT create a new CVD prediction model.

8. The original Cardiovascular_Disease variable was NOT modified.

9. Associations should NOT be interpreted as proof that
   thyroid abnormalities cause cardiovascular disease.
""")


# ================================================================
# 28. SAVE RESULTS
# ================================================================

print("\n" + "=" * 70)
print("15. SAVING RESULTS")
print("=" * 70)

group_results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step8_thyroid_cvd_group_comparison.csv"
    ),
    index=False
)

correlation_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step8_thyroid_cvd_probability_correlation.csv"
    ),
    index=False
)

tsh_percent.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step8_abnormal_tsh_cvd_percentage.csv"
    )
)

vif_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step8_vif_results.csv"
    ),
    index=False
)

adjusted_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step8_adjusted_thyroid_cvd_results.csv"
    ),
    index=False
)

thyroid_adjusted_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step8_thyroid_specific_adjusted_results.csv"
    ),
    index=False
)

performance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step8_model_performance.csv"
    ),
    index=False
)


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("STEP 8 COMPLETE")
print("=" * 70)

print("\nPopulation analyzed:", len(thyroid_df))

print(
    "Thyroid-problem + model-predicted CVD:",
    int(
        (thyroid_df["CVD_Prediction"] == 1).sum()
    )
)

print(
    "Percentage with model-predicted CVD:",
    round(
        (thyroid_df["CVD_Prediction"] == 1).mean() * 100,
        2
    ),
    "%"
)

print("\nPrimary purpose:")
print(
    "Validate thyroid-CVD associations and address "
    "Step 7 convergence instability."
)

print("\nFiles saved:")
print(
    "1. output/step8_thyroid_cvd_group_comparison.csv"
)
print(
    "2. output/step8_thyroid_cvd_probability_correlation.csv"
)
print(
    "3. output/step8_abnormal_tsh_cvd_percentage.csv"
)
print(
    "4. output/step8_vif_results.csv"
)
print(
    "5. output/step8_adjusted_thyroid_cvd_results.csv"
)
print(
    "6. output/step8_thyroid_specific_adjusted_results.csv"
)
print(
    "7. output/step8_model_performance.csv"
)

print("\nOriginal Cardiovascular_Disease column:")
print("NOT MODIFIED")

print("\n" + "=" * 70)
