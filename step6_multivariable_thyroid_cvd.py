import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests

# ============================================================================
# STEP 6 - MULTIVARIABLE THYROID + CVD ANALYSIS
# ============================================================================

print("=" * 75)
print("STEP 6 - MULTIVARIABLE THYROID + CVD ANALYSIS")
print("=" * 75)


# ============================================================================
# 1. LOAD STEP-5 DATA
# ============================================================================

input_file = "output/step5_integrated_thyroid_cvd_dataset.csv"

df = pd.read_csv(input_file)

print("\nDataset loaded:")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================================

required_columns = [
    "Age",
    "Gender",
    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO",
    "BMI",
    "SYS_BP",
    "HDL",
    "Total_Cholesterol",
    "Diabetes",
    "Smoking_100_Cigarettes",
    "CVD_Prediction"
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
# 3. DATA VALIDATION
# ============================================================================

print("\n" + "=" * 75)
print("1. DATA VALIDATION")
print("=" * 75)

print("\nTotal records:", len(df))

print("\nMissing values:")
print(df[required_columns].isnull().sum())


# ============================================================================
# 4. KEEP THYROID-PROBLEM PARTICIPANTS
# ============================================================================

print("\n" + "=" * 75)
print("2. THYROID-PROBLEM POPULATION")
print("=" * 75)

# Step 5 dataset should already contain thyroid-problem participants.
# This additional filter protects the analysis if the input contains others.

if "Thyroid_Problem" in df.columns:

    thyroid_df = df[
        df["Thyroid_Problem"] == 1
    ].copy()

else:

    thyroid_df = df[
        df["Thyroid_Diagnosis"] == 1
    ].copy()


print("\nThyroid-problem records:", len(thyroid_df))

print("\nCVD prediction distribution:")
print(
    thyroid_df["CVD_Prediction"].value_counts()
)


# ============================================================================
# 5. SELECT VARIABLES
# ============================================================================

print("\n" + "=" * 75)
print("3. VARIABLE SELECTION")
print("=" * 75)

# --------------------------------------------------------------------------
# Outcome
# --------------------------------------------------------------------------

outcome = "CVD_Prediction"


# --------------------------------------------------------------------------
# Thyroid biomarkers
# --------------------------------------------------------------------------

thyroid_predictors = [
    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO"
]


# --------------------------------------------------------------------------
# Conventional cardiovascular risk factors
#
# We intentionally DO NOT include:
#   Risk_Factor_Count
#
# because it is constructed from several individual risk factors.
# --------------------------------------------------------------------------

cardiovascular_predictors = [
    "Age",
    "BMI",
    "SYS_BP",
    "HDL",
    "Total_Cholesterol",
    "Diabetes",
    "Smoking_100_Cigarettes"
]


predictors = (
    thyroid_predictors +
    cardiovascular_predictors
)

print("\nThyroid predictors:")
for x in thyroid_predictors:
    print("-", x)

print("\nCardiovascular predictors:")
for x in cardiovascular_predictors:
    print("-", x)

print("\nOutcome:")
print("-", outcome)


# ============================================================================
# 6. PREPARE ANALYSIS DATA
# ============================================================================

print("\n" + "=" * 75)
print("4. PREPARING ANALYSIS DATA")
print("=" * 75)

analysis_df = thyroid_df[
    predictors + [outcome]
].copy()


# Convert everything to numeric
for col in predictors + [outcome]:

    analysis_df[col] = pd.to_numeric(
        analysis_df[col],
        errors="coerce"
    )


# Remove missing observations
analysis_df = analysis_df.dropna()


print("\nRecords available for regression:", len(analysis_df))

print("\nOutcome distribution:")
print(
    analysis_df[outcome].value_counts()
)


# Check whether both outcome classes exist
if analysis_df[outcome].nunique() < 2:

    raise ValueError(
        "CVD_Prediction contains fewer than two outcome classes."
    )


# ============================================================================
# 7. STANDARDIZE CONTINUOUS PREDICTORS
# ============================================================================

print("\n" + "=" * 75)
print("5. STANDARDIZING CONTINUOUS VARIABLES")
print("=" * 75)

# Standardization makes coefficients more comparable.
# Odds ratios represent the change in odds associated with
# approximately one standard deviation increase.

continuous_predictors = [
    "Age",
    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO",
    "BMI",
    "SYS_BP",
    "HDL",
    "Total_Cholesterol"
]

binary_predictors = [
    "Diabetes",
    "Smoking_100_Cigarettes"
]

model_df = analysis_df.copy()

standardization_info = []

for col in continuous_predictors:

    mean_value = model_df[col].mean()
    std_value = model_df[col].std()

    if std_value == 0:

        print(
            f"WARNING: {col} has zero variance and will be excluded."
        )

        continue

    model_df[col] = (
        model_df[col] - mean_value
    ) / std_value

    standardization_info.append({
        "Variable": col,
        "Original_Mean": mean_value,
        "Original_SD": std_value
    })


print("\nContinuous variables standardized:")
for col in continuous_predictors:
    print("-", col)


# ============================================================================
# 8. CREATE MODEL MATRIX
# ============================================================================

print("\n" + "=" * 75)
print("6. BUILDING MULTIVARIABLE LOGISTIC REGRESSION")
print("=" * 75)

available_predictors = (
    continuous_predictors +
    binary_predictors
)

X = model_df[available_predictors].copy()

y = model_df[outcome].astype(int)


# Add intercept
X = sm.add_constant(X)


print("\nNumber of predictors:", len(available_predictors))
print("Number of observations:", len(X))


# ============================================================================
# 9. CHECK MULTICOLLINEARITY
# ============================================================================

print("\n" + "=" * 75)
print("7. MULTICOLLINEARITY CHECK - VIF")
print("=" * 75)

vif_data = []

X_vif = X.drop(columns=["const"])

for i, column in enumerate(X_vif.columns):

    try:

        vif_value = variance_inflation_factor(
            X_vif.values,
            i
        )

    except Exception:

        vif_value = np.nan

    vif_data.append({
        "Feature": column,
        "VIF": vif_value
    })


vif_df = pd.DataFrame(vif_data)

print("\nVIF results:")
print(vif_df.to_string(index=False))


# ============================================================================
# 10. FIT LOGISTIC REGRESSION
# ============================================================================

print("\n" + "=" * 75)
print("8. MULTIVARIABLE LOGISTIC REGRESSION")
print("=" * 75)

try:

    logistic_model = sm.Logit(
        y,
        X
    )

    model_result = logistic_model.fit(
        disp=False,
        maxiter=200
    )

except Exception as e:

    print("\nStandard logistic regression failed:")
    print(e)

    print("\nTrying regularized logistic regression...")

    model_result = None


# ============================================================================
# 11. MODEL RESULTS
# ============================================================================

if model_result is not None:

    print("\n" + "=" * 75)
    print("9. MODEL SUMMARY")
    print("=" * 75)

    print(
        model_result.summary()
    )


    # ------------------------------------------------------------------------
    # Odds ratios
    # ------------------------------------------------------------------------

    print("\n" + "=" * 75)
    print("10. ADJUSTED ODDS RATIOS")
    print("=" * 75)

    coefficients = model_result.params
    p_values = model_result.pvalues
    confidence_intervals = model_result.conf_int()

    results = []

    for feature in coefficients.index:

        if feature == "const":
            continue

        beta = coefficients[feature]

        lower_beta = confidence_intervals.loc[
            feature, 0
        ]

        upper_beta = confidence_intervals.loc[
            feature, 1
        ]

        odds_ratio = np.exp(beta)

        ci_lower = np.exp(lower_beta)
        ci_upper = np.exp(upper_beta)

        p_value = p_values[feature]

        results.append({

            "Feature": feature,

            "Coefficient": beta,

            "Adjusted_Odds_Ratio": odds_ratio,

            "CI_Lower_95": ci_lower,

            "CI_Upper_95": ci_upper,

            "p_value": p_value
        })


    regression_results = pd.DataFrame(results)


    # ------------------------------------------------------------------------
    # FDR correction
    # ------------------------------------------------------------------------

    if len(regression_results) > 0:

        corrected = multipletests(
            regression_results["p_value"],
            method="fdr_bh"
        )

        regression_results["FDR_Adjusted_p"] = corrected[1]

        regression_results["FDR_Significant"] = (
            regression_results["FDR_Adjusted_p"] < 0.05
        )


    # ------------------------------------------------------------------------
    # Interpretation category
    # ------------------------------------------------------------------------

    interpretations = []

    for _, row in regression_results.iterrows():

        or_value = row["Adjusted_Odds_Ratio"]
        fdr_p = row["FDR_Adjusted_p"]

        if fdr_p < 0.05:

            if or_value > 1:

                interpretation = (
                    "Independent positive association"
                )

            else:

                interpretation = (
                    "Independent negative association"
                )

        else:

            interpretation = (
                "Not independently significant"
            )

        interpretations.append(
            interpretation
        )


    regression_results["Interpretation"] = interpretations


    print(
        "\nAdjusted odds-ratio results:"
    )

    print(
        regression_results.to_string(
            index=False
        )
    )


    # =========================================================================
    # 12. THYROID-SPECIFIC RESULTS
    # =========================================================================

    print("\n" + "=" * 75)
    print("11. THYROID-SPECIFIC ADJUSTED RESULTS")
    print("=" * 75)

    thyroid_results = regression_results[
        regression_results["Feature"].isin(
            thyroid_predictors
        )
    ].copy()

    print(
        thyroid_results.to_string(
            index=False
        )
    )


    # =========================================================================
    # 13. SIGNIFICANT INDEPENDENT PREDICTORS
    # =========================================================================

    print("\n" + "=" * 75)
    print("12. INDEPENDENTLY SIGNIFICANT PREDICTORS")
    print("=" * 75)

    significant_results = regression_results[
        regression_results["FDR_Adjusted_p"] < 0.05
    ].copy()

    if len(significant_results) > 0:

        print(
            significant_results.to_string(
                index=False
            )
        )

    else:

        print(
            "\nNo predictor remained statistically significant "
            "after FDR correction."
        )


    # =========================================================================
    # 14. MODEL PERFORMANCE
    # =========================================================================

    print("\n" + "=" * 75)
    print("13. MODEL PERFORMANCE")
    print("=" * 75)

    print(
        "\nMcFadden pseudo R-squared:",
        model_result.prsquared
    )

    print(
        "\nAIC:",
        model_result.aic
    )

    print(
        "BIC:",
        model_result.bic
    )


    # =========================================================================
    # 15. PREDICTED PROBABILITY
    # =========================================================================

    print("\n" + "=" * 75)
    print("14. PREDICTED PROBABILITY")
    print("=" * 75)

    model_df["Adjusted_CVD_Probability"] = (
        model_result.predict(X)
    )

    print(
        "\nMean adjusted CVD probability:",
        model_df[
            "Adjusted_CVD_Probability"
        ].mean()
    )

    print(
        "Median adjusted CVD probability:",
        model_df[
            "Adjusted_CVD_Probability"
        ].median()
    )


    # =========================================================================
    # 16. SAVE RESULTS
    # =========================================================================

    print("\n" + "=" * 75)
    print("15. SAVING RESULTS")
    print("=" * 75)

    regression_results.to_csv(
        "output/step6_multivariable_logistic_regression.csv",
        index=False
    )

    thyroid_results.to_csv(
        "output/step6_thyroid_adjusted_results.csv",
        index=False
    )

    vif_df.to_csv(
        "output/step6_vif_results.csv",
        index=False
    )

    pd.DataFrame(
        standardization_info
    ).to_csv(
        "output/step6_standardization_info.csv",
        index=False
    )

    model_df.to_csv(
        "output/step6_adjusted_predictions.csv",
        index=False
    )

    print(
        "\nSaved:"
    )

    print(
        "1. output/step6_multivariable_logistic_regression.csv"
    )

    print(
        "2. output/step6_thyroid_adjusted_results.csv"
    )

    print(
        "3. output/step6_vif_results.csv"
    )

    print(
        "4. output/step6_standardization_info.csv"
    )

    print(
        "5. output/step6_adjusted_predictions.csv"
    )


    # =========================================================================
    # 17. FINAL SUMMARY
    # =========================================================================

    print("\n" + "=" * 75)
    print("STEP 6 COMPLETE")
    print("=" * 75)

    print(
        "\nPopulation analyzed:",
        len(model_df)
    )

    print(
        "Outcome:",
        "CVD_Prediction"
    )

    print(
        "\nThyroid biomarkers adjusted for:"
    )

    for feature in thyroid_predictors:
        print("-", feature)

    print(
        "\nConventional CVD factors adjusted for:"
    )

    for feature in cardiovascular_predictors:
        print("-", feature)

    print(
        "\nIMPORTANT:"
    )

    print(
        "CVD_Probability was NOT used as a predictor "
        "because CVD_Prediction is derived from it."
    )

    print(
        "Risk_Factor_Count was NOT included because it "
        "summarizes several individual risk factors."
    )

    print(
        "\nOriginal Cardiovascular_Disease column:"
    )

    print(
        "NOT MODIFIED"
    )

    print("\n" + "=" * 75)


else:

    # =========================================================================
    # FALLBACK INFORMATION
    # =========================================================================

    print("\n" + "=" * 75)
    print("LOGISTIC REGRESSION COULD NOT BE FIT")
    print("=" * 75)

    print(
        "\nThe dataset may have separation or "
        "multicollinearity problems."
    )

    print(
        "\nVIF results should be examined first:"
    )

    print(
        vif_df.to_string(
            index=False
        )
    )

    print("\n" + "=" * 75)
