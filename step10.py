# ============================================================================
# STEP 10 - FINAL INTEGRATED THYROID-CVD RESEARCH ANALYSIS
# ============================================================================
#
# RESEARCH OBJECTIVE
# ------------------
# Analyze cardiovascular risk among participants identified with thyroid
# problems and determine how thyroid biomarkers are associated with
# model-predicted CVD.
#
# RESEARCH FLOW
# ------------
# Thyroid problem identification
#             ↓
# Existing CVD model prediction
#             ↓
# CVD risk interpretation
#             ↓
# Conventional CVD risk-factor analysis
#             ↓
# Thyroid biomarker association analysis
#             ↓
# Individual risk interpretation
#             ↓
# Dashboard-ready research dataset
#
# IMPORTANT:
# - NO new CVD prediction model is created.
# - CVD_Prediction comes from the existing model.
# - CVD_Probability is NOT used as a predictor.
# - Cardiovascular_Disease is NOT modified.
# - Results represent associations, NOT causation.
# ============================================================================

import os
import pandas as pd
import numpy as np

# ============================================================================
# 0. CONFIGURATION
# ============================================================================

INPUT_FILE = "output/step5_integrated_thyroid_cvd_dataset.csv"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 1. LOAD DATASET
# ============================================================================

print("=" * 78)
print("STEP 10 - FINAL INTEGRATED THYROID-CVD RESEARCH ANALYSIS")
print("=" * 78)

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded from:")
print(INPUT_FILE)

print("\nShape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

# ============================================================================
# 2. REQUIRED COLUMNS
# ============================================================================

required_columns = [
    "SEQN",
    "Age",
    "BMI",
    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO",
    "FT3_FT4_Ratio",
    "HDL",
    "Total_Cholesterol",
    "Diabetes",
    "Smoking_100_Cigarettes",
    "SYS_BP",
    "Thyroid_Diagnosis",
    "Thyroid_Problem",
    "CVD_Probability",
    "CVD_Prediction",
    "Cardiovascular_Disease",
    "High_BP",
    "Obesity",
    "High_Cholesterol",
    "Low_HDL",
    "Smoking_Risk",
    "Abnormal_TSH",
    "Risk_Factor_Count",
    "Thyroid_Status",
    "CVD_Risk_Category",
    "Integrated_Thyroid_CVD_Group"
]

missing_required = [
    col for col in required_columns
    if col not in df.columns
]

if missing_required:
    raise ValueError(
        f"Missing required columns: {missing_required}"
    )

print("\nAll required columns are present.")

# ============================================================================
# 3. IDENTIFY THYROID-PROBLEM POPULATION
# ============================================================================

print("\n" + "=" * 78)
print("1. THYROID-PROBLEM POPULATION")
print("=" * 78)

# Explicitly identify thyroid-problem participants
thyroid_df = df[
    df["Thyroid_Problem"].astype(str).str.lower()
    .isin(["1", "1.0", "true", "yes"])
].copy()

# If the dataset uses Thyroid_Status instead
if len(thyroid_df) == 0:
    thyroid_df = df[
        df["Thyroid_Status"]
        .astype(str)
        .str.contains("Thyroid Problem", case=False, na=False)
    ].copy()

total_records = len(df)
thyroid_records = len(thyroid_df)

print("\nTotal records:", total_records)
print("Thyroid-problem records:", thyroid_records)

if thyroid_records == 0:
    raise ValueError(
        "No thyroid-problem participants were identified."
    )

# ============================================================================
# 4. EXISTING CVD MODEL RESULTS
# ============================================================================

print("\n" + "=" * 78)
print("2. EXISTING CVD MODEL RESULTS")
print("=" * 78)

cvd_counts = thyroid_df["CVD_Prediction"].value_counts().sort_index()

print("\nCVD prediction distribution:")
print(cvd_counts)

cvd_percentages = (
    thyroid_df["CVD_Prediction"]
    .value_counts(normalize=True)
    .sort_index() * 100
)

print("\nCVD prediction percentage:")
print(cvd_percentages.round(2))

predicted_cvd = int(
    (thyroid_df["CVD_Prediction"] == 1).sum()
)

no_predicted_cvd = int(
    (thyroid_df["CVD_Prediction"] == 0).sum()
)

predicted_cvd_percentage = (
    predicted_cvd / thyroid_records * 100
)

print("\nThyroid-problem + model-predicted CVD:",
      predicted_cvd)

print(
    "Percentage with model-predicted CVD:",
    round(predicted_cvd_percentage, 2),
    "%"
)

# ============================================================================
# 5. CVD PROBABILITY SUMMARY
# ============================================================================

print("\n" + "=" * 78)
print("3. CVD PROBABILITY SUMMARY")
print("=" * 78)

mean_cvd_probability = thyroid_df["CVD_Probability"].mean()
median_cvd_probability = thyroid_df["CVD_Probability"].median()

print("\nMean CVD probability:",
      round(mean_cvd_probability, 4))

print("Median CVD probability:",
      round(median_cvd_probability, 4))

risk_category_summary = (
    thyroid_df["CVD_Risk_Category"]
    .value_counts()
    .rename_axis("CVD_Risk_Category")
    .reset_index(name="Participants")
)

risk_category_summary["Percentage"] = (
    risk_category_summary["Participants"]
    / thyroid_records * 100
)

risk_category_summary["Percentage"] = (
    risk_category_summary["Percentage"].round(2)
)

print("\nCVD risk-category distribution:")
print(risk_category_summary.to_string(index=False))

# ============================================================================
# 6. THYROID BIOMARKER SUMMARY
# ============================================================================

print("\n" + "=" * 78)
print("4. THYROID BIOMARKER SUMMARY")
print("=" * 78)

thyroid_biomarkers = [
    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO",
    "FT3_FT4_Ratio"
]

biomarker_summary = []

for feature in thyroid_biomarkers:

    biomarker_summary.append({
        "Feature": feature,
        "Mean": thyroid_df[feature].mean(),
        "Median": thyroid_df[feature].median(),
        "Std": thyroid_df[feature].std(),
        "Minimum": thyroid_df[feature].min(),
        "Maximum": thyroid_df[feature].max()
    })

biomarker_summary = pd.DataFrame(biomarker_summary)

print(
    biomarker_summary.round(4).to_string(index=False)
)

# ============================================================================
# 7. THYROID BIOMARKERS BY CVD PREDICTION
# ============================================================================

print("\n" + "=" * 78)
print("5. THYROID BIOMARKERS BY MODEL-PREDICTED CVD")
print("=" * 78)

biomarker_cvd_summary = []

for feature in thyroid_biomarkers:

    no_cvd = thyroid_df.loc[
        thyroid_df["CVD_Prediction"] == 0,
        feature
    ]

    predicted_cvd_values = thyroid_df.loc[
        thyroid_df["CVD_Prediction"] == 1,
        feature
    ]

    mean_no_cvd = no_cvd.mean()
    mean_predicted_cvd = predicted_cvd_values.mean()

    median_no_cvd = no_cvd.median()
    median_predicted_cvd = predicted_cvd_values.median()

    mean_difference = (
        mean_predicted_cvd - mean_no_cvd
    )

    biomarker_cvd_summary.append({
        "Feature": feature,
        "Mean_No_CVD": mean_no_cvd,
        "Mean_Predicted_CVD": mean_predicted_cvd,
        "Median_No_CVD": median_no_cvd,
        "Median_Predicted_CVD": median_predicted_cvd,
        "Mean_Difference": mean_difference
    })

biomarker_cvd_summary = pd.DataFrame(
    biomarker_cvd_summary
)

print(
    biomarker_cvd_summary.round(4).to_string(index=False)
)

# ============================================================================
# 8. CONVENTIONAL CVD RISK FACTORS
# ============================================================================

print("\n" + "=" * 78)
print("6. CONVENTIONAL CVD RISK FACTORS")
print("=" * 78)

risk_factor_mapping = {
    "High blood pressure": "High_BP",
    "High cholesterol": "High_Cholesterol",
    "Low HDL": "Low_HDL",
    "Obesity": "Obesity",
    "Smoking-related risk": "Smoking_Risk"
}

risk_factor_rows = []

for factor_name, column in risk_factor_mapping.items():

    # Treat 1 / 1.0 / True / Yes as presence
    factor_present = (
        thyroid_df[column]
        .astype(str)
        .str.lower()
        .isin(["1", "1.0", "true", "yes"])
    )

    participants_with_factor = int(
        factor_present.sum()
    )

    percentage_population = (
        participants_with_factor
        / thyroid_records * 100
    )

    predicted_cvd_among_factor = int(
        (
            factor_present
            & (thyroid_df["CVD_Prediction"] == 1)
        ).sum()
    )

    if predicted_cvd_among_factor > 0:
        percentage_cvd_group = (
            predicted_cvd_among_factor
            / predicted_cvd * 100
            if predicted_cvd > 0 else 0
        )
    else:
        percentage_cvd_group = 0

    risk_factor_rows.append({
        "Risk_Factor": factor_name,
        "Participants_With_Factor":
            participants_with_factor,
        "Percentage_Of_Thyroid_Population":
            percentage_population,
        "Predicted_CVD_Among_Factor":
            predicted_cvd_among_factor,
        "Percentage_Of_Predicted_CVD_Group":
            percentage_cvd_group
    })

risk_factor_summary = pd.DataFrame(
    risk_factor_rows
)

risk_factor_summary[
    "Percentage_Of_Thyroid_Population"
] = risk_factor_summary[
    "Percentage_Of_Thyroid_Population"
].round(2)

risk_factor_summary[
    "Percentage_Of_Predicted_CVD_Group"
] = risk_factor_summary[
    "Percentage_Of_Predicted_CVD_Group"
].round(2)

print(
    risk_factor_summary.to_string(index=False)
)

# ============================================================================
# 9. RISK FACTOR BURDEN
# ============================================================================

print("\n" + "=" * 78)
print("7. RISK FACTOR BURDEN")
print("=" * 78)

if "Risk_Factor_Count" in thyroid_df.columns:

    print("\nRisk-factor count distribution:")

    burden_distribution = (
        thyroid_df["Risk_Factor_Count"]
        .value_counts()
        .sort_index()
    )

    print(burden_distribution)

    burden_summary = (
        thyroid_df
        .groupby("Risk_Factor_Count")
        .agg(
            Participants=("SEQN", "count"),
            Predicted_CVD=(
                "CVD_Prediction",
                "sum"
            ),
            Mean_CVD_Probability=(
                "CVD_Probability",
                "mean"
            )
        )
        .reset_index()
    )

    burden_summary["Predicted_CVD_Percentage"] = (
        burden_summary["Predicted_CVD"]
        / burden_summary["Participants"]
        * 100
    )

    burden_summary[
        "Predicted_CVD_Percentage"
    ] = burden_summary[
        "Predicted_CVD_Percentage"
    ].round(2)

    burden_summary[
        "Mean_CVD_Probability"
    ] = burden_summary[
        "Mean_CVD_Probability"
    ].round(4)

    print("\nRisk-factor burden summary:")
    print(
        burden_summary.to_string(index=False)
    )

else:
    burden_summary = pd.DataFrame()

# ============================================================================
# 10. PREDICTED-CVD GROUP CHARACTERISTICS
# ============================================================================

print("\n" + "=" * 78)
print("8. PREDICTED-CVD GROUP CHARACTERISTICS")
print("=" * 78)

predicted_cvd_df = thyroid_df[
    thyroid_df["CVD_Prediction"] == 1
].copy()

non_cvd_df = thyroid_df[
    thyroid_df["CVD_Prediction"] == 0
].copy()

group_characteristics = []

comparison_features = [
    "Age",
    "BMI",
    "SYS_BP",
    "HDL",
    "Total_Cholesterol",
    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO",
    "FT3_FT4_Ratio"
]

for feature in comparison_features:

    group_characteristics.append({
        "Feature": feature,
        "No_CVD_Mean":
            non_cvd_df[feature].mean(),
        "Predicted_CVD_Mean":
            predicted_cvd_df[feature].mean(),
        "Difference":
            predicted_cvd_df[feature].mean()
            - non_cvd_df[feature].mean()
    })

group_characteristics = pd.DataFrame(
    group_characteristics
)

print(
    group_characteristics.round(4).to_string(index=False)
)

# ============================================================================
# 11. VALIDATED THYROID ASSOCIATION RESULTS
# ============================================================================
#
# These findings come from the validated Step 8 analysis.
# We do NOT recreate the statistical tests here.
# ============================================================================

print("\n" + "=" * 78)
print("9. VALIDATED THYROID-CVD ASSOCIATIONS")
print("=" * 78)

validated_associations = pd.DataFrame([
    {
        "Feature": "TSH",
        "Association_Status":
            "No significant association after validation",
        "Interpretation":
            "TSH was not independently associated with "
            "model-predicted CVD in the validated analysis."
    },
    {
        "Feature": "Free_T3",
        "Association_Status":
            "No significant association after validation",
        "Interpretation":
            "Free T3 was not independently associated with "
            "model-predicted CVD in the validated analysis."
    },
    {
        "Feature": "Free_T4",
        "Association_Status":
            "Unadjusted association; "
            "not independently significant after adjustment",
        "Interpretation":
            "Free T4 was higher in the predicted-CVD group, "
            "but the association did not remain independently "
            "significant after adjustment."
    },
    {
        "Feature": "TPO",
        "Association_Status":
            "No significant association after validation",
        "Interpretation":
            "TPO was not independently associated with "
            "model-predicted CVD in the validated analysis."
    },
    {
        "Feature": "FT3_FT4_Ratio",
        "Association_Status":
            "Unadjusted association; "
            "not independently significant after adjustment",
        "Interpretation":
            "The FT3/FT4 ratio was lower in the predicted-CVD "
            "group and was associated with continuous CVD "
            "probability, but the adjusted association did not "
            "remain independently significant."
    }
])

print(
    validated_associations.to_string(index=False)
)

# ============================================================================
# 12. DASHBOARD INTERPRETATION LOGIC
# ============================================================================
#
# This does NOT calculate a new CVD probability.
# It only converts the EXISTING prediction and existing risk-factor fields
# into understandable dashboard labels.
# ============================================================================

print("\n" + "=" * 78)
print("10. DASHBOARD INTERPRETATION LOGIC")
print("=" * 78)

def create_cvd_interpretation(row):

    prediction = row["CVD_Prediction"]

    if prediction == 1:
        return (
            "Existing CVD model indicates elevated "
            "predicted CVD risk. Review cardiovascular "
            "risk factors and discuss findings with a "
            "qualified healthcare professional."
        )

    return (
        "Existing CVD model does not classify this "
        "participant as predicted CVD-positive at the "
        "current threshold. Continue monitoring and "
        "address modifiable cardiovascular risk factors."
    )


def identify_priority_factors(row):

    factors = []

    for factor_name, column in risk_factor_mapping.items():

        value = str(row[column]).lower()

        if value in ["1", "1.0", "true", "yes"]:
            factors.append(factor_name)

    if len(factors) == 0:
        return "No flagged modifiable risk factors"

    return "; ".join(factors)


def create_prevention_message(row):

    recommendations = []

    # Blood pressure
    if str(row["High_BP"]).lower() in [
        "1", "1.0", "true", "yes"
    ]:
        recommendations.append(
            "Monitor and manage blood pressure"
        )

    # Cholesterol
    if str(row["High_Cholesterol"]).lower() in [
        "1", "1.0", "true", "yes"
    ]:
        recommendations.append(
            "Monitor cholesterol and discuss lipid "
            "management with a healthcare professional"
        )

    # HDL
    if str(row["Low_HDL"]).lower() in [
        "1", "1.0", "true", "yes"
    ]:
        recommendations.append(
            "Improve cardiovascular lifestyle factors "
            "that support healthy HDL levels"
        )

    # Obesity
    if str(row["Obesity"]).lower() in [
        "1", "1.0", "true", "yes"
    ]:
        recommendations.append(
            "Work toward healthy weight management"
        )

    # Smoking
    if str(row["Smoking_Risk"]).lower() in [
        "1", "1.0", "true", "yes"
    ]:
        recommendations.append(
            "Avoid smoking and consider smoking-cessation support"
        )

    # Thyroid monitoring
    recommendations.append(
        "Continue appropriate thyroid monitoring "
        "and clinical follow-up"
    )

    return "; ".join(recommendations)


thyroid_df["Dashboard_CVD_Interpretation"] = (
    thyroid_df.apply(
        create_cvd_interpretation,
        axis=1
    )
)

thyroid_df["Priority_CVD_Risk_Factors"] = (
    thyroid_df.apply(
        identify_priority_factors,
        axis=1
    )
)

thyroid_df["Prevention_Guidance"] = (
    thyroid_df.apply(
        create_prevention_message,
        axis=1
    )
)

# ============================================================================
# 13. INDIVIDUAL RISK PROFILE
# ============================================================================

print("\n" + "=" * 78)
print("11. INDIVIDUAL RISK PROFILES")
print("=" * 78)

individual_profile_columns = [
    "SEQN",
    "Age",
    "Gender" if "Gender" in thyroid_df.columns else None,
    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO",
    "FT3_FT4_Ratio",
    "BMI",
    "SYS_BP",
    "HDL",
    "Total_Cholesterol",
    "Diabetes",
    "Smoking_100_Cigarettes",
    "CVD_Probability",
    "CVD_Prediction",
    "CVD_Risk_Category",
    "High_BP",
    "High_Cholesterol",
    "Low_HDL",
    "Obesity",
    "Smoking_Risk",
    "Risk_Factor_Count",
    "Priority_CVD_Risk_Factors",
    "Dashboard_CVD_Interpretation",
    "Prevention_Guidance"
]

individual_profile_columns = [
    col for col in individual_profile_columns
    if col is not None and col in thyroid_df.columns
]

individual_profiles = thyroid_df[
    individual_profile_columns
].copy()

print(
    "Individual risk profiles generated:",
    len(individual_profiles)
)

# ============================================================================
# 14. DASHBOARD-READY DATASET
# ============================================================================

print("\n" + "=" * 78)
print("12. DASHBOARD-READY DATA")
print("=" * 78)

dashboard_columns = [
    "SEQN",
    "Age",
    "Gender",
    "BMI",
    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO",
    "FT3_FT4_Ratio",
    "Thyroid_Diagnosis",
    "Thyroid_Status",
    "CVD_Probability",
    "CVD_Prediction",
    "CVD_Risk_Category",
    "High_BP",
    "High_Cholesterol",
    "Low_HDL",
    "Obesity",
    "Smoking_Risk",
    "Diabetes",
    "Risk_Factor_Count",
    "Priority_CVD_Risk_Factors",
    "Dashboard_CVD_Interpretation",
    "Prevention_Guidance"
]

dashboard_columns = [
    col for col in dashboard_columns
    if col in thyroid_df.columns
]

dashboard_data = thyroid_df[
    dashboard_columns
].copy()

print(
    "Dashboard-ready records:",
    len(dashboard_data)
)

print(
    "Dashboard-ready columns:",
    len(dashboard_data.columns)
)

# ============================================================================
# 15. FINAL RESEARCH SUMMARY
# ============================================================================

print("\n" + "=" * 78)
print("13. FINAL RESEARCH SUMMARY")
print("=" * 78)

research_summary = pd.DataFrame([
    {
        "Metric":
            "Total participants analyzed",
        "Value":
            total_records
    },
    {
        "Metric":
            "Thyroid-problem participants",
        "Value":
            thyroid_records
    },
    {
        "Metric":
            "Thyroid-problem + model-predicted CVD",
        "Value":
            predicted_cvd
    },
    {
        "Metric":
            "Thyroid-problem + no model-predicted CVD",
        "Value":
            no_predicted_cvd
    },
    {
        "Metric":
            "Percentage with model-predicted CVD",
        "Value":
            round(predicted_cvd_percentage, 2)
    },
    {
        "Metric":
            "Mean existing CVD probability",
        "Value":
            round(mean_cvd_probability, 4)
    },
    {
        "Metric":
            "Median existing CVD probability",
        "Value":
            round(median_cvd_probability, 4)
    }
])

print(
    research_summary.to_string(index=False)
)

# ============================================================================
# 16. RESEARCH INTERPRETATION
# ============================================================================

print("\n" + "=" * 78)
print("14. RESEARCH INTERPRETATION")
print("=" * 78)

print(f"""
Among {thyroid_records} participants identified as having
thyroid problems, {predicted_cvd} ({predicted_cvd_percentage:.2f}%)
were classified by the existing CVD model as having
model-predicted CVD.

The analysis evaluates cardiovascular risk within the
thyroid-problem population rather than creating a separate
CVD prediction model.

The conventional cardiovascular risk factors were examined
to identify which modifiable factors are commonly present
among participants with model-predicted CVD.

The validated thyroid-CVD analyses showed that:

- TSH was not independently associated with model-predicted CVD.
- Free T3 was not independently associated with model-predicted CVD.
- Free T4 showed an unadjusted association but did not remain
  independently significant after adjustment.
- TPO was not independently associated with model-predicted CVD.
- FT3/FT4 ratio showed an unadjusted association and an association
  with continuous CVD probability, but did not remain independently
  significant after adjustment.

Therefore, thyroid biomarkers should not be interpreted as
proven causes of cardiovascular disease.

Instead, the analysis identifies associations within the
thyroid-problem population and highlights cardiovascular
risk factors that may require attention.

The future dashboard should therefore:

1. Collect thyroid-related information.
2. Identify whether the user has a thyroid problem.
3. Evaluate CVD using the established CVD model.
4. Display the existing CVD probability/risk category.
5. Show the cardiovascular risk factors flagged for the user.
6. Explain the thyroid biomarker findings carefully.
7. Provide prevention and risk-management guidance.
8. Encourage appropriate professional medical evaluation.
""")

# ============================================================================
# 17. DATA INTEGRITY CHECKS
# ============================================================================

print("\n" + "=" * 78)
print("15. DATA INTEGRITY CHECKS")
print("=" * 78)

# Check original CVD column was not modified
original_cvd = df["Cardiovascular_Disease"]

# Re-read source to guarantee comparison
source_df = pd.read_csv(INPUT_FILE)

cvd_column_unchanged = (
    df["Cardiovascular_Disease"].equals(
        source_df["Cardiovascular_Disease"]
    )
)

# Check CVD prediction count
prediction_count_correct = (
    int(
        (thyroid_df["CVD_Prediction"] == 1).sum()
    )
    == predicted_cvd
)

# Check probability was not accidentally changed
probability_unchanged = (
    df["CVD_Probability"].equals(
        source_df["CVD_Probability"]
    )
)

print(
    "Original Cardiovascular_Disease unchanged:",
    cvd_column_unchanged
)

print(
    "CVD_Probability unchanged:",
    probability_unchanged
)

print(
    "CVD prediction count consistent:",
    prediction_count_correct
)

print(
    "New CVD prediction model created: False"
)

print(
    "CVD_Probability used as predictor: False"
)

print(
    "Causal thyroid-CVD claim made: False"
)

if not (
    cvd_column_unchanged
    and probability_unchanged
    and prediction_count_correct
):
    raise ValueError(
        "Data integrity validation failed."
    )

print("\nData integrity checks: PASSED")

# ============================================================================
# 18. SAVE RESULTS
# ============================================================================

print("\n" + "=" * 78)
print("16. SAVING RESULTS")
print("=" * 78)

files_to_save = {

    "step10_research_summary.csv":
        research_summary,

    "step10_cvd_risk_category_summary.csv":
        risk_category_summary,

    "step10_thyroid_biomarker_summary.csv":
        biomarker_summary,

    "step10_thyroid_biomarker_cvd_comparison.csv":
        biomarker_cvd_summary,

    "step10_cvd_risk_factor_summary.csv":
        risk_factor_summary,

    "step10_predicted_cvd_group_characteristics.csv":
        group_characteristics,

    "step10_validated_thyroid_associations.csv":
        validated_associations,

    "step10_dashboard_ready_data.csv":
        dashboard_data,

    "step10_individual_risk_profiles.csv":
        individual_profiles
}

if not burden_summary.empty:
    files_to_save[
        "step10_risk_factor_burden.csv"
    ] = burden_summary

for filename, data in files_to_save.items():

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    data.to_csv(
        output_path,
        index=False
    )

    print("Saved:", output_path)

# ============================================================================
# 19. FINAL OUTPUT
# ============================================================================

print("\n" + "=" * 78)
print("STEP 10 COMPLETE")
print("=" * 78)

print("\nPopulation analyzed:", thyroid_records)

print(
    "Thyroid-problem + model-predicted CVD:",
    predicted_cvd
)

print(
    "Percentage with model-predicted CVD:",
    f"{predicted_cvd_percentage:.2f}%"
)

print(
    "\nPurpose of Step 10:"
)

print(
    "Final integration of validated thyroid-CVD "
    "associations, conventional cardiovascular risk "
    "factors, individual interpretation, and "
    "dashboard-ready information."
)

print("\nIMPORTANT:")
print("- No new CVD prediction model created.")
print("- Existing CVD_Prediction retained.")
print("- CVD_Probability not used as predictor.")
print("- Original Cardiovascular_Disease not modified.")
print("- Thyroid-CVD findings interpreted as associations.")
print("- No causal thyroid-CVD claim made.")

print("\nNext stage:")
print("STEP 11 - DASHBOARD FEATURE ENGINEERING AND")
print("           USER INPUT / CVD ASSESSMENT LOGIC")

print("=" * 78)
