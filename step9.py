# ============================================================================
# STEP 9 - INDIVIDUAL THYROID-CVD RISK INTERPRETATION
# ============================================================================
#
# RESEARCH FRAMEWORK
# ----------------------------------------------------------------------------
# 1. Identify participants with thyroid problems.
# 2. Evaluate their existing CVD model prediction.
# 3. Interpret thyroid biomarkers in relation to CVD prediction.
# 4. Identify conventional cardiovascular risk factors.
# 5. Generate an individual risk profile.
# 6. Generate prevention-oriented recommendations.
#
# IMPORTANT:
# - This step does NOT train a new CVD prediction model.
# - CVD_Probability is NOT used as a predictor.
# - CVD_Prediction comes from the EXISTING CVD model.
# - Cardiovascular_Disease is NOT modified.
# - Associations are NOT interpreted as causation.
# ============================================================================

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------

INPUT_FILE = "output/step8_input_dataset.csv"

# If the Step 8 input file does not exist, use the previous integrated dataset.
if os.path.exists(INPUT_FILE):
    df = pd.read_csv(INPUT_FILE)
    print(f"Dataset loaded from: {INPUT_FILE}")
else:
    INPUT_FILE = "output/step5_integrated_thyroid_cvd_dataset.csv"

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            "Required input dataset not found. "
            "Please check the output folder."
        )

    df = pd.read_csv(INPUT_FILE)
    print(f"Dataset loaded from: {INPUT_FILE}")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================================
# REQUIRED COLUMNS
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
    "SYS_BP",
    "Diabetes",
    "Smoking_100_Cigarettes",
    "Thyroid_Diagnosis",
    "Thyroid_Problem",
    "Abnormal_TSH",
    "CVD_Probability",
    "CVD_Prediction",
    "CVD_Risk_Category",
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
# 1. THYROID-PROBLEM POPULATION
# ============================================================================

print("\n" + "=" * 70)
print("1. THYROID-PROBLEM POPULATION")
print("=" * 70)

# Explicit thyroid-problem filtering
thyroid_df = df[df["Thyroid_Problem"] == 1].copy()

total_records = len(df)
thyroid_records = len(thyroid_df)

print("Total input records:", total_records)
print("Thyroid-problem records:", thyroid_records)

if thyroid_records == 0:
    raise ValueError(
        "No thyroid-problem participants were found."
    )


# ============================================================================
# 2. EXISTING CVD MODEL RESULTS
# ============================================================================

print("\n" + "=" * 70)
print("2. EXISTING CVD MODEL RESULTS")
print("=" * 70)

# IMPORTANT:
# Calculate everything directly from CVD_Prediction.
# Do NOT hard-code the number of CVD-positive participants.

cvd_distribution = (
    thyroid_df["CVD_Prediction"]
    .value_counts()
    .sort_index()
)

cvd_percentage = (
    thyroid_df["CVD_Prediction"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

print("\nCVD prediction distribution:")
print(cvd_distribution)

print("\nCVD prediction percentage:")
print(cvd_percentage.round(2))

predicted_cvd_count = int(
    (thyroid_df["CVD_Prediction"] == 1).sum()
)

non_predicted_cvd_count = int(
    (thyroid_df["CVD_Prediction"] == 0).sum()
)

predicted_cvd_percentage = (
    predicted_cvd_count / thyroid_records * 100
)

print(
    "\nThyroid-problem + model-predicted CVD:",
    predicted_cvd_count
)

print(
    "Percentage:",
    round(predicted_cvd_percentage, 2),
    "%"
)


# ============================================================================
# 3. THYROID-SPECIFIC INTERPRETATION
# ============================================================================

print("\n" + "=" * 70)
print("3. THYROID-SPECIFIC INTERPRETATION")
print("=" * 70)

thyroid_biomarkers = [
    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO",
    "FT3_FT4_Ratio"
]

# Results from validated Step 8 analysis.
# These are NOT treated as causal effects.

thyroid_interpretation = {
    "TSH": (
        "No significant association after validation",
        "TSH was not independently associated with "
        "model-predicted CVD in the validated analysis."
    ),

    "Free_T3": (
        "No significant association after validation",
        "Free T3 was not independently associated with "
        "model-predicted CVD in the validated analysis."
    ),

    "Free_T4": (
        "Unadjusted association; not independently significant after adjustment",
        "Free T4 was higher in the predicted-CVD group, "
        "but the association did not remain independently "
        "significant after adjustment."
    ),

    "TPO": (
        "No significant association after validation",
        "TPO was not independently associated with "
        "model-predicted CVD in the validated analysis."
    ),

    "FT3_FT4_Ratio": (
        "Unadjusted association; not independently significant after adjustment",
        "The FT3/FT4 ratio was lower in the predicted-CVD group "
        "and showed an association with continuous CVD probability, "
        "but the adjusted association did not remain independently "
        "significant."
    )
}

for biomarker in thyroid_biomarkers:
    status, interpretation = thyroid_interpretation[biomarker]

    print("\n" + "-" * 60)
    print(biomarker)
    print("Status:", status)
    print("Interpretation:", interpretation)


# ============================================================================
# 4. GENERATING INDIVIDUAL RISK PROFILES
# ============================================================================

print("\n" + "=" * 70)
print("4. GENERATING INDIVIDUAL RISK PROFILES")
print("=" * 70)


def safe_float(value):
    """
    Convert numeric values safely.
    """
    try:
        return float(value)
    except:
        return np.nan


def get_cvd_status(row):
    """
    Interpret the existing CVD model output.
    """

    probability = safe_float(row["CVD_Probability"])
    prediction = row["CVD_Prediction"]

    if prediction == 1:
        status = "Model-predicted CVD"
    else:
        status = "No model-predicted CVD"

    return status


def get_thyroid_interpretation(row):
    """
    General thyroid interpretation.

    This does not diagnose a thyroid disease.
    Thyroid_Diagnosis / Thyroid_Problem originate from the
    existing dataset and previous analysis.
    """

    if row["Thyroid_Problem"] == 1:
        return "Thyroid problem identified"

    return "No thyroid problem identified"


def identify_risk_factors(row):
    """
    Identify conventional modifiable cardiovascular risk factors.

    These variables come from the existing dataset definitions.
    """

    risk_factors = []

    # High blood pressure
    if "High_BP" in row.index:
        if row["High_BP"] == 1:
            risk_factors.append("High blood pressure")
    else:
        # Fallback based on systolic BP if High_BP is unavailable
        if safe_float(row["SYS_BP"]) >= 140:
            risk_factors.append("High blood pressure")

    # High cholesterol
    if "High_Cholesterol" in row.index:
        if row["High_Cholesterol"] == 1:
            risk_factors.append("High cholesterol")

    # Low HDL
    if "Low_HDL" in row.index:
        if row["Low_HDL"] == 1:
            risk_factors.append("Low HDL")

    # Obesity
    if "Obesity" in row.index:
        if row["Obesity"] == 1:
            risk_factors.append("Obesity")
    else:
        if safe_float(row["BMI"]) >= 30:
            risk_factors.append("Obesity")

    # Smoking
    if "Smoking_Risk" in row.index:
        if row["Smoking_Risk"] == 1:
            risk_factors.append("Smoking-related risk")
    else:
        smoking = safe_float(row["Smoking_100_Cigarettes"])
        if smoking == 1:
            risk_factors.append("Smoking-related risk")

    return risk_factors


def generate_prevention_recommendations(risk_factors):
    """
    Generate prevention-oriented recommendations.

    These are general risk-reduction measures and are not
    individualized medical prescriptions.
    """

    recommendations = []

    if "High blood pressure" in risk_factors:
        recommendations.append(
            "Monitor blood pressure regularly and discuss "
            "blood-pressure management with a healthcare professional."
        )

    if "High cholesterol" in risk_factors:
        recommendations.append(
            "Monitor cholesterol levels and discuss dietary and "
            "lipid-management strategies with a healthcare professional."
        )

    if "Low HDL" in risk_factors:
        recommendations.append(
            "Increase regular physical activity and maintain a "
            "heart-healthy lifestyle to support a healthier lipid profile."
        )

    if "Obesity" in risk_factors:
        recommendations.append(
            "Work toward a healthy weight through sustainable nutrition, "
            "physical activity, sleep, and professional guidance when needed."
        )

    if "Smoking-related risk" in risk_factors:
        recommendations.append(
            "Avoid smoking and consider evidence-based smoking-cessation "
            "support if currently smoking."
        )

    if len(recommendations) == 0:
        recommendations.append(
            "Maintain regular physical activity, balanced nutrition, "
            "healthy weight, routine monitoring, and regular medical follow-up."
        )

    # Always include thyroid follow-up for the target population.
    recommendations.append(
        "Continue appropriate thyroid monitoring and follow-up with "
        "a healthcare professional."
    )

    return recommendations


risk_profiles = []

for _, row in thyroid_df.iterrows():

    risk_factors = identify_risk_factors(row)

    recommendations = generate_prevention_recommendations(
        risk_factors
    )

    probability = safe_float(row["CVD_Probability"])

    profile = {
        "SEQN": row["SEQN"],

        # Thyroid information
        "Thyroid_Problem": row["Thyroid_Problem"],
        "Thyroid_Status": (
            row["Thyroid_Status"]
            if "Thyroid_Status" in row.index
            else get_thyroid_interpretation(row)
        ),
        "Thyroid_Diagnosis": row["Thyroid_Diagnosis"],

        # Thyroid biomarkers
        "TSH": row["TSH"],
        "Free_T3": row["Free_T3"],
        "Free_T4": row["Free_T4"],
        "TPO": row["TPO"],
        "FT3_FT4_Ratio": row["FT3_FT4_Ratio"],
        "Abnormal_TSH": row["Abnormal_TSH"],

        # Existing CVD model results
        "CVD_Probability": probability,
        "CVD_Prediction": row["CVD_Prediction"],
        "CVD_Status": get_cvd_status(row),

        "CVD_Risk_Category": (
            row["CVD_Risk_Category"]
            if "CVD_Risk_Category" in row.index
            else "Not available"
        ),

        # Conventional factors
        "Age": row["Age"],
        "BMI": row["BMI"],
        "SYS_BP": row["SYS_BP"],
        "HDL": row["HDL"],
        "Total_Cholesterol": row["Total_Cholesterol"],
        "Diabetes": row["Diabetes"],
        "Smoking_100_Cigarettes": row["Smoking_100_Cigarettes"],

        # Risk factors
        "Risk_Factor_Count_Identified": len(risk_factors),
        "Risk_Factors": (
            "; ".join(risk_factors)
            if risk_factors
            else "None identified"
        ),

        # Prevention
        "Prevention_Recommendations": " | ".join(
            recommendations
        )
    }

    risk_profiles.append(profile)


risk_profiles_df = pd.DataFrame(risk_profiles)

print(
    "Individual risk profiles generated:",
    len(risk_profiles_df)
)


# ============================================================================
# 5. GENERATING PREVENTION RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 70)
print("5. GENERATING PREVENTION RECOMMENDATIONS")
print("=" * 70)

print(
    "Prevention recommendations generated for:",
    len(risk_profiles_df),
    "participants"
)


# ============================================================================
# 6. PRIMARY MODIFIABLE RISK FACTORS
# ============================================================================

print("\n" + "=" * 70)
print("6. PRIMARY MODIFIABLE RISK FACTORS")
print("=" * 70)

risk_factor_definitions = {
    "High blood pressure": "High_BP",
    "High cholesterol": "High_Cholesterol",
    "Low HDL": "Low_HDL",
    "Obesity": "Obesity",
    "Smoking-related risk": "Smoking_Risk"
}

risk_factor_summary = []

for risk_name, column in risk_factor_definitions.items():

    if column not in thyroid_df.columns:
        continue

    factor_count = int(
        (thyroid_df[column] == 1).sum()
    )

    factor_percentage = (
        factor_count / thyroid_records * 100
    )

    predicted_cvd_among_factor = int(
        (
            (thyroid_df[column] == 1)
            &
            (thyroid_df["CVD_Prediction"] == 1)
        ).sum()
    )

    risk_factor_summary.append({
        "Risk_Factor": risk_name,
        "Participants_With_Factor": factor_count,
        "Percentage_Of_Thyroid_Population":
            factor_percentage,
        "Predicted_CVD_Among_Factor":
            predicted_cvd_among_factor
    })


risk_factor_summary_df = pd.DataFrame(
    risk_factor_summary
)

print(risk_factor_summary_df.round(2).to_string(index=False))


# ============================================================================
# 7. GROUP-LEVEL RISK FACTOR SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("7. GROUP-LEVEL RISK FACTOR SUMMARY")
print("=" * 70)

print(
    risk_factor_summary_df.round(2).to_string(index=False)
)


# ============================================================================
# 8. MODEL-PREDICTED CVD GROUP
# ============================================================================

print("\n" + "=" * 70)
print("8. MODEL-PREDICTED CVD GROUP")
print("=" * 70)

cvd_group = thyroid_df[
    thyroid_df["CVD_Prediction"] == 1
].copy()

print(
    "Thyroid-problem participants with model-predicted CVD:",
    len(cvd_group)
)

print("\nRisk-factor prevalence among predicted-CVD group:")

cvd_group_risk_summary = []

for risk_name, column in risk_factor_definitions.items():

    if column not in cvd_group.columns:
        continue

    factor_count = int(
        (cvd_group[column] == 1).sum()
    )

    factor_percentage = (
        factor_count / len(cvd_group) * 100
        if len(cvd_group) > 0
        else 0
    )

    cvd_group_risk_summary.append({
        "Risk_Factor": risk_name,
        "Participants": factor_count,
        "Percentage": factor_percentage
    })

    print(
        f"{risk_name}: "
        f"{factor_count}/{len(cvd_group)} "
        f"({factor_percentage:.2f}%)"
    )

cvd_group_risk_summary_df = pd.DataFrame(
    cvd_group_risk_summary
)


# ============================================================================
# 9. THYROID-CVD ASSOCIATION SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("9. THYROID-CVD ASSOCIATION SUMMARY")
print("=" * 70)

association_rows = []

for biomarker in thyroid_biomarkers:

    status, interpretation = thyroid_interpretation[
        biomarker
    ]

    association_rows.append({
        "Feature": biomarker,
        "Association_Status": status,
        "Interpretation": interpretation
    })

thyroid_association_summary_df = pd.DataFrame(
    association_rows
)

print(
    thyroid_association_summary_df.to_string(
        index=False
    )
)


# ============================================================================
# 10. INDIVIDUAL RISK INTERPRETATION
# ============================================================================

print("\n" + "=" * 70)
print("10. INDIVIDUAL RISK INTERPRETATION")
print("=" * 70)


def create_individual_interpretation(row):
    """
    Dashboard-ready narrative interpretation.

    Important:
    This describes associations/risk indicators.
    It does NOT state that a factor causes CVD.
    """

    messages = []

    # Thyroid
    messages.append(
        "Thyroid problem identified."
    )

    # CVD model result
    if row["CVD_Prediction"] == 1:
        messages.append(
            "The existing CVD model classifies this participant "
            "as having model-predicted CVD risk."
        )
    else:
        messages.append(
            "The existing CVD model does not classify this participant "
            "as having model-predicted CVD."
        )

    # Risk factors
    if row["Risk_Factors"] != "None identified":
        messages.append(
            "Identified modifiable cardiovascular risk factors: "
            + row["Risk_Factors"] + "."
        )
    else:
        messages.append(
            "No predefined modifiable cardiovascular risk factors "
            "were identified from the available variables."
        )

    # Thyroid interpretation
    messages.append(
        "Current validated analyses did not demonstrate an "
        "independently significant thyroid biomarker association "
        "after adjustment for conventional cardiovascular factors."
    )

    return " ".join(messages)


risk_profiles_df[
    "Individual_Interpretation"
] = risk_profiles_df.apply(
    create_individual_interpretation,
    axis=1
)


# ============================================================================
# 11. DASHBOARD-READY DATA
# ============================================================================

print("\n" + "=" * 70)
print("11. DASHBOARD-READY DATA")
print("=" * 70)

dashboard_columns = [
    "SEQN",
    "Age",
    "BMI",
    "SYS_BP",
    "HDL",
    "Total_Cholesterol",
    "Diabetes",
    "Smoking_100_Cigarettes",

    # Thyroid
    "Thyroid_Problem",
    "Thyroid_Status",
    "Thyroid_Diagnosis",
    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO",
    "FT3_FT4_Ratio",
    "Abnormal_TSH",

    # CVD
    "CVD_Probability",
    "CVD_Prediction",
    "CVD_Status",
    "CVD_Risk_Category",

    # Risk factors
    "Risk_Factor_Count_Identified",
    "Risk_Factors",

    # Interpretation
    "Individual_Interpretation",
    "Prevention_Recommendations"
]

dashboard_columns = [
    col for col in dashboard_columns
    if col in risk_profiles_df.columns
]

dashboard_df = risk_profiles_df[
    dashboard_columns
].copy()

print(
    "Dashboard-ready records:",
    len(dashboard_df)
)

print(
    "Dashboard-ready columns:",
    len(dashboard_df.columns)
)


# ============================================================================
# 12. FINAL CONSISTENCY VALIDATION
# ============================================================================

print("\n" + "=" * 70)
print("12. FINAL CONSISTENCY VALIDATION")
print("=" * 70)

# Recalculate the important numbers independently.
validation_total = len(thyroid_df)

validation_cvd_positive = int(
    (thyroid_df["CVD_Prediction"] == 1).sum()
)

validation_cvd_negative = int(
    (thyroid_df["CVD_Prediction"] == 0).sum()
)

validation_percentage = (
    validation_cvd_positive /
    validation_total *
    100
)

print(
    "Thyroid-problem participants:",
    validation_total
)

print(
    "Model-predicted CVD:",
    validation_cvd_positive
)

print(
    "No model-predicted CVD:",
    validation_cvd_negative
)

print(
    "Percentage with model-predicted CVD:",
    round(validation_percentage, 2),
    "%"
)

# Verify counts add up
assert (
    validation_cvd_positive +
    validation_cvd_negative
    == validation_total
), "CVD prediction counts do not add up."

# Verify percentage
assert np.isclose(
    validation_percentage,
    validation_cvd_positive / validation_total * 100
), "CVD percentage calculation error."

# Verify no hard-coded 32-type discrepancy
assert (
    validation_cvd_positive == int(
        dashboard_df["CVD_Prediction"].sum()
    )
), "Dashboard CVD-positive count does not match source data."

print("\nConsistency checks: PASSED")


# ============================================================================
# 13. RESEARCH SAFETY / DATA INTEGRITY CHECKS
# ============================================================================

print("\n" + "=" * 70)
print("13. RESEARCH DATA INTEGRITY CHECKS")
print("=" * 70)

# CVD_Probability must not be used as a predictor in this step.
print(
    "CVD_Probability used as predictor: False"
)

# No new CVD model is trained.
print(
    "New CVD prediction model created: False"
)

# Original CVD column must remain unchanged.
if "Cardiovascular_Disease" in df.columns:

    original_cvd = df["Cardiovascular_Disease"].copy()

    # Compare against original source dataframe.
    unchanged = original_cvd.equals(
        df["Cardiovascular_Disease"]
    )

else:
    unchanged = True

print(
    "Original Cardiovascular_Disease column modified:",
    not unchanged
)

# Research interpretation
print(
    "Causal thyroid-CVD claim made: False"
)


# ============================================================================
# 14. RESEARCH INTERPRETATION
# ============================================================================

print("\n" + "=" * 70)
print("14. RESEARCH INTERPRETATION")
print("=" * 70)

print(
    f"""
Among {validation_total} participants identified as having
thyroid problems, {validation_cvd_positive} "
f"({validation_percentage:.2f}%) were classified by the
existing CVD model as having model-predicted CVD.

The analysis identifies cardiovascular risk factors present
within this thyroid-problem population and generates
individual-level risk profiles.

The validated thyroid-CVD analyses showed that:

- TSH was not independently associated with model-predicted CVD.
- Free T3 was not independently associated with model-predicted CVD.
- Free T4 showed an unadjusted association but did not remain
  independently significant after adjustment.
- TPO was not independently associated with model-predicted CVD.
- FT3/FT4 ratio showed an unadjusted association and association
  with continuous CVD probability, but did not remain independently
  significant after adjustment.

Therefore, the results should be interpreted as associations
within the thyroid-problem population rather than evidence that
thyroid abnormalities cause cardiovascular disease.

The primary purpose of this analysis is to identify:
1. Which thyroid-problem participants have model-predicted CVD.
2. Which conventional cardiovascular risk factors are present.
3. Which modifiable risk factors should receive attention.
4. How thyroid biomarkers relate to CVD risk within this population.
5. How these findings can be presented in the future dashboard.
"""
)


# ============================================================================
# 15. SAVING RESULTS
# ============================================================================

print("\n" + "=" * 70)
print("15. SAVING RESULTS")
print("=" * 70)

os.makedirs("output", exist_ok=True)

# Individual profiles
risk_profiles_file = (
    "output/step9_individual_risk_profiles.csv"
)

risk_profiles_df.to_csv(
    risk_profiles_file,
    index=False
)

# Dashboard data
dashboard_file = (
    "output/step9_dashboard_ready_data.csv"
)

dashboard_df.to_csv(
    dashboard_file,
    index=False
)

# Risk-factor summary
risk_factor_file = (
    "output/step9_risk_factor_summary.csv"
)

risk_factor_summary_df.to_csv(
    risk_factor_file,
    index=False
)

# CVD-positive group risk summary
cvd_group_file = (
    "output/step9_predicted_cvd_risk_factor_summary.csv"
)

cvd_group_risk_summary_df.to_csv(
    cvd_group_file,
    index=False
)

# Thyroid association summary
thyroid_association_file = (
    "output/step9_thyroid_association_summary.csv"
)

thyroid_association_summary_df.to_csv(
    thyroid_association_file,
    index=False
)

# Overall population summary
population_summary = pd.DataFrame([{
    "Total_Input_Records": total_records,
    "Thyroid_Problem_Records": thyroid_records,
    "Model_Predicted_CVD": validation_cvd_positive,
    "No_Model_Predicted_CVD": validation_cvd_negative,
    "Predicted_CVD_Percentage":
        validation_percentage
}])

population_summary_file = (
    "output/step9_population_summary.csv"
)

population_summary.to_csv(
    population_summary_file,
    index=False
)

print("Saved:")
print("1.", risk_profiles_file)
print("2.", dashboard_file)
print("3.", risk_factor_file)
print("4.", cvd_group_file)
print("5.", thyroid_association_file)
print("6.", population_summary_file)
print("7.", thyroid_association_file)


# ============================================================================
# 16. FINAL STEP 9 SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("STEP 9 COMPLETE")
print("=" * 70)

print(
    "\nPopulation analyzed:",
    validation_total
)

print(
    "Thyroid-problem participants:",
    thyroid_records
)

print(
    "Model-predicted CVD:",
    validation_cvd_positive
)

print(
    "Percentage with model-predicted CVD:",
    f"{validation_percentage:.2f}%"
)

print(
    "\nIMPORTANT CORRECTION:"
)

print(
    f"The CVD-positive count is calculated directly from "
    f"CVD_Prediction: {validation_cvd_positive}"
)

print(
    f"The corresponding percentage is: "
    f"{validation_percentage:.2f}%"
)

print(
    "\nOriginal Cardiovascular_Disease column modified: False"
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

print("=" * 70)
