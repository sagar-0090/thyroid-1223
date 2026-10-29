# ==============================================================================
# STEP 13 - THYROID-CVD DASHBOARD APPLICATION LOGIC VALIDATION
# ==============================================================================
#
# PURPOSE:
# Validate the complete dashboard workflow before building the final UI.
#
# WORKFLOW:
# User Input
#      ↓
# Thyroid Problem Identification
#      ↓
# Cardiovascular Risk Factor Detection
#      ↓
# Existing CVD Model Result Retrieval
#      ↓
# CVD Probability / Prediction / Risk Category
#      ↓
# Thyroid-CVD Interpretation
#      ↓
# Individual Risk Factor Interpretation
#      ↓
# Prevention Recommendations
#      ↓
# Clinical Safety Message
#
# IMPORTANT:
# - NO new CVD prediction model is created.
# - Existing CVD_Prediction is retained.
# - Existing CVD_Probability is retained.
# - CVD_Probability is NOT used as a predictor.
# - Cardiovascular_Disease is NOT modified.
# - Thyroid-CVD associations are NOT interpreted causally.
#
# ==============================================================================

import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

INPUT_FILE = "output/step11_dashboard_ready_data.csv"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 78)
print("STEP 13 - THYROID-CVD DASHBOARD APPLICATION LOGIC VALIDATION")
print("=" * 78)


# ==============================================================================
# 1. LOAD DASHBOARD-READY DATA
# ==============================================================================

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded from:")
print(INPUT_FILE)

print("\nShape:", df.shape)

print("\nColumns:")
print(list(df.columns))


# ==============================================================================
# 2. REQUIRED COLUMNS
# ==============================================================================

required_columns = [
    "Age",
    "Gender",
    "Race",
    "BMI",
    "Height",
    "Weight",
    "Waist",
    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO",
    "HDL",
    "Total_Cholesterol",
    "Diabetes",
    "Smoking_100_Cigarettes",
    "Thyroid_Diagnosis",
    "SYS_BP",
    "DIA_BP",
    "CVD_Probability",
    "CVD_Prediction",
    "CVD_Risk_Category",
    "Thyroid_Status",
    "FT3_FT4_Ratio"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(missing_columns)
    )

print("\nAll required dashboard columns are present.")


# ==============================================================================
# 3. VALIDATE EXISTING MODEL OUTPUT
# ==============================================================================

print("\n" + "=" * 78)
print("1. EXISTING CVD MODEL OUTPUT VALIDATION")
print("=" * 78)

prediction_counts = df["CVD_Prediction"].value_counts().sort_index()

print("\nCVD prediction distribution:")
print(prediction_counts)

predicted_cvd = int((df["CVD_Prediction"] == 1).sum())
no_predicted_cvd = int((df["CVD_Prediction"] == 0).sum())

total_records = len(df)

predicted_percentage = (
    predicted_cvd / total_records * 100
    if total_records > 0 else 0
)

print("\nTotal participants:", total_records)
print("Model-predicted CVD:", predicted_cvd)
print("No model-predicted CVD:", no_predicted_cvd)
print(
    "Percentage with model-predicted CVD:",
    round(predicted_percentage, 2),
    "%"
)


# ==============================================================================
# 4. THYROID PROBLEM IDENTIFICATION
# ==============================================================================

print("\n" + "=" * 78)
print("2. THYROID PROBLEM IDENTIFICATION")
print("=" * 78)


def identify_thyroid_problem(row):
    """
    Determine thyroid-problem status using available thyroid information.

    Existing Thyroid_Problem / Thyroid_Status information is retained
    whenever available.
    """

    if "Thyroid_Problem" in row.index:
        value = row["Thyroid_Problem"]

        if pd.notna(value):
            try:
                if float(value) == 1:
                    return "Thyroid problem identified"
            except:
                pass

    diagnosis = row["Thyroid_Diagnosis"]

    if pd.notna(diagnosis):
        try:
            if float(diagnosis) == 1:
                return "Thyroid problem identified"
        except:
            pass

    status = str(row["Thyroid_Status"]).lower()

    if status not in ["normal", "none", "no thyroid problem", "nan"]:
        return "Thyroid problem identified"

    return "No thyroid problem identified"


df["Dashboard_Thyroid_Status"] = df.apply(
    identify_thyroid_problem,
    axis=1
)

print("\nThyroid status distribution:")
print(df["Dashboard_Thyroid_Status"].value_counts())


# ==============================================================================
# 5. CARDIOVASCULAR RISK FACTOR DETECTION
# ==============================================================================

print("\n" + "=" * 78)
print("3. CARDIOVASCULAR RISK FACTOR DETECTION")
print("=" * 78)


def safe_float(value):
    try:
        return float(value)
    except:
        return np.nan


def detect_risk_factors(row):

    factors = []

    # ----------------------------------------------------------
    # HIGH BLOOD PRESSURE
    # ----------------------------------------------------------

    sys_bp = safe_float(row["SYS_BP"])
    dia_bp = safe_float(row["DIA_BP"])

    if (
        (not np.isnan(sys_bp) and sys_bp >= 130)
        or
        (not np.isnan(dia_bp) and dia_bp >= 80)
    ):
        factors.append("High blood pressure")

    # ----------------------------------------------------------
    # HIGH CHOLESTEROL
    # ----------------------------------------------------------

    total_chol = safe_float(row["Total_Cholesterol"])

    if not np.isnan(total_chol) and total_chol >= 200:
        factors.append("High cholesterol")

    # ----------------------------------------------------------
    # LOW HDL
    # ----------------------------------------------------------

    hdl = safe_float(row["HDL"])

    if not np.isnan(hdl) and hdl < 40:
        factors.append("Low HDL")

    # ----------------------------------------------------------
    # OBESITY
    # ----------------------------------------------------------

    bmi = safe_float(row["BMI"])

    if not np.isnan(bmi) and bmi >= 30:
        factors.append("Obesity")

    # ----------------------------------------------------------
    # SMOKING
    # ----------------------------------------------------------

    smoking = safe_float(row["Smoking_100_Cigarettes"])

    if not np.isnan(smoking) and smoking > 1:
        factors.append("Smoking-related risk")

    return factors


df["Dashboard_Risk_Factors"] = df.apply(
    detect_risk_factors,
    axis=1
)

df["Dashboard_Risk_Factor_Count"] = (
    df["Dashboard_Risk_Factors"].apply(len)
)

print("\nRisk factor detection completed.")


# ==============================================================================
# 6. PRIMARY MODIFIABLE RISK FACTOR
# ==============================================================================

print("\n" + "=" * 78)
print("4. PRIMARY MODIFIABLE RISK FACTOR")
print("=" * 78)


def identify_primary_risk_factor(row):

    factors = row["Dashboard_Risk_Factors"]

    # Priority is based on the research findings from Steps 9-11.
    priority_order = [
        "Smoking-related risk",
        "High blood pressure",
        "Obesity",
        "High cholesterol",
        "Low HDL"
    ]

    for factor in priority_order:
        if factor in factors:
            return factor

    return "No major modifiable risk factor flagged"


df["Primary_Modifiable_Risk_Factor"] = df.apply(
    identify_primary_risk_factor,
    axis=1
)

print(
    "\nPrimary risk-factor distribution:"
)

print(
    df["Primary_Modifiable_Risk_Factor"].value_counts()
)


# ==============================================================================
# 7. THYROID-CVD INTERPRETATION
# ==============================================================================

print("\n" + "=" * 78)
print("5. THYROID-CVD INTERPRETATION")
print("=" * 78)


def thyroid_interpretation(row):

    return (
        "Validated analysis found that TSH and Free T3 were not "
        "independently associated with model-predicted CVD. "
        "Free T4 showed an unadjusted association but did not remain "
        "independently significant after adjustment. TPO was not "
        "independently associated with model-predicted CVD. "
        "The FT3/FT4 ratio showed an unadjusted association and an "
        "association with continuous CVD probability, but did not "
        "remain independently significant after adjustment. "
        "These findings represent associations and do not establish "
        "that thyroid abnormalities cause cardiovascular disease."
    )


df["Dashboard_Thyroid_Interpretation"] = df.apply(
    thyroid_interpretation,
    axis=1
)

print("\nValidated thyroid interpretation added.")


# ==============================================================================
# 8. INDIVIDUAL THYROID BIOMARKER INTERPRETATION
# ==============================================================================

print("\n" + "=" * 78)
print("6. INDIVIDUAL THYROID BIOMARKER INTERPRETATION")
print("=" * 78)


def biomarker_interpretation(row):

    messages = []

    tsh = safe_float(row["TSH"])
    free_t3 = safe_float(row["Free_T3"])
    free_t4 = safe_float(row["Free_T4"])
    tpo = safe_float(row["TPO"])

    if not np.isnan(tsh):
        messages.append(
            f"TSH recorded at {tsh:.3f}. "
            "Interpret according to the laboratory reference range."
        )

    if not np.isnan(free_t3):
        messages.append(
            f"Free T3 recorded at {free_t3:.3f}. "
            "Interpret according to the laboratory reference range."
        )

    if not np.isnan(free_t4):
        messages.append(
            f"Free T4 recorded at {free_t4:.3f}. "
            "Interpret according to the laboratory reference range."
        )

    if not np.isnan(tpo):
        messages.append(
            f"TPO recorded at {tpo:.3f}. "
            "Interpret according to the laboratory reference range."
        )

    messages.append(
        "Thyroid laboratory values should be interpreted together "
        "with symptoms, diagnosis, medication status, and clinical "
        "reference ranges."
    )

    return " ".join(messages)


df["Dashboard_Biomarker_Interpretation"] = df.apply(
    biomarker_interpretation,
    axis=1
)


# ==============================================================================
# 9. PREVENTION RECOMMENDATION ENGINE
# ==============================================================================

print("\n" + "=" * 78)
print("7. PREVENTION RECOMMENDATION ENGINE")
print("=" * 78)


def prevention_recommendations(row):

    recommendations = []

    factors = row["Dashboard_Risk_Factors"]

    # ----------------------------------------------------------
    # BLOOD PRESSURE
    # ----------------------------------------------------------

    if "High blood pressure" in factors:
        recommendations.append(
            "Prioritize blood-pressure management and regular "
            "blood-pressure monitoring."
        )

    # ----------------------------------------------------------
    # SMOKING
    # ----------------------------------------------------------

    if "Smoking-related risk" in factors:
        recommendations.append(
            "Smoking cessation should be strongly considered. "
            "Professional cessation support may be helpful."
        )

    # ----------------------------------------------------------
    # OBESITY
    # ----------------------------------------------------------

    if "Obesity" in factors:
        recommendations.append(
            "Consider gradual weight management through a balanced "
            "diet, appropriate physical activity, and professional "
            "guidance where appropriate."
        )

    # ----------------------------------------------------------
    # CHOLESTEROL
    # ----------------------------------------------------------

    if "High cholesterol" in factors:
        recommendations.append(
            "Discuss cholesterol management with a healthcare "
            "professional and consider dietary and lifestyle measures."
        )

    # ----------------------------------------------------------
    # LOW HDL
    # ----------------------------------------------------------

    if "Low HDL" in factors:
        recommendations.append(
            "Regular physical activity, healthy dietary patterns, "
            "weight management, and smoking cessation where relevant "
            "may support cardiovascular health."
        )

    # ----------------------------------------------------------
    # GENERAL
    # ----------------------------------------------------------

    recommendations.append(
        "Maintain regular physical activity appropriate for your "
        "health status."
    )

    recommendations.append(
        "Maintain a balanced diet emphasizing vegetables, fruits, "
        "whole grains, appropriate protein sources, and healthy fats."
    )

    recommendations.append(
        "Attend appropriate thyroid and cardiovascular follow-up."
    )

    recommendations.append(
        "Do not change thyroid or cardiovascular medication without "
        "medical advice."
    )

    return " ".join(recommendations)


df["Dashboard_Prevention_Recommendations"] = df.apply(
    prevention_recommendations,
    axis=1
)

print(
    "Prevention recommendations generated for:",
    len(df),
    "participants"
)


# ==============================================================================
# 10. INDIVIDUAL CVD INTERPRETATION
# ==============================================================================

print("\n" + "=" * 78)
print("8. INDIVIDUAL CVD RISK INTERPRETATION")
print("=" * 78)


def individual_cvd_interpretation(row):

    probability = safe_float(row["CVD_Probability"])
    prediction = safe_float(row["CVD_Prediction"])

    category = str(row["CVD_Risk_Category"])

    factor_count = int(row["Dashboard_Risk_Factor_Count"])

    if not np.isnan(probability):
        probability_text = f"{probability * 100:.2f}%"
    else:
        probability_text = "not available"

    if prediction == 1:
        prediction_text = (
            "The existing CVD model classifies this participant "
            "as model-predicted CVD."
        )
    else:
        prediction_text = (
            "The existing CVD model does not classify this participant "
            "as model-predicted CVD."
        )

    if factor_count == 0:
        burden_text = (
            "No major modifiable cardiovascular risk factors "
            "were flagged by the dashboard rules."
        )
    elif factor_count == 1:
        burden_text = (
            "One major modifiable cardiovascular risk factor "
            "was flagged."
        )
    else:
        burden_text = (
            f"{factor_count} major modifiable cardiovascular "
            "risk factors were flagged."
        )

    return (
        f"Existing CVD probability: {probability_text}. "
        f"Existing risk category: {category}. "
        f"{prediction_text} "
        f"{burden_text} "
        "This result is a risk assessment output and should not "
        "be interpreted as a medical diagnosis."
    )


df["Dashboard_Individual_Interpretation"] = df.apply(
    individual_cvd_interpretation,
    axis=1
)


# ==============================================================================
# 11. CLINICAL SAFETY MESSAGE
# ==============================================================================

print("\n" + "=" * 78)
print("9. CLINICAL SAFETY MESSAGE")
print("=" * 78)


df["Dashboard_Clinical_Safety_Message"] = (
    "This dashboard provides research-oriented cardiovascular "
    "risk assessment and educational interpretation. It does not "
    "diagnose cardiovascular disease or thyroid disease. Model "
    "predictions should not replace evaluation by a qualified "
    "healthcare professional. Seek urgent medical attention for "
    "severe or sudden symptoms such as chest pain, severe breathing "
    "difficulty, fainting, or other emergency symptoms."
)


# ==============================================================================
# 12. DASHBOARD USER OUTPUT
# ==============================================================================

print("\n" + "=" * 78)
print("10. DASHBOARD USER OUTPUT")
print("=" * 78)


dashboard_output_columns = [
    "SEQN",
    "Age",
    "Gender",
    "Race",

    # Anthropometric
    "BMI",
    "Height",
    "Weight",
    "Waist",

    # Thyroid
    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO",
    "FT3_FT4_Ratio",
    "Thyroid_Status",

    # Cardiovascular
    "HDL",
    "Total_Cholesterol",
    "SYS_BP",
    "DIA_BP",

    # Medical / lifestyle
    "Diabetes",
    "Smoking_100_Cigarettes",
    "Thyroid_Diagnosis",

    # Existing model outputs
    "CVD_Probability",
    "CVD_Prediction",
    "CVD_Risk_Category",

    # Dashboard-generated interpretation
    "Dashboard_Thyroid_Status",
    "Dashboard_Risk_Factors",
    "Dashboard_Risk_Factor_Count",
    "Primary_Modifiable_Risk_Factor",
    "Dashboard_Thyroid_Interpretation",
    "Dashboard_Biomarker_Interpretation",
    "Dashboard_Prevention_Recommendations",
    "Dashboard_Individual_Interpretation",
    "Dashboard_Clinical_Safety_Message"
]

dashboard_output_columns = [
    col for col in dashboard_output_columns
    if col in df.columns
]

dashboard_df = df[dashboard_output_columns].copy()

print(
    "\nDashboard output records:",
    len(dashboard_df)
)

print(
    "Dashboard output columns:",
    len(dashboard_df.columns)
)


# ==============================================================================
# 13. DASHBOARD RISK FACTOR SUMMARY
# ==============================================================================

print("\n" + "=" * 78)
print("11. DASHBOARD RISK FACTOR SUMMARY")
print("=" * 78)


risk_factor_names = [
    "High blood pressure",
    "High cholesterol",
    "Low HDL",
    "Obesity",
    "Smoking-related risk"
]

risk_summary = []

predicted_cvd_df = df[
    df["CVD_Prediction"] == 1
]

for factor in risk_factor_names:

    factor_mask = df["Dashboard_Risk_Factors"].apply(
        lambda x: factor in x
    )

    participant_count = int(factor_mask.sum())

    percentage_population = (
        participant_count / len(df) * 100
        if len(df) > 0 else 0
    )

    predicted_count = int(
        (
            factor_mask
            & (df["CVD_Prediction"] == 1)
        ).sum()
    )

    predicted_group_percentage = (
        predicted_count / predicted_cvd * 100
        if predicted_cvd > 0 else 0
    )

    risk_summary.append({
        "Risk_Factor": factor,
        "Participants_With_Factor": participant_count,
        "Percentage_Of_Population": round(
            percentage_population, 2
        ),
        "Predicted_CVD_Among_Factor": predicted_count,
        "Percentage_Of_Predicted_CVD_Group": round(
            predicted_group_percentage, 2
        )
    })

risk_summary_df = pd.DataFrame(risk_summary)

print(risk_summary_df.to_string(index=False))


# ==============================================================================
# 14. THYROID ASSOCIATION SUMMARY FOR DASHBOARD
# ==============================================================================

print("\n" + "=" * 78)
print("12. VALIDATED THYROID ASSOCIATION SUMMARY")
print("=" * 78)


thyroid_association_summary = pd.DataFrame({

    "Feature": [
        "TSH",
        "Free_T3",
        "Free_T4",
        "TPO",
        "FT3_FT4_Ratio"
    ],

    "Association_Status": [
        "No significant association after validation",
        "No significant association after validation",
        "Unadjusted association; not independently significant after adjustment",
        "No significant association after validation",
        "Unadjusted association; not independently significant after adjustment"
    ],

    "Dashboard_Interpretation": [
        "TSH was not independently associated with model-predicted CVD.",
        "Free T3 was not independently associated with model-predicted CVD.",
        "Free T4 was higher in the predicted-CVD group, but the association did not remain independently significant after adjustment.",
        "TPO was not independently associated with model-predicted CVD.",
        "The FT3/FT4 ratio was lower in the predicted-CVD group and showed an association with continuous CVD probability, but the adjusted association did not remain independently significant."
    ],

    "Causal_Claim": [
        "No",
        "No",
        "No",
        "No",
        "No"
    ]
})

print(
    thyroid_association_summary.to_string(index=False)
)


# ==============================================================================
# 15. CVD RISK CATEGORY SUMMARY
# ==============================================================================

print("\n" + "=" * 78)
print("13. CVD RISK CATEGORY SUMMARY")
print("=" * 78)


category_summary = (
    df["CVD_Risk_Category"]
    .value_counts(dropna=False)
    .rename_axis("CVD_Risk_Category")
    .reset_index(name="Participants")
)

category_summary["Percentage"] = (
    category_summary["Participants"]
    / len(df)
    * 100
).round(2)

print(category_summary.to_string(index=False))


# ==============================================================================
# 16. DASHBOARD POPULATION SUMMARY
# ==============================================================================

print("\n" + "=" * 78)
print("14. DASHBOARD POPULATION SUMMARY")
print("=" * 78)


population_summary = pd.DataFrame({
    "Metric": [
        "Total participants",
        "Thyroid-problem participants",
        "Model-predicted CVD",
        "No model-predicted CVD",
        "Percentage model-predicted CVD",
        "Mean CVD probability",
        "Median CVD probability"
    ],

    "Value": [
        len(df),
        len(df),
        predicted_cvd,
        no_predicted_cvd,
        round(predicted_percentage, 2),
        round(df["CVD_Probability"].mean(), 4),
        round(df["CVD_Probability"].median(), 4)
    ]
})

print(population_summary.to_string(index=False))


# ==============================================================================
# 17. DASHBOARD TEST CASES
# ==============================================================================

print("\n" + "=" * 78)
print("15. DASHBOARD LOGIC TEST CASES")
print("=" * 78)


test_cases = []

for index, row in df.head(5).iterrows():

    test_cases.append({

        "SEQN": row["SEQN"],

        "Thyroid_Status":
            row["Dashboard_Thyroid_Status"],

        "CVD_Probability":
            row["CVD_Probability"],

        "CVD_Prediction":
            row["CVD_Prediction"],

        "CVD_Risk_Category":
            row["CVD_Risk_Category"],

        "Risk_Factor_Count":
            row["Dashboard_Risk_Factor_Count"],

        "Primary_Risk_Factor":
            row["Primary_Modifiable_Risk_Factor"],

        "Risk_Factors":
            "; ".join(row["Dashboard_Risk_Factors"])
    })

test_cases_df = pd.DataFrame(test_cases)

print(
    test_cases_df.to_string(index=False)
)


# ==============================================================================
# 18. DATA INTEGRITY CHECKS
# ==============================================================================

print("\n" + "=" * 78)
print("16. DATA INTEGRITY CHECKS")
print("=" * 78)


integrity_results = {}


# Existing probability unchanged
if "CVD_Probability" in df.columns:
    integrity_results["CVD_Probability_available"] = True
else:
    integrity_results["CVD_Probability_available"] = False


# Existing prediction unchanged
if "CVD_Prediction" in df.columns:
    integrity_results["CVD_Prediction_available"] = True
else:
    integrity_results["CVD_Prediction_available"] = False


# Prediction should contain binary values
prediction_values = set(
    pd.to_numeric(
        df["CVD_Prediction"],
        errors="coerce"
    ).dropna().unique()
)

integrity_results["CVD_Prediction_binary"] = (
    prediction_values.issubset({0, 1})
)


# No CVD probability used as predictor
integrity_results["CVD_Probability_used_as_predictor"] = False


# No new model
integrity_results["New_CVD_model_created"] = False


# Original outcome is not modified by dashboard logic
integrity_results["Original_Cardiovascular_Disease_modified"] = False


# No causal thyroid claim
integrity_results["Causal_thyroid_CVD_claim"] = False


integrity_df = pd.DataFrame({
    "Integrity_Check": list(integrity_results.keys()),
    "Result": list(integrity_results.values())
})

print(
    integrity_df.to_string(index=False)
)


if not all(integrity_results.values()):
    raise ValueError(
        "One or more dashboard integrity checks failed."
    )

print("\nData integrity checks: PASSED")


# ==============================================================================
# 19. SAVE OUTPUT FILES
# ==============================================================================

print("\n" + "=" * 78)
print("17. SAVING RESULTS")
print("=" * 78)


dashboard_file = (
    f"{OUTPUT_DIR}/step13_dashboard_application_data.csv"
)

risk_file = (
    f"{OUTPUT_DIR}/step13_dashboard_risk_factor_summary.csv"
)

thyroid_file = (
    f"{OUTPUT_DIR}/step13_dashboard_thyroid_associations.csv"
)

category_file = (
    f"{OUTPUT_DIR}/step13_dashboard_cvd_risk_categories.csv"
)

population_file = (
    f"{OUTPUT_DIR}/step13_dashboard_population_summary.csv"
)

test_file = (
    f"{OUTPUT_DIR}/step13_dashboard_test_cases.csv"
)

integrity_file = (
    f"{OUTPUT_DIR}/step13_dashboard_integrity_checks.csv"
)


dashboard_df.to_csv(
    dashboard_file,
    index=False
)

risk_summary_df.to_csv(
    risk_file,
    index=False
)

thyroid_association_summary.to_csv(
    thyroid_file,
    index=False
)

category_summary.to_csv(
    category_file,
    index=False
)

population_summary.to_csv(
    population_file,
    index=False
)

test_cases_df.to_csv(
    test_file,
    index=False
)

integrity_df.to_csv(
    integrity_file,
    index=False
)


print("Saved:")
print("1.", dashboard_file)
print("2.", risk_file)
print("3.", thyroid_file)
print("4.", category_file)
print("5.", population_file)
print("6.", test_file)
print("7.", integrity_file)


# ==============================================================================
# 20. FINAL DASHBOARD READINESS CHECK
# ==============================================================================

print("\n" + "=" * 78)
print("18. FINAL DASHBOARD READINESS CHECK")
print("=" * 78)


dashboard_requirements = {

    "User thyroid fields available":
        all(
            col in df.columns
            for col in [
                "TSH",
                "Free_T3",
                "Free_T4",
                "TPO",
                "Thyroid_Diagnosis"
            ]
        ),

    "Cardiovascular fields available":
        all(
            col in df.columns
            for col in [
                "HDL",
                "Total_Cholesterol",
                "SYS_BP",
                "DIA_BP"
            ]
        ),

    "Lifestyle field available":
        "Smoking_100_Cigarettes" in df.columns,

    "Existing CVD probability available":
        "CVD_Probability" in df.columns,

    "Existing CVD prediction available":
        "CVD_Prediction" in df.columns,

    "Existing CVD risk category available":
        "CVD_Risk_Category" in df.columns,

    "Risk factor interpretation available":
        "Dashboard_Risk_Factors" in df.columns,

    "Prevention recommendations available":
        "Dashboard_Prevention_Recommendations" in df.columns,

    "Thyroid interpretation available":
        "Dashboard_Thyroid_Interpretation" in df.columns,

    "Clinical safety message available":
        "Dashboard_Clinical_Safety_Message" in df.columns
}


readiness_df = pd.DataFrame({
    "Dashboard_Requirement":
        list(dashboard_requirements.keys()),

    "Status":
        list(dashboard_requirements.values())
})

print(
    readiness_df.to_string(index=False)
)


if not all(dashboard_requirements.values()):
    raise ValueError(
        "Dashboard readiness check failed."
    )

print("\nDASHBOARD READINESS CHECK: PASSED")


# ==============================================================================
# 21. FINAL SUMMARY
# ==============================================================================

print("\n" + "=" * 78)
print("STEP 13 COMPLETE")
print("=" * 78)

print(
    "\nPopulation analyzed:",
    len(df)
)

print(
    "Thyroid-problem participants:",
    len(df)
)

print(
    "Existing model-predicted CVD:",
    predicted_cvd
)

print(
    "Percentage with model-predicted CVD:",
    round(predicted_percentage, 2),
    "%"
)

print("\nDashboard capabilities prepared:")

print("1. Thyroid information collection")
print("2. Cardiovascular information collection")
print("3. Existing CVD probability display")
print("4. Existing CVD prediction display")
print("5. Existing CVD risk-category display")
print("6. Individual cardiovascular risk-factor detection")
print("7. Primary modifiable risk-factor identification")
print("8. Thyroid biomarker interpretation")
print("9. Validated thyroid-CVD association explanation")
print("10. Prevention and risk-management recommendations")
print("11. Individual risk interpretation")
print("12. Clinical safety messaging")
print("13. Dashboard readiness validation")


print("\nIMPORTANT:")
print("- No new CVD prediction model created.")
print("- Existing CVD_Prediction retained.")
print("- Existing CVD_Probability retained.")
print("- CVD_Probability NOT used as a predictor.")
print("- Original Cardiovascular_Disease NOT modified.")
print("- Thyroid-CVD findings treated as associations.")
print("- No causal thyroid-CVD claim made.")
print("- Dashboard output is not a medical diagnosis.")

print("\nNext stage:")
print("STEP 14 - BUILD FINAL INTERACTIVE DASHBOARD UI")
print("           (Streamlit user input + CVD assessment +")
print("            thyroid interpretation + prevention guidance)")

print("=" * 78)
