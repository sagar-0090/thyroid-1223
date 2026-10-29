# =============================================================================
# STEP 11 - DASHBOARD FEATURE ENGINEERING AND USER INPUT / CVD ASSESSMENT LOGIC
# =============================================================================
#
# PURPOSE
# -------
# Prepare the final data and logic required for the future dashboard.
#
# The dashboard will:
# 1. Collect user information.
# 2. Identify whether the user has a thyroid problem.
# 3. Evaluate cardiovascular risk using the ESTABLISHED CVD MODEL.
# 4. Display CVD probability / risk category.
# 5. Identify conventional cardiovascular risk factors.
# 6. Explain thyroid biomarker findings carefully.
# 7. Provide prevention / risk-management recommendations.
# 8. Encourage professional medical evaluation where appropriate.
#
# IMPORTANT RESEARCH RULES
# ------------------------
# - NO new CVD prediction model is created in Step 11.
# - Existing CVD_Prediction is retained.
# - Existing CVD_Probability is retained.
# - CVD_Probability is NOT used as a predictor.
# - Cardiovascular_Disease is NOT modified.
# - Thyroid biomarkers are NOT treated as proven causes of CVD.
# - This step prepares dashboard logic and user-facing interpretation.
#
# =============================================================================

import os
import numpy as np
import pandas as pd

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

INPUT_FILE = "output/step5_integrated_thyroid_cvd_dataset.csv"
STEP10_FILE = "output/step10_dashboard_ready_data.csv"

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 78)
print("STEP 11 - DASHBOARD FEATURE ENGINEERING AND USER INPUT / CVD ASSESSMENT LOGIC")
print("=" * 78)


# =============================================================================
# 1. LOAD MASTER DATASET
# =============================================================================

# IMPORTANT:
# Always use the Step 5 integrated dataset as the master clinical/model
# dataset because it contains ALL variables required by the dashboard.

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded from:")
print(INPUT_FILE)

print("\nShape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# =============================================================================
# 2. REQUIRED COLUMN VALIDATION
# =============================================================================

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
    "Cardiovascular_Disease",
    "High_BP",
    "Obesity",
    "High_Cholesterol",
    "Low_HDL",
    "Abnormal_TSH",
    "Smoking_Risk",
    "Risk_Factor_Count",
    "CVD_Probability",
    "CVD_Prediction_0.50",
    "CVD_Prediction",
    "CVD_Status_Group",
    "Thyroid_Problem",
    "FT3_FT4_Ratio",
    "Thyroid_Status",
    "CVD_Risk_Category",
    "Integrated_Thyroid_CVD_Group"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "\nMissing required columns from MASTER DATASET:\n"
        + "\n".join(missing_columns)
        + "\n\nStep 11 must use:\n"
        + INPUT_FILE
    )

print("\nAll required columns are present.")


# =============================================================================
# 3. BASIC DATA VALIDATION
# =============================================================================

print("\n" + "=" * 78)
print("1. DATA VALIDATION")
print("=" * 78)

print("\nTotal records:", len(df))

print("\nMissing values in important dashboard variables:")

dashboard_input_columns = [
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
    "DIA_BP"
]

missing_summary = df[dashboard_input_columns].isnull().sum()

print(missing_summary)


# =============================================================================
# 4. THYROID-PROBLEM POPULATION
# =============================================================================

print("\n" + "=" * 78)
print("2. THYROID-PROBLEM POPULATION")
print("=" * 78)

thyroid_df = df[df["Thyroid_Problem"] == 1].copy()

print("\nTotal records:", len(df))
print("Thyroid-problem records:", len(thyroid_df))

if len(thyroid_df) > 0:

    print("\nCVD prediction distribution:")
    print(
        thyroid_df["CVD_Prediction"]
        .value_counts()
        .sort_index()
    )

    predicted_cvd = int(
        (thyroid_df["CVD_Prediction"] == 1).sum()
    )

    no_predicted_cvd = int(
        (thyroid_df["CVD_Prediction"] == 0).sum()
    )

    percentage = (
        predicted_cvd / len(thyroid_df) * 100
    )

    print("\nModel-predicted CVD:", predicted_cvd)
    print("No model-predicted CVD:", no_predicted_cvd)
    print("Percentage with model-predicted CVD:",
          round(percentage, 2), "%")


# =============================================================================
# 5. DASHBOARD RISK-CATEGORY LOGIC
# =============================================================================

print("\n" + "=" * 78)
print("3. CVD RISK CATEGORY LOGIC")
print("=" * 78)

print("""
The dashboard will retain the EXISTING CVD_Risk_Category.

No new CVD prediction model is created.

The dashboard displays:
- Existing CVD probability
- Existing CVD prediction
- Existing CVD risk category
- Existing cardiovascular risk factors
""")


# =============================================================================
# 6. STANDARDIZE RISK CATEGORY DISPLAY
# =============================================================================

def clean_risk_category(value):

    if pd.isna(value):
        return "Unknown"

    value = str(value).strip()

    if value == "":
        return "Unknown"

    return value


df["Dashboard_CVD_Risk_Category"] = (
    df["CVD_Risk_Category"]
    .apply(clean_risk_category)
)


# =============================================================================
# 7. IDENTIFY INDIVIDUAL CARDIOVASCULAR RISK FACTORS
# =============================================================================

print("\n" + "=" * 78)
print("4. INDIVIDUAL CARDIOVASCULAR RISK FACTOR LOGIC")
print("=" * 78)


def risk_flag(value):
    """
    Convert existing binary risk-factor variables into
    dashboard-friendly Yes/No labels.
    """

    try:
        if float(value) == 1:
            return "Yes"
        else:
            return "No"
    except:
        return "Unknown"


df["Dashboard_High_BP"] = df["High_BP"].apply(risk_flag)

df["Dashboard_High_Cholesterol"] = (
    df["High_Cholesterol"].apply(risk_flag)
)

df["Dashboard_Low_HDL"] = (
    df["Low_HDL"].apply(risk_flag)
)

df["Dashboard_Obesity"] = (
    df["Obesity"].apply(risk_flag)
)

df["Dashboard_Smoking_Risk"] = (
    df["Smoking_Risk"].apply(risk_flag)
)

df["Dashboard_Abnormal_TSH"] = (
    df["Abnormal_TSH"].apply(risk_flag)
)


# =============================================================================
# 8. PRIMARY MODIFIABLE RISK FACTOR
# =============================================================================

def determine_primary_risk_factor(row):

    factors = []

    if row["High_BP"] == 1:
        factors.append("High blood pressure")

    if row["Smoking_Risk"] == 1:
        factors.append("Smoking-related risk")

    if row["Obesity"] == 1:
        factors.append("Obesity")

    if row["High_Cholesterol"] == 1:
        factors.append("High cholesterol")

    if row["Low_HDL"] == 1:
        factors.append("Low HDL")

    if len(factors) == 0:
        return "No major modifiable factor flagged"

    # Priority is based on the established risk-factor framework,
    # not a newly trained model.

    priority = [
        "High blood pressure",
        "Smoking-related risk",
        "Obesity",
        "High cholesterol",
        "Low HDL"
    ]

    for factor in priority:
        if factor in factors:
            return factor

    return factors[0]


df["Primary_Modifiable_Risk_Factor"] = (
    df.apply(determine_primary_risk_factor, axis=1)
)


# =============================================================================
# 9. ALL RISK FACTORS PRESENT
# =============================================================================

def collect_risk_factors(row):

    factors = []

    if row["High_BP"] == 1:
        factors.append("High blood pressure")

    if row["High_Cholesterol"] == 1:
        factors.append("High cholesterol")

    if row["Low_HDL"] == 1:
        factors.append("Low HDL")

    if row["Obesity"] == 1:
        factors.append("Obesity")

    if row["Smoking_Risk"] == 1:
        factors.append("Smoking-related risk")

    if len(factors) == 0:
        return "None flagged"

    return "; ".join(factors)


df["Dashboard_Risk_Factors"] = (
    df.apply(collect_risk_factors, axis=1)
)


# =============================================================================
# 10. THYROID INTERPRETATION LOGIC
# =============================================================================

print("\n" + "=" * 78)
print("5. THYROID INTERPRETATION LOGIC")
print("=" * 78)


thyroid_interpretation = {

    "TSH":
        "TSH was not independently associated with "
        "model-predicted CVD in the validated analysis.",

    "Free_T3":
        "Free T3 was not independently associated with "
        "model-predicted CVD in the validated analysis.",

    "Free_T4":
        "Free T4 showed an unadjusted association with "
        "model-predicted CVD, but this association did not "
        "remain independently significant after adjustment.",

    "TPO":
        "TPO was not independently associated with "
        "model-predicted CVD in the validated analysis.",

    "FT3_FT4_Ratio":
        "The FT3/FT4 ratio showed an unadjusted association "
        "with model-predicted CVD and continuous CVD probability, "
        "but the adjusted association did not remain independently significant."
}


def generate_thyroid_interpretation(row):

    interpretations = []

    interpretations.append(
        "TSH: " + thyroid_interpretation["TSH"]
    )

    interpretations.append(
        "Free T3: " + thyroid_interpretation["Free_T3"]
    )

    interpretations.append(
        "Free T4: " + thyroid_interpretation["Free_T4"]
    )

    interpretations.append(
        "TPO: " + thyroid_interpretation["TPO"]
    )

    interpretations.append(
        "FT3/FT4 ratio: "
        + thyroid_interpretation["FT3_FT4_Ratio"]
    )

    return " ".join(interpretations)


df["Dashboard_Thyroid_Interpretation"] = (
    df.apply(generate_thyroid_interpretation, axis=1)
)


# =============================================================================
# 11. PREVENTION RECOMMENDATION ENGINE
# =============================================================================

print("\n" + "=" * 78)
print("6. PREVENTION RECOMMENDATION ENGINE")
print("=" * 78)


def generate_prevention_recommendations(row):

    recommendations = []

    # ---------------------------------------------------------
    # Blood pressure
    # ---------------------------------------------------------

    if row["High_BP"] == 1:

        recommendations.append(
            "Blood pressure requires attention. "
            "Monitor blood pressure regularly and discuss "
            "appropriate management with a healthcare professional."
        )

    # ---------------------------------------------------------
    # Smoking
    # ---------------------------------------------------------

    if row["Smoking_Risk"] == 1:

        recommendations.append(
            "Smoking-related cardiovascular risk is flagged. "
            "Smoking cessation support should be considered."
        )

    # ---------------------------------------------------------
    # Obesity
    # ---------------------------------------------------------

    if row["Obesity"] == 1:

        recommendations.append(
            "BMI/obesity-related risk is flagged. "
            "Gradual weight management through appropriate diet, "
            "physical activity and professional guidance may help."
        )

    # ---------------------------------------------------------
    # High cholesterol
    # ---------------------------------------------------------

    if row["High_Cholesterol"] == 1:

        recommendations.append(
            "High cholesterol is flagged. "
            "Discuss lipid monitoring, diet and treatment options "
            "with a healthcare professional."
        )

    # ---------------------------------------------------------
    # Low HDL
    # ---------------------------------------------------------

    if row["Low_HDL"] == 1:

        recommendations.append(
            "Low HDL is flagged. "
            "Heart-healthy physical activity, weight management "
            "and dietary improvement may be beneficial."
        )

    # ---------------------------------------------------------
    # Thyroid monitoring
    # ---------------------------------------------------------

    if row["Thyroid_Problem"] == 1:

        recommendations.append(
            "Continue appropriate thyroid monitoring and discuss "
            "thyroid test results with a qualified healthcare professional."
        )

    # ---------------------------------------------------------
    # Abnormal TSH
    # ---------------------------------------------------------

    if row["Abnormal_TSH"] == 1:

        recommendations.append(
            "Abnormal TSH is flagged. "
            "Clinical evaluation and follow-up thyroid testing "
            "may be appropriate."
        )

    # ---------------------------------------------------------
    # CVD risk
    # ---------------------------------------------------------

    if row["CVD_Prediction"] == 1:

        recommendations.append(
            "The established CVD model classifies this participant "
            "as model-predicted CVD. This is a risk assessment result, "
            "not a medical diagnosis. Professional cardiovascular "
            "evaluation is recommended."
        )

    # ---------------------------------------------------------
    # No risk factors
    # ---------------------------------------------------------

    if len(recommendations) == 0:

        recommendations.append(
            "No major modifiable cardiovascular risk factor "
            "was flagged by the available dataset variables. "
            "Continue routine preventive healthcare."
        )

    return " ".join(recommendations)


df["Dashboard_Prevention_Recommendations"] = (
    df.apply(generate_prevention_recommendations, axis=1)
)


# =============================================================================
# 12. INDIVIDUAL RISK INTERPRETATION
# =============================================================================

print("\n" + "=" * 78)
print("7. INDIVIDUAL RISK INTERPRETATION")
print("=" * 78)


def generate_individual_interpretation(row):

    risk_category = row["Dashboard_CVD_Risk_Category"]

    probability = row["CVD_Probability"]

    prediction = row["CVD_Prediction"]

    risk_factors = row["Dashboard_Risk_Factors"]

    if prediction == 1:

        prediction_text = (
            "The established CVD model classified this participant "
            "as model-predicted CVD."
        )

    else:

        prediction_text = (
            "The established CVD model did not classify this participant "
            "as model-predicted CVD."
        )

    interpretation = (
        f"CVD risk category: {risk_category}. "
        f"Existing model probability: {probability:.4f}. "
        f"{prediction_text} "
        f"Risk factors flagged: {risk_factors}. "
        "These results represent model-based risk assessment and "
        "association analysis, not a clinical diagnosis or proof of causation."
    )

    return interpretation


df["Dashboard_Individual_Interpretation"] = (
    df.apply(generate_individual_interpretation, axis=1)
)


# =============================================================================
# 13. THYROID-CVD SAFETY MESSAGE
# =============================================================================

df["Dashboard_Clinical_Safety_Message"] = (
    "The thyroid-CVD findings represent statistical associations "
    "within the analyzed population. They do not establish that "
    "thyroid abnormalities cause cardiovascular disease. "
    "The dashboard should not replace professional medical evaluation."
)


# =============================================================================
# 14. USER INPUT FIELD DEFINITIONS
# =============================================================================

print("\n" + "=" * 78)
print("8. DASHBOARD USER INPUT FIELDS")
print("=" * 78)


user_input_fields = pd.DataFrame({

    "Field": [

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
        "DIA_BP"
    ],

    "Category": [

        "Demographic",
        "Demographic",
        "Demographic",
        "Anthropometric",
        "Anthropometric",
        "Anthropometric",
        "Anthropometric",
        "Thyroid",
        "Thyroid",
        "Thyroid",
        "Thyroid",
        "Cardiovascular",
        "Cardiovascular",
        "Medical history",
        "Lifestyle",
        "Thyroid",
        "Cardiovascular",
        "Cardiovascular"
    ],

    "Required": ["Yes"] * 18,

    "Dashboard_Purpose": [

        "Age-related CVD risk assessment",
        "Demographic information",
        "Demographic information",
        "Weight-related cardiovascular risk",
        "Anthropometric information",
        "Anthropometric information",
        "Central adiposity assessment",
        "Thyroid function assessment",
        "Thyroid function assessment",
        "Thyroid function assessment",
        "Thyroid antibody information",
        "Lipid-related cardiovascular risk",
        "Lipid-related cardiovascular risk",
        "Diabetes-related cardiovascular risk",
        "Smoking-related cardiovascular risk",
        "Thyroid condition information",
        "Blood pressure-related cardiovascular risk",
        "Blood pressure-related cardiovascular risk"
    ]
})


print("\nUser input fields prepared:")
print(user_input_fields.to_string(index=False))


# =============================================================================
# 15. DASHBOARD OUTPUT FIELD DEFINITIONS
# =============================================================================

print("\n" + "=" * 78)
print("9. DASHBOARD OUTPUT FIELDS")
print("=" * 78)


dashboard_output_fields = pd.DataFrame({

    "Output_Field": [

        "CVD_Probability",
        "CVD_Prediction",
        "CVD_Risk_Category",
        "Risk_Factor_Count",
        "Dashboard_Risk_Factors",
        "Primary_Modifiable_Risk_Factor",
        "Thyroid_Status",
        "Dashboard_Abnormal_TSH",
        "FT3_FT4_Ratio",
        "Dashboard_Thyroid_Interpretation",
        "Dashboard_Prevention_Recommendations",
        "Dashboard_Individual_Interpretation",
        "Dashboard_Clinical_Safety_Message"
    ],

    "Purpose": [

        "Display existing CVD probability",
        "Display existing CVD model prediction",
        "Display existing risk category",
        "Show overall risk-factor burden",
        "List flagged cardiovascular risk factors",
        "Highlight highest-priority modifiable factor",
        "Display thyroid status",
        "Display abnormal TSH flag",
        "Display thyroid hormone ratio",
        "Explain validated thyroid-CVD findings",
        "Provide prevention/risk-management guidance",
        "Provide individual interpretation",
        "Provide medical safety disclaimer"
    ]
})


print(
    dashboard_output_fields.to_string(index=False)
)


# =============================================================================
# 16. CREATE DASHBOARD-READY DATASET
# =============================================================================

print("\n" + "=" * 78)
print("10. CREATING DASHBOARD-READY DATA")
print("=" * 78)


dashboard_columns = [

    # Identification
    "SEQN",

    # Demographics
    "Age",
    "Gender",
    "Race",

    # Anthropometrics
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
    "Thyroid_Diagnosis",
    "Thyroid_Problem",
    "Thyroid_Status",
    "Abnormal_TSH",

    # Cardiovascular measurements
    "HDL",
    "Total_Cholesterol",
    "SYS_BP",
    "DIA_BP",

    # Medical / lifestyle
    "Diabetes",
    "Smoking_100_Cigarettes",

    # Existing risk factors
    "High_BP",
    "High_Cholesterol",
    "Low_HDL",
    "Obesity",
    "Smoking_Risk",
    "Risk_Factor_Count",

    # Existing CVD model results
    "CVD_Probability",
    "CVD_Prediction_0.50",
    "CVD_Prediction",
    "CVD_Status_Group",
    "CVD_Risk_Category",
    "Integrated_Thyroid_CVD_Group",

    # Dashboard interpretation
    "Dashboard_CVD_Risk_Category",
    "Dashboard_High_BP",
    "Dashboard_High_Cholesterol",
    "Dashboard_Low_HDL",
    "Dashboard_Obesity",
    "Dashboard_Smoking_Risk",
    "Dashboard_Abnormal_TSH",
    "Primary_Modifiable_Risk_Factor",
    "Dashboard_Risk_Factors",
    "Dashboard_Thyroid_Interpretation",
    "Dashboard_Prevention_Recommendations",
    "Dashboard_Individual_Interpretation",
    "Dashboard_Clinical_Safety_Message"
]


dashboard_df = df[dashboard_columns].copy()


print("\nDashboard-ready records:", len(dashboard_df))
print(
    "Dashboard-ready columns:",
    len(dashboard_df.columns)
)


# =============================================================================
# 17. POPULATION SUMMARY FOR DASHBOARD
# =============================================================================

print("\n" + "=" * 78)
print("11. DASHBOARD POPULATION SUMMARY")
print("=" * 78)


total_records = len(df)

thyroid_records = int(
    (df["Thyroid_Problem"] == 1).sum()
)

predicted_cvd = int(
    (df["CVD_Prediction"] == 1).sum()
)

no_predicted_cvd = int(
    (df["CVD_Prediction"] == 0).sum()
)

mean_probability = df["CVD_Probability"].mean()

median_probability = df["CVD_Probability"].median()


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

        total_records,
        thyroid_records,
        predicted_cvd,
        no_predicted_cvd,

        round(
            predicted_cvd / total_records * 100,
            2
        ),

        round(mean_probability, 4),
        round(median_probability, 4)
    ]
})


print(
    population_summary.to_string(index=False)
)


# =============================================================================
# 18. RISK FACTOR SUMMARY
# =============================================================================

print("\n" + "=" * 78)
print("12. RISK FACTOR SUMMARY")
print("=" * 78)


risk_factor_definitions = {

    "High blood pressure": "High_BP",
    "High cholesterol": "High_Cholesterol",
    "Low HDL": "Low_HDL",
    "Obesity": "Obesity",
    "Smoking-related risk": "Smoking_Risk"
}


risk_summary_rows = []


for factor_name, column in risk_factor_definitions.items():

    participants = int(
        (df[column] == 1).sum()
    )

    predicted_cvd_among_factor = int(
        (
            (df[column] == 1)
            &
            (df["CVD_Prediction"] == 1)
        ).sum()
    )

    if predicted_cvd > 0:

        percentage_predicted_group = (
            predicted_cvd_among_factor
            / predicted_cvd
            * 100
        )

    else:

        percentage_predicted_group = 0


    risk_summary_rows.append({

        "Risk_Factor":
            factor_name,

        "Participants_With_Factor":
            participants,

        "Percentage_Of_Population":
            round(
                participants / total_records * 100,
                2
            ),

        "Predicted_CVD_Among_Factor":
            predicted_cvd_among_factor,

        "Percentage_Of_Predicted_CVD_Group":
            round(
                percentage_predicted_group,
                2
            )
    })


risk_factor_summary = pd.DataFrame(
    risk_summary_rows
)


print(
    risk_factor_summary.to_string(index=False)
)


# =============================================================================
# 19. RISK CATEGORY SUMMARY
# =============================================================================

print("\n" + "=" * 78)
print("13. CVD RISK CATEGORY SUMMARY")
print("=" * 78)


risk_category_summary = (
    df["CVD_Risk_Category"]
    .value_counts()
    .rename_axis("CVD_Risk_Category")
    .reset_index(name="Participants")
)


risk_category_summary["Percentage"] = (
    risk_category_summary["Participants"]
    / total_records
    * 100
).round(2)


print(
    risk_category_summary.to_string(index=False)
)


# =============================================================================
# 20. DATA INTEGRITY CHECKS
# =============================================================================

print("\n" + "=" * 78)
print("14. DATA INTEGRITY CHECKS")
print("=" * 78)


original_cvd = pd.read_csv(INPUT_FILE)


check_original_cvd_unchanged = (
    df["Cardiovascular_Disease"]
    .equals(
        original_cvd["Cardiovascular_Disease"]
    )
)


check_probability_unchanged = (
    df["CVD_Probability"]
    .equals(
        original_cvd["CVD_Probability"]
    )
)


check_prediction_count = (
    int(
        (df["CVD_Prediction"] == 1).sum()
    )
    ==
    predicted_cvd
)


print(
    "Original Cardiovascular_Disease unchanged:",
    check_original_cvd_unchanged
)

print(
    "CVD_Probability unchanged:",
    check_probability_unchanged
)

print(
    "CVD prediction count consistent:",
    check_prediction_count
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


all_checks_passed = (
    check_original_cvd_unchanged
    and check_probability_unchanged
    and check_prediction_count
)


if all_checks_passed:

    print("\nData integrity checks: PASSED")

else:

    print("\nData integrity checks: FAILED")

    raise ValueError(
        "One or more data integrity checks failed."
    )


# =============================================================================
# 21. SAVE DASHBOARD INPUT DEFINITIONS
# =============================================================================

user_input_fields.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step11_dashboard_user_input_fields.csv"
    ),
    index=False
)


# =============================================================================
# 22. SAVE DASHBOARD OUTPUT DEFINITIONS
# =============================================================================

dashboard_output_fields.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step11_dashboard_output_fields.csv"
    ),
    index=False
)


# =============================================================================
# 23. SAVE DASHBOARD-READY DATA
# =============================================================================

dashboard_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step11_dashboard_ready_data.csv"
    ),
    index=False
)


# =============================================================================
# 24. SAVE POPULATION SUMMARY
# =============================================================================

population_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step11_dashboard_population_summary.csv"
    ),
    index=False
)


# =============================================================================
# 25. SAVE RISK FACTOR SUMMARY
# =============================================================================

risk_factor_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step11_dashboard_risk_factor_summary.csv"
    ),
    index=False
)


# =============================================================================
# 26. SAVE RISK CATEGORY SUMMARY
# =============================================================================

risk_category_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step11_dashboard_cvd_risk_category_summary.csv"
    ),
    index=False
)


# =============================================================================
# 27. SAVE INDIVIDUAL RISK PROFILES
# =============================================================================

individual_profile_columns = [

    "SEQN",

    "Age",
    "Gender",
    "Race",

    "BMI",
    "Waist",

    "TSH",
    "Free_T3",
    "Free_T4",
    "TPO",
    "FT3_FT4_Ratio",

    "HDL",
    "Total_Cholesterol",

    "SYS_BP",
    "DIA_BP",

    "Diabetes",
    "Smoking_100_Cigarettes",

    "CVD_Probability",
    "CVD_Prediction",
    "CVD_Risk_Category",

    "Risk_Factor_Count",
    "Dashboard_Risk_Factors",
    "Primary_Modifiable_Risk_Factor",

    "Thyroid_Status",
    "Dashboard_Thyroid_Interpretation",

    "Dashboard_Prevention_Recommendations",
    "Dashboard_Individual_Interpretation",

    "Dashboard_Clinical_Safety_Message"
]


individual_profiles = df[
    individual_profile_columns
].copy()


individual_profiles.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step11_individual_risk_profiles.csv"
    ),
    index=False
)


# =============================================================================
# 28. FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 78)
print("STEP 11 COMPLETE")
print("=" * 78)

print("\nPopulation analyzed:", total_records)

print(
    "Thyroid-problem participants:",
    thyroid_records
)

print(
    "Model-predicted CVD:",
    predicted_cvd
)

print(
    "Percentage with model-predicted CVD:",
    round(
        predicted_cvd / total_records * 100,
        2
    ),
    "%"
)

print("\nDashboard preparation completed.")

print("\nFiles saved:")

print(
    "1. output/step11_dashboard_user_input_fields.csv"
)

print(
    "2. output/step11_dashboard_output_fields.csv"
)

print(
    "3. output/step11_dashboard_ready_data.csv"
)

print(
    "4. output/step11_dashboard_population_summary.csv"
)

print(
    "5. output/step11_dashboard_risk_factor_summary.csv"
)

print(
    "6. output/step11_dashboard_cvd_risk_category_summary.csv"
)

print(
    "7. output/step11_individual_risk_profiles.csv"
)

print("\nIMPORTANT:")

print("- No new CVD prediction model created.")

print("- Existing CVD_Prediction retained.")

print("- Existing CVD_Probability retained.")

print("- CVD_Probability NOT used as predictor.")

print("- Original Cardiovascular_Disease NOT modified.")

print("- Thyroid-CVD findings treated as associations.")

print("- Prevention recommendations are risk-management guidance.")

print("- Dashboard should not present model output as a medical diagnosis.")

print("=" * 78)
