# ==============================================================================
# STEP 12 - INTERACTIVE THYROID-CVD RISK ASSESSMENT DASHBOARD
# ==============================================================================
#
# Purpose:
#   Interactive dashboard for users with thyroid problems to:
#   1. Enter thyroid and cardiovascular information
#   2. Assess cardiovascular risk factors
#   3. Display existing model results for research records
#   4. Interpret thyroid biomarkers carefully
#   5. Identify modifiable CVD risk factors
#   6. Generate prevention/risk-management recommendations
#
# IMPORTANT RESEARCH RULES:
#   - No new CVD prediction model is trained here.
#   - Original Cardiovascular_Disease is never modified.
#   - Existing CVD_Prediction is retained for dataset records.
#   - Existing CVD_Probability is retained for dataset records.
#   - Thyroid-CVD findings are interpreted as associations, NOT causation.
#   - Dashboard output is NOT a medical diagnosis.
#
# ==============================================================================

import os
import numpy as np
import pandas as pd
import streamlit as st


# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="Thyroid-CVD Risk Assessment",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==============================================================================
# CONSTANTS
# ==============================================================================

DATA_PATH = "output/step11_dashboard_ready_data.csv"

REQUIRED_COLUMNS = [
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


# ==============================================================================
# LOAD DATA
# ==============================================================================

@st.cache_data
def load_dataset():

    if not os.path.exists(DATA_PATH):
        return None

    df = pd.read_csv(DATA_PATH)

    return df


df = load_dataset()


# ==============================================================================
# HEADER
# ==============================================================================

st.title("🦋 Thyroid–CVD Risk Assessment Dashboard")

st.markdown(
    """
    ### Integrated Thyroid and Cardiovascular Risk Assessment

    This dashboard is designed for individuals with thyroid problems.
    It combines thyroid information with conventional cardiovascular
    risk factors to provide an understandable risk assessment and
    prevention guidance.

    **This dashboard is a research/decision-support tool and is not a
    medical diagnosis.**
    """
)


# ==============================================================================
# DATASET VALIDATION
# ==============================================================================

if df is None:

    st.error(
        f"""
        Dataset not found.

        Expected file:

        `{DATA_PATH}`

        Please run Steps 5–11 first.
        """
    )

    st.stop()


missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in df.columns
]

if missing_columns:

    st.error("Required columns are missing from the dashboard dataset:")

    st.write(missing_columns)

    st.stop()


# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================

st.sidebar.title("Dashboard Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "🏠 Overview",
        "👤 User Risk Assessment",
        "🫀 CVD Risk Factors",
        "🦋 Thyroid Assessment",
        "📊 Research Findings",
        "💡 Prevention Guidance",
        "⚠️ Safety & Limitations"
    ]
)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def safe_float(value, default=0.0):

    try:
        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


# ------------------------------------------------------------------------------
# THYROID INTERPRETATION
# ------------------------------------------------------------------------------

def thyroid_interpretation(tsH, free_t3, free_t4, tpo, ratio):

    findings = []

    # TSH
    if tsH < 0.4:
        findings.append(
            "TSH is below the reference threshold used by this dashboard."
        )

    elif tsH > 4.0:
        findings.append(
            "TSH is above the reference threshold used by this dashboard."
        )

    else:
        findings.append(
            "TSH is within the dashboard reference range."
        )

    # Free T3
    if free_t3 < 2.3:
        findings.append(
            "Free T3 is relatively low and may require clinical interpretation."
        )

    elif free_t3 > 4.2:
        findings.append(
            "Free T3 is relatively high and may require clinical interpretation."
        )

    else:
        findings.append(
            "Free T3 is within the dashboard reference range."
        )

    # Free T4
    if free_t4 < 0.8:
        findings.append(
            "Free T4 is relatively low."
        )

    elif free_t4 > 1.8:
        findings.append(
            "Free T4 is relatively high."
        )

    else:
        findings.append(
            "Free T4 is within the dashboard reference range."
        )

    # TPO
    if tpo > 35:
        findings.append(
            "TPO is elevated according to the dashboard threshold."
        )
    else:
        findings.append(
            "TPO is not elevated according to the dashboard threshold."
        )

    # Ratio
    if ratio > 0:
        if ratio < 2.5:
            findings.append(
                "The FT3/FT4 ratio is relatively low."
            )

        elif ratio > 5.0:
            findings.append(
                "The FT3/FT4 ratio is relatively high."
            )

        else:
            findings.append(
                "The FT3/FT4 ratio is within the dashboard interpretation range."
            )

    return findings


# ------------------------------------------------------------------------------
# RISK FACTOR DETECTION
# ------------------------------------------------------------------------------

def identify_risk_factors(
    age,
    bmi,
    systolic_bp,
    diastolic_bp,
    hdl,
    total_cholesterol,
    diabetes,
    smoking
):

    factors = []

    # Blood pressure
    if systolic_bp >= 130 or diastolic_bp >= 80:
        factors.append("High blood pressure")

    # Obesity
    if bmi >= 30:
        factors.append("Obesity")

    # Cholesterol
    if total_cholesterol >= 200:
        factors.append("High cholesterol")

    # HDL
    if hdl < 40:
        factors.append("Low HDL")

    # Smoking
    if smoking in [1, 2, "1", "2", True]:
        factors.append("Smoking-related risk")

    # Diabetes
    if diabetes in [1, 2, 3, "1", "2", "3", True]:
        factors.append("Diabetes")

    # Age-related risk
    if age >= 65:
        factors.append("Age-related cardiovascular risk")

    return factors


# ------------------------------------------------------------------------------
# PREVENTION RECOMMENDATIONS
# ------------------------------------------------------------------------------

def generate_recommendations(
    factors,
    thyroid_findings,
    cvd_prediction=None,
    cvd_probability=None
):

    recommendations = []

    if "High blood pressure" in factors:
        recommendations.append(
            "Prioritize blood-pressure monitoring and discuss persistent "
            "elevation with a healthcare professional."
        )

    if "Obesity" in factors:
        recommendations.append(
            "Consider gradual, sustainable weight management through "
            "nutrition, physical activity, sleep and professional guidance."
        )

    if "High cholesterol" in factors:
        recommendations.append(
            "Review cholesterol levels with a healthcare professional and "
            "discuss diet, lifestyle and treatment options when appropriate."
        )

    if "Low HDL" in factors:
        recommendations.append(
            "Focus on regular physical activity, healthy dietary patterns "
            "and reduction of other cardiovascular risk factors."
        )

    if "Smoking-related risk" in factors:
        recommendations.append(
            "Smoking cessation is strongly recommended because smoking "
            "is an important modifiable cardiovascular risk factor."
        )

    if "Diabetes" in factors:
        recommendations.append(
            "Maintain appropriate blood-glucose management and discuss "
            "cardiovascular risk with a healthcare professional."
        )

    if "Age-related cardiovascular risk" in factors:
        recommendations.append(
            "Because cardiovascular risk generally increases with age, "
            "regular cardiovascular risk assessment is appropriate."
        )

    # Thyroid follow-up
    if thyroid_findings:
        recommendations.append(
            "Continue appropriate thyroid monitoring and discuss abnormal "
            "thyroid laboratory findings with a qualified healthcare professional."
        )

    if cvd_prediction == 1:
        recommendations.append(
            "The existing model classifies this record as model-predicted "
            "CVD risk. Professional cardiovascular evaluation is recommended."
        )

    if cvd_probability is not None:
        if cvd_probability >= 0.70:
            recommendations.append(
                "The existing model probability is high; prompt professional "
                "assessment is appropriate."
            )

    if len(recommendations) == 0:
        recommendations.append(
            "Continue routine preventive care, healthy lifestyle practices "
            "and appropriate thyroid and cardiovascular monitoring."
        )

    return recommendations


# ------------------------------------------------------------------------------
# INDIVIDUAL INTERPRETATION
# ------------------------------------------------------------------------------

def generate_individual_interpretation(
    factors,
    thyroid_problem,
    cvd_prediction=None,
    cvd_probability=None
):

    messages = []

    if thyroid_problem:
        messages.append(
            "The individual is being assessed within the thyroid-problem pathway."
        )

    else:
        messages.append(
            "The entered information does not currently indicate a thyroid "
            "problem according to the dashboard inputs."
        )

    if cvd_prediction == 1:
        messages.append(
            "The existing CVD model classifies this record as predicted CVD risk."
        )

    elif cvd_prediction == 0:
        messages.append(
            "The existing CVD model does not classify this record as predicted CVD."
        )

    if cvd_probability is not None:
        messages.append(
            f"The existing model CVD probability is approximately "
            f"{cvd_probability * 100:.1f}%."
        )

    if factors:

        messages.append(
            "The following cardiovascular risk factors require attention: "
            + ", ".join(factors)
            + "."
        )

    else:

        messages.append(
            "No major dashboard-defined modifiable cardiovascular risk "
            "factor was flagged from the entered information."
        )

    messages.append(
        "Thyroid biomarkers should be interpreted together with clinical "
        "history and cardiovascular risk factors. The research analysis "
        "does not establish that thyroid abnormalities cause CVD."
    )

    return " ".join(messages)


# ==============================================================================
# PAGE 1 - OVERVIEW
# ==============================================================================

if page == "🏠 Overview":

    st.header("Dashboard Overview")

    total = len(df)

    predicted = int(
        pd.to_numeric(
            df["CVD_Prediction"],
            errors="coerce"
        ).fillna(0).sum()
    )

    no_predicted = total - predicted

    mean_probability = safe_float(
        df["CVD_Probability"].mean()
    )

    median_probability = safe_float(
        df["CVD_Probability"].median()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Participants",
        total
    )

    col2.metric(
        "Model-Predicted CVD",
        predicted
    )

    col3.metric(
        "No Model-Predicted CVD",
        no_predicted
    )

    col4.metric(
        "Mean CVD Probability",
        f"{mean_probability * 100:.1f}%"
    )

    st.divider()

    st.subheader("Population Summary")

    st.write(
        f"""
        The research dataset contains **{total} participants** identified
        as having thyroid problems.

        **{predicted} ({predicted / total * 100:.2f}%)** are classified by
        the existing CVD model as having model-predicted CVD.

        The mean existing CVD probability is
        **{mean_probability * 100:.2f}%**, while the median is
        **{median_probability * 100:.2f}%**.
        """
    )

    st.info(
        """
        The dashboard does not create a new CVD prediction model.
        It is designed to interpret cardiovascular risk within the
        thyroid-problem population.
        """
    )

    st.subheader("Main Risk Factors in the Research Population")

    risk_columns = {
        "High blood pressure": "High_BP",
        "High cholesterol": "High_Cholesterol",
        "Low HDL": "Low_HDL",
        "Obesity": "Obesity",
        "Smoking-related risk": "Smoking_Risk"
    }

    overview_data = []

    for name, column in risk_columns.items():

        if column in df.columns:

            count = int(
                pd.to_numeric(
                    df[column],
                    errors="coerce"
                ).fillna(0).sum()
            )

            percentage = count / total * 100

            overview_data.append(
                {
                    "Risk Factor": name,
                    "Participants": count,
                    "Percentage": round(percentage, 2)
                }
            )

    st.dataframe(
        pd.DataFrame(overview_data),
        use_container_width=True,
        hide_index=True
    )


# ==============================================================================
# PAGE 2 - USER RISK ASSESSMENT
# ==============================================================================

elif page == "👤 User Risk Assessment":

    st.header("👤 Individual Thyroid-CVD Risk Assessment")

    st.write(
        """
        Enter the user's information below. The dashboard will identify
        cardiovascular risk factors, calculate the thyroid hormone ratio,
        interpret thyroid-related findings and provide prevention guidance.
        """
    )

    st.warning(
        """
        User-input mode does NOT train or create a new CVD prediction model.
        Existing model probability/prediction can only be displayed when
        an existing research record is selected.
        """
    )

    st.subheader("1. Demographic Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=50
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female", "Other"]
        )

    with col3:
        race = st.text_input(
            "Race / Ethnicity",
            value=""
        )

    st.subheader("2. Anthropometric Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        height = st.number_input(
            "Height (cm)",
            min_value=50.0,
            max_value=250.0,
            value=165.0
        )

    with col2:
        weight = st.number_input(
            "Weight (kg)",
            min_value=10.0,
            max_value=300.0,
            value=70.0
        )

    calculated_bmi = weight / ((height / 100) ** 2)

    with col3:
        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=80.0,
            value=float(round(calculated_bmi, 2)),
            step=0.1
        )

    with col4:
        waist = st.number_input(
            "Waist circumference (cm)",
            min_value=30.0,
            max_value=200.0,
            value=85.0
        )

    st.subheader("3. Thyroid Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        tsh = st.number_input(
            "TSH",
            min_value=0.0,
            max_value=50.0,
            value=2.5,
            step=0.01
        )

    with col2:
        free_t3 = st.number_input(
            "Free T3",
            min_value=0.0,
            max_value=20.0,
            value=2.9,
            step=0.01
        )

    with col3:
        free_t4 = st.number_input(
            "Free T4",
            min_value=0.0,
            max_value=20.0,
            value=0.9,
            step=0.01
        )

    with col4:
        tpo = st.number_input(
            "TPO",
            min_value=0.0,
            max_value=2000.0,
            value=1.0,
            step=0.1
        )

    thyroid_diagnosis = st.selectbox(
        "Thyroid diagnosis",
        [
            "No known diagnosis",
            "Hypothyroidism",
            "Hyperthyroidism",
            "Autoimmune thyroid condition",
            "Other thyroid condition"
        ]
    )

    st.subheader("4. Cardiovascular Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        systolic_bp = st.number_input(
            "Systolic BP",
            min_value=50,
            max_value=300,
            value=120
        )

    with col2:
        diastolic_bp = st.number_input(
            "Diastolic BP",
            min_value=30,
            max_value=200,
            value=80
        )

    with col3:
        hdl = st.number_input(
            "HDL",
            min_value=1.0,
            max_value=200.0,
            value=55.0
        )

    with col4:
        total_cholesterol = st.number_input(
            "Total Cholesterol",
            min_value=20.0,
            max_value=600.0,
            value=200.0
        )

    st.subheader("5. Medical and Lifestyle Information")

    col1, col2 = st.columns(2)

    with col1:
        diabetes = st.selectbox(
            "Diabetes status",
            [
                "No",
                "Yes"
            ]
        )

    with col2:
        smoking = st.selectbox(
            "Smoking history",
            [
                "No",
                "Yes"
            ]
        )

    st.divider()

    assess_button = st.button(
        "🔍 Assess Thyroid-CVD Risk",
        type="primary",
        use_container_width=True
    )

    if assess_button:

        # ----------------------------------------------------------------------
        # Thyroid ratio
        # ----------------------------------------------------------------------

        if free_t4 > 0:

            ratio = free_t3 / free_t4

        else:

            ratio = np.nan

        # ----------------------------------------------------------------------
        # Thyroid problem
        # ----------------------------------------------------------------------

        thyroid_problem = (
            thyroid_diagnosis != "No known diagnosis"
            or tsh < 0.4
            or tsh > 4.0
            or tpo > 35
        )

        # ----------------------------------------------------------------------
        # Convert inputs
        # ----------------------------------------------------------------------

        diabetes_value = 1 if diabetes == "Yes" else 0
        smoking_value = 1 if smoking == "Yes" else 0

        # ----------------------------------------------------------------------
        # Risk factors
        # ----------------------------------------------------------------------

        factors = identify_risk_factors(
            age=age,
            bmi=bmi,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            hdl=hdl,
            total_cholesterol=total_cholesterol,
            diabetes=diabetes_value,
            smoking=smoking_value
        )

        # ----------------------------------------------------------------------
        # Thyroid findings
        # ----------------------------------------------------------------------

        thyroid_findings = thyroid_interpretation(
            tsh,
            free_t3,
            free_t4,
            tpo,
            ratio
        )

        # ----------------------------------------------------------------------
        # Results
        # ----------------------------------------------------------------------

        st.success("Assessment completed.")

        st.subheader("Assessment Summary")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Thyroid Problem",
            "Yes" if thyroid_problem else "No"
        )

        col2.metric(
            "Risk Factors",
            len(factors)
        )

        col3.metric(
            "FT3 / FT4 Ratio",
            f"{ratio:.2f}" if not np.isnan(ratio) else "N/A"
        )

        col4.metric(
            "Blood Pressure",
            f"{systolic_bp}/{diastolic_bp}"
        )

        # ----------------------------------------------------------------------
        # Risk factor results
        # ----------------------------------------------------------------------

        st.subheader("🫀 Cardiovascular Risk Factors")

        if factors:

            for factor in factors:

                st.warning(
                    f"⚠️ {factor}"
                )

        else:

            st.success(
                "No major dashboard-defined cardiovascular risk factor was flagged."
            )

        # ----------------------------------------------------------------------
        # Thyroid results
        # ----------------------------------------------------------------------

        st.subheader("🦋 Thyroid Assessment")

        for finding in thyroid_findings:

            st.write("• " + finding)

        # ----------------------------------------------------------------------
        # Research thyroid interpretation
        # ----------------------------------------------------------------------

        st.subheader("🔬 Thyroid-CVD Research Interpretation")

        st.info(
            """
            Based on the validated research analysis:

            • TSH was not independently associated with model-predicted CVD.

            • Free T3 was not independently associated with model-predicted CVD.

            • Free T4 showed an unadjusted association but did not remain
              independently significant after adjustment.

            • TPO was not independently associated with model-predicted CVD.

            • FT3/FT4 ratio showed an unadjusted association and an association
              with continuous CVD probability, but did not remain independently
              significant after adjustment.

            These findings represent associations within the study population
            and do not establish that thyroid abnormalities cause CVD.
            """
        )

        # ----------------------------------------------------------------------
        # Prevention
        # ----------------------------------------------------------------------

        recommendations = generate_recommendations(
            factors=factors,
            thyroid_findings=thyroid_findings
        )

        st.subheader("💡 Personalized Prevention Guidance")

        for recommendation in recommendations:

            st.write("✅ " + recommendation)

        # ----------------------------------------------------------------------
        # Individual interpretation
        # ----------------------------------------------------------------------

        interpretation = generate_individual_interpretation(
            factors=factors,
            thyroid_problem=thyroid_problem
        )

        st.subheader("📋 Individual Interpretation")

        st.write(interpretation)

        # ----------------------------------------------------------------------
        # Safety
        # ----------------------------------------------------------------------

        st.error(
            """
            IMPORTANT:

            This assessment is not a medical diagnosis. Abnormal blood
            pressure, cholesterol, glucose, thyroid tests, symptoms or other
            concerning findings should be evaluated by a qualified healthcare
            professional.
            """
        )


# ==============================================================================
# PAGE 3 - CVD RISK FACTORS
# ==============================================================================

elif page == "🫀 CVD Risk Factors":

    st.header("🫀 Cardiovascular Risk Factors")

    st.write(
        """
        These findings describe conventional cardiovascular risk factors
        observed in the thyroid-problem research population.
        """
    )

    risk_mapping = {
        "High blood pressure": "High_BP",
        "High cholesterol": "High_Cholesterol",
        "Low HDL": "Low_HDL",
        "Obesity": "Obesity",
        "Smoking-related risk": "Smoking_Risk"
    }

    risk_results = []

    total = len(df)

    for factor, column in risk_mapping.items():

        if column not in df.columns:
            continue

        factor_values = pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0)

        participants = int(
            (factor_values > 0).sum()
        )

        predicted_cvd = int(
            (
                (factor_values > 0)
                &
                (
                    pd.to_numeric(
                        df["CVD_Prediction"],
                        errors="coerce"
                    ).fillna(0) == 1
                )
            ).sum()
        )

        predicted_group_total = int(
            (
                pd.to_numeric(
                    df["CVD_Prediction"],
                    errors="coerce"
                ).fillna(0) == 1
            ).sum()
        )

        percentage_population = (
            participants / total * 100
            if total > 0
            else 0
        )

        percentage_predicted = (
            predicted_cvd / predicted_group_total * 100
            if predicted_group_total > 0
            else 0
        )

        risk_results.append(
            {
                "Risk Factor": factor,
                "Participants": participants,
                "Population %": round(
                    percentage_population,
                    2
                ),
                "Predicted CVD Among Factor": predicted_cvd,
                "Predicted CVD Group %": round(
                    percentage_predicted,
                    2
                )
            }
        )

    risk_df = pd.DataFrame(risk_results)

    st.dataframe(
        risk_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Risk Factor Interpretation")

    st.write(
        """
        The most frequently observed conventional cardiovascular risk factors
        in the research population should receive particular attention.

        Importantly, the percentages shown here describe prevalence and
        association within the dataset. They should not be interpreted as
        proof that any individual factor caused cardiovascular disease.
        """
    )


# ==============================================================================
# PAGE 4 - THYROID ASSESSMENT
# ==============================================================================

elif page == "🦋 Thyroid Assessment":

    st.header("🦋 Thyroid Assessment")

    st.write(
        """
        This section summarizes thyroid biomarkers and their validated
        relationship with model-predicted CVD in the research dataset.
        """
    )

    thyroid_columns = [
        "TSH",
        "Free_T3",
        "Free_T4",
        "TPO",
        "FT3_FT4_Ratio"
    ]

    available = [
        col for col in thyroid_columns
        if col in df.columns
    ]

    summary_rows = []

    for col in available:

        series = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        summary_rows.append(
            {
                "Biomarker": col,
                "Mean": round(series.mean(), 4),
                "Median": round(series.median(), 4),
                "Std": round(series.std(), 4),
                "Minimum": round(series.min(), 4),
                "Maximum": round(series.max(), 4)
            }
        )

    st.dataframe(
        pd.DataFrame(summary_rows),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Validated Thyroid-CVD Findings")

    findings = [
        (
            "TSH",
            "No independent association after validation."
        ),
        (
            "Free T3",
            "No independent association after validation."
        ),
        (
            "Free T4",
            "Unadjusted association, but not independently significant after adjustment."
        ),
        (
            "TPO",
            "No independent association after validation."
        ),
        (
            "FT3/FT4 Ratio",
            "Unadjusted association and association with continuous CVD probability, "
            "but not independently significant after adjustment."
        )
    ]

    for biomarker, interpretation in findings:

        st.markdown(
            f"**{biomarker}:** {interpretation}"
        )

    st.info(
        """
        Research conclusion:

        The analysis does not demonstrate that thyroid biomarkers independently
        cause cardiovascular disease. Conventional cardiovascular factors remain
        important components of cardiovascular risk assessment.
        """
    )


# ==============================================================================
# PAGE 5 - RESEARCH FINDINGS
# ==============================================================================

elif page == "📊 Research Findings":

    st.header("📊 Research Findings")

    total = len(df)

    predicted = int(
        (
            pd.to_numeric(
                df["CVD_Prediction"],
                errors="coerce"
            ).fillna(0) == 1
        ).sum()
    )

    st.subheader("Existing CVD Model")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Participants",
        total
    )

    col2.metric(
        "Predicted CVD",
        predicted
    )

    col3.metric(
        "Percentage",
        f"{predicted / total * 100:.2f}%"
    )

    st.divider()

    st.subheader("CVD Risk Categories")

    if "CVD_Risk_Category" in df.columns:

        category_summary = (
            df["CVD_Risk_Category"]
            .value_counts()
            .rename_axis("Risk Category")
            .reset_index(name="Participants")
        )

        category_summary["Percentage"] = (
            category_summary["Participants"]
            / total
            * 100
        ).round(2)

        st.dataframe(
            category_summary,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            category_summary.set_index("Risk Category")["Participants"]
        )

    st.divider()

    st.subheader("Risk-Factor Burden")

    if "Risk_Factor_Count" in df.columns:

        burden = (
            pd.to_numeric(
                df["Risk_Factor_Count"],
                errors="coerce"
            )
            .value_counts()
            .sort_index()
            .rename_axis("Risk Factor Count")
            .reset_index(name="Participants")
        )

        st.dataframe(
            burden,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            burden.set_index("Risk Factor Count")["Participants"]
        )

    st.divider()

    st.subheader("Research Interpretation")

    st.write(
        """
        The integrated analysis suggests that cardiovascular risk within the
        thyroid-problem population is strongly characterized by conventional
        cardiovascular risk factors.

        The thyroid-specific analyses identified some unadjusted associations,
        particularly involving Free T4 and the FT3/FT4 ratio. However, these
        associations did not remain independently significant after adjustment
        for conventional cardiovascular factors.

        Therefore, the dashboard prioritizes identification and management of
        modifiable cardiovascular risk factors while maintaining appropriate
        thyroid monitoring.
        """
    )


# ==============================================================================
# PAGE 6 - PREVENTION GUIDANCE
# ==============================================================================

elif page == "💡 Prevention Guidance":

    st.header("💡 Cardiovascular Prevention & Risk Management")

    st.write(
        """
        The following guidance is designed around the modifiable risk factors
        identified in the research analysis.
        """
    )

    recommendations = {
        "🩸 Blood Pressure": [
            "Monitor blood pressure regularly.",
            "Discuss persistently elevated readings with a healthcare professional.",
            "Maintain appropriate physical activity and dietary habits."
        ],

        "⚖️ Weight Management": [
            "Aim for gradual and sustainable weight management when appropriate.",
            "Combine healthy nutrition with regular physical activity.",
            "Avoid extreme or unsupervised weight-loss approaches."
        ],

        "🥗 Cholesterol Management": [
            "Monitor total cholesterol and HDL.",
            "Discuss abnormal lipid results with a healthcare professional.",
            "Follow an overall heart-healthy dietary pattern."
        ],

        "🚭 Smoking": [
            "Avoid tobacco use.",
            "If currently smoking, seek evidence-based smoking cessation support.",
            "Reducing and ultimately stopping smoking can substantially reduce cardiovascular risk."
        ],

        "🩺 Diabetes": [
            "Monitor blood glucose appropriately.",
            "Follow the treatment plan recommended by the healthcare professional.",
            "Consider cardiovascular risk alongside diabetes management."
        ],

        "🦋 Thyroid Monitoring": [
            "Maintain appropriate thyroid follow-up.",
            "Repeat thyroid testing when recommended.",
            "Do not change thyroid medication without professional guidance."
        ],

        "🏃 General Cardiovascular Health": [
            "Maintain regular physical activity appropriate to individual health status.",
            "Maintain a balanced diet.",
            "Prioritize adequate sleep.",
            "Attend recommended health screenings."
        ]
    }

    for title, items in recommendations.items():

        with st.expander(title):

            for item in items:

                st.write("• " + item)

    st.divider()

    st.success(
        """
        Prevention principle:

        The goal is not simply to identify risk, but to identify which
        modifiable factors can be addressed and encourage appropriate
        professional follow-up.
        """
    )


# ==============================================================================
# PAGE 7 - SAFETY & LIMITATIONS
# ==============================================================================

elif page == "⚠️ Safety & Limitations":

    st.header("⚠️ Safety, Research Limitations & Interpretation")

    st.error(
        """
        IMPORTANT MEDICAL DISCLAIMER

        This dashboard is a research and educational decision-support tool.
        It does not provide a medical diagnosis, treatment plan or emergency
        medical advice.
        """
    )

    st.subheader("What the dashboard does")

    st.write(
        """
        • Collects thyroid-related and cardiovascular information.

        • Identifies cardiovascular risk factors.

        • Displays existing model results for research records.

        • Interprets thyroid biomarkers using the validated research findings.

        • Provides risk-management and prevention guidance.

        • Encourages professional medical evaluation where appropriate.
        """
    )

    st.subheader("What the dashboard does NOT do")

    st.write(
        """
        • It does not create a new CVD prediction model.

        • It does not modify the original cardiovascular disease variable.

        • It does not use CVD probability as a predictor.

        • It does not claim that thyroid abnormalities cause CVD.

        • It does not replace a physician or qualified healthcare professional.
        """
    )

    st.subheader("Research Interpretation")

    st.info(
        """
        The validated research analysis found that thyroid biomarkers did not
        remain independently significant after adjustment and FDR correction.

        Free T4 and FT3/FT4 ratio showed unadjusted associations, but these
        did not remain independently significant after adjustment.

        Consequently, thyroid findings should be interpreted alongside
        conventional cardiovascular risk factors rather than treated as
        standalone causes of CVD.
        """
    )

    st.subheader("When professional evaluation is important")

    st.write(
        """
        Users should seek appropriate medical evaluation for concerning
        symptoms, substantially abnormal blood pressure, abnormal thyroid
        laboratory results, diabetes, abnormal cholesterol results, or other
        cardiovascular concerns.

        Emergency symptoms should be handled through appropriate emergency
        medical services rather than this dashboard.
        """
    )


# ==============================================================================
# FOOTER
# ==============================================================================

st.divider()

st.caption(
    """
    Thyroid-CVD Research Dashboard | Step 12

    Research/educational decision-support tool.
    Existing CVD model outputs are retained where available.
    No new CVD prediction model is created in this dashboard.
    """
)
