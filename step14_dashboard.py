# ==============================================================================
# STEP 14 - FINAL INTERACTIVE THYROID-CVD DASHBOARD
# ==============================================================================
#
# THYROID + CARDIOVASCULAR RISK ASSESSMENT DASHBOARD
#
# PURPOSE:
# Build the final interactive Streamlit dashboard where a user can:
#
#   1. Enter personal information
#   2. Enter thyroid information
#   3. Enter cardiovascular information
#   4. Enter lifestyle / medical-history information
#   5. Determine whether thyroid problems are present
#   6. Calculate CVD risk using the EXISTING trained CVD model
#   7. Identify cardiovascular risk factors
#   8. Interpret thyroid biomarkers
#   9. Receive prevention / risk-management recommendations
#
# IMPORTANT RESEARCH RULES:
#
# - Do NOT create a new CVD model in this dashboard.
# - Use the ORIGINAL trained CVD model.
# - Do NOT use CVD_Probability as a predictor.
# - Do NOT modify Cardiovascular_Disease.
# - Do NOT claim that thyroid abnormalities cause CVD.
# - Dashboard output is NOT a medical diagnosis.
#
# ==============================================================================

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")


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

MODEL_DIRECTORY = "output"

# Possible model filenames from previous pipeline versions.
# The dashboard will search these automatically.

MODEL_CANDIDATES = [
    "cvd_model.pkl",
    "cvd_model.pickle",
    "trained_cvd_model.pkl",
    "trained_cvd_model.pickle",
    "final_cvd_model.pkl",
    "final_cvd_model.pickle",
    "step5_cvd_model.pkl",
    "step5_cvd_model.pickle",
    "model.pkl",
    "model.pickle",
    "cvd_pipeline.pkl",
    "cvd_pipeline.pickle"
]

SCALER_CANDIDATES = [
    "cvd_scaler.pkl",
    "cvd_scaler.pickle",
    "trained_cvd_scaler.pkl",
    "trained_cvd_scaler.pickle",
    "scaler.pkl",
    "scaler.pickle",
    "standard_scaler.pkl",
    "standard_scaler.pickle"
]

FEATURE_CANDIDATES = [
    "cvd_features.pkl",
    "cvd_features.pickle",
    "model_features.pkl",
    "model_features.pickle",
    "feature_names.pkl",
    "feature_names.pickle"
]


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def safe_float(value):
    """Convert value to float safely."""
    try:
        return float(value)
    except Exception:
        return np.nan


def find_existing_file(candidates):
    """
    Search output directory for the first existing candidate file.
    """
    for filename in candidates:

        path = os.path.join(MODEL_DIRECTORY, filename)

        if os.path.exists(path):
            return path

    return None


def load_pickle_file(path):
    """
    Load a pickle file safely.
    """
    with open(path, "rb") as file:
        return pickle.load(file)


# ==============================================================================
# MODEL LOADING
# ==============================================================================

@st.cache_resource
def load_existing_model():

    model_path = find_existing_file(MODEL_CANDIDATES)

    scaler_path = find_existing_file(SCALER_CANDIDATES)

    feature_path = find_existing_file(FEATURE_CANDIDATES)

    model = None
    scaler = None
    features = None

    if model_path is not None:
        try:
            model = load_pickle_file(model_path)
        except Exception:
            model = None

    if scaler_path is not None:
        try:
            scaler = load_pickle_file(scaler_path)
        except Exception:
            scaler = None

    if feature_path is not None:
        try:
            features = load_pickle_file(feature_path)
        except Exception:
            features = None

    return model, scaler, features, model_path, scaler_path, feature_path


model, scaler, model_features, model_path, scaler_path, feature_path = (
    load_existing_model()
)


# ==============================================================================
# TITLE
# ==============================================================================

st.title("🦋 Thyroid-CVD Risk Assessment Dashboard")

st.markdown(
    """
    ### Integrated Thyroid and Cardiovascular Risk Assessment

    This dashboard is designed for people with thyroid-related concerns.
    It combines thyroid information with conventional cardiovascular
    risk factors to provide an individualized cardiovascular risk
    assessment and prevention guidance.
    """
)


# ==============================================================================
# SAFETY MESSAGE
# ==============================================================================

st.warning(
    """
    **Important:** This dashboard provides a risk assessment and
    educational interpretation. It does not diagnose cardiovascular
    disease or thyroid disease and does not replace evaluation by a
    qualified healthcare professional.
    """
)


# ==============================================================================
# SIDEBAR
# ==============================================================================

with st.sidebar:

    st.header("Dashboard Information")

    st.markdown(
        """
        **Assessment workflow**

        1. Personal information
        2. Thyroid information
        3. Cardiovascular measurements
        4. Lifestyle and medical history
        5. CVD risk assessment
        6. Thyroid interpretation
        7. Risk-factor analysis
        8. Prevention guidance
        """
    )

    st.divider()

    st.subheader("Research safeguards")

    st.write("✓ Existing CVD model retained")
    st.write("✓ CVD probability not used as predictor")
    st.write("✓ Original CVD outcome not modified")
    st.write("✓ No causal thyroid-CVD claim")


# ==============================================================================
# MODEL STATUS
# ==============================================================================

with st.sidebar:

    st.divider()

    st.subheader("Model Status")

    if model is not None:

        st.success("Existing CVD model loaded")

        if model_path:
            st.caption(
                f"Model: {os.path.basename(model_path)}"
            )

    else:

        st.error(
            "Existing CVD model file was not found."
        )

        st.caption(
            "The dashboard cannot calculate a new user's CVD "
            "probability until the original trained model is placed "
            "in the output folder."
        )


# ==============================================================================
# USER INPUT SECTION
# ==============================================================================

st.header("1. User Information")

st.markdown(
    "Enter the user's information below."
)


# ==============================================================================
# DEMOGRAPHICS
# ==============================================================================

st.subheader("Demographic Information")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age (years)",
        min_value=1,
        max_value=120,
        value=50,
        step=1
    )

with col2:

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female",
            "Other / Not specified"
        ]
    )

with col3:

    race = st.selectbox(
        "Race / Ethnicity",
        [
            "Not specified",
            "Asian",
            "White",
            "Black",
            "Hispanic",
            "Other"
        ]
    )


# ==============================================================================
# BODY MEASUREMENTS
# ==============================================================================

st.subheader("Anthropometric Information")

col1, col2, col3, col4 = st.columns(4)

with col1:

    height = st.number_input(
        "Height (cm)",
        min_value=50.0,
        max_value=250.0,
        value=165.0,
        step=0.1
    )

with col2:

    weight = st.number_input(
        "Weight (kg)",
        min_value=10.0,
        max_value=300.0,
        value=70.0,
        step=0.1
    )

with col3:

    bmi_manual = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=80.0,
        value=25.0,
        step=0.1
    )

with col4:

    waist = st.number_input(
        "Waist circumference (cm)",
        min_value=30.0,
        max_value=250.0,
        value=85.0,
        step=0.1
    )


# ==============================================================================
# THYROID INFORMATION
# ==============================================================================

st.header("2. Thyroid Information")

st.subheader("Thyroid Diagnosis")

thyroid_diagnosis = st.selectbox(
    "Has a healthcare professional diagnosed a thyroid condition?",
    [
        "No",
        "Yes"
    ]
)


col1, col2 = st.columns(2)

with col1:

    tsh = st.number_input(
        "TSH",
        min_value=0.0,
        max_value=50.0,
        value=2.5,
        step=0.001
    )

with col2:

    free_t3 = st.number_input(
        "Free T3",
        min_value=0.0,
        max_value=20.0,
        value=2.9,
        step=0.01
    )


col1, col2 = st.columns(2)

with col1:

    free_t4 = st.number_input(
        "Free T4",
        min_value=0.0,
        max_value=20.0,
        value=0.9,
        step=0.01
    )

with col2:

    tpo = st.number_input(
        "TPO",
        min_value=0.0,
        max_value=2000.0,
        value=1.2,
        step=0.1
    )


# ==============================================================================
# THYROID RATIO
# ==============================================================================

if free_t4 > 0:

    ft3_ft4_ratio = free_t3 / free_t4

else:

    ft3_ft4_ratio = np.nan


st.metric(
    "Calculated FT3 / FT4 Ratio",
    f"{ft3_ft4_ratio:.3f}"
)


# ==============================================================================
# CARDIOVASCULAR INFORMATION
# ==============================================================================

st.header("3. Cardiovascular Information")

col1, col2 = st.columns(2)

with col1:

    sys_bp = st.number_input(
        "Systolic Blood Pressure (mmHg)",
        min_value=50,
        max_value=300,
        value=120,
        step=1
    )

with col2:

    dia_bp = st.number_input(
        "Diastolic Blood Pressure (mmHg)",
        min_value=30,
        max_value=200,
        value=80,
        step=1
    )


col1, col2 = st.columns(2)

with col1:

    hdl = st.number_input(
        "HDL Cholesterol",
        min_value=1.0,
        max_value=200.0,
        value=55.0,
        step=0.1
    )

with col2:

    total_cholesterol = st.number_input(
        "Total Cholesterol",
        min_value=1.0,
        max_value=500.0,
        value=200.0,
        step=0.1
    )


# ==============================================================================
# MEDICAL HISTORY
# ==============================================================================

st.header("4. Medical History and Lifestyle")

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
            "Never / No",
            "Yes / Previous or current"
        ]
    )


# ==============================================================================
# THYROID STATUS DETERMINATION
# ==============================================================================

if thyroid_diagnosis == "Yes":

    thyroid_problem = True

else:

    # Dashboard does not independently diagnose thyroid disease.
    # Abnormal laboratory findings are flagged for clinical review.

    thyroid_problem = False


# ==============================================================================
# RUN ASSESSMENT
# ==============================================================================

st.header("5. Run Cardiovascular Risk Assessment")

run_assessment = st.button(
    "🫀 Assess CVD Risk",
    type="primary",
    use_container_width=True
)


# ==============================================================================
# ASSESSMENT FUNCTION
# ==============================================================================

def detect_user_risk_factors():

    factors = []

    # Blood pressure
    if sys_bp >= 130 or dia_bp >= 80:
        factors.append("High blood pressure")

    # Cholesterol
    if total_cholesterol >= 200:
        factors.append("High cholesterol")

    # HDL
    if hdl < 40:
        factors.append("Low HDL")

    # Obesity
    if bmi_manual >= 30:
        factors.append("Obesity")

    # Smoking
    if smoking == "Yes / Previous or current":
        factors.append("Smoking-related risk")

    return factors


def get_primary_risk_factor(factors):

    priority = [
        "Smoking-related risk",
        "High blood pressure",
        "Obesity",
        "High cholesterol",
        "Low HDL"
    ]

    for factor in priority:

        if factor in factors:
            return factor

    return "No major modifiable risk factor flagged"


def generate_prevention_recommendations(factors):

    recommendations = []

    if "High blood pressure" in factors:

        recommendations.append(
            "Prioritize blood-pressure monitoring and discuss "
            "blood-pressure management with a healthcare professional."
        )

    if "Smoking-related risk" in factors:

        recommendations.append(
            "Consider smoking cessation and use professional "
            "cessation support if needed."
        )

    if "Obesity" in factors:

        recommendations.append(
            "Consider gradual weight management through appropriate "
            "nutrition and physical activity."
        )

    if "High cholesterol" in factors:

        recommendations.append(
            "Discuss cholesterol management and appropriate "
            "dietary/lifestyle measures with a healthcare professional."
        )

    if "Low HDL" in factors:

        recommendations.append(
            "Regular physical activity, healthy dietary patterns, "
            "weight management, and avoiding tobacco can support "
            "cardiovascular health."
        )

    recommendations.extend([
        "Maintain regular physical activity appropriate for your health.",
        "Follow a balanced dietary pattern.",
        "Attend regular thyroid and cardiovascular follow-up.",
        "Do not change prescribed medication without medical advice."
    ])

    return recommendations


# ==============================================================================
# MODEL INPUT PREPARATION
# ==============================================================================

def create_model_input():

    data = {

        "Age": age,

        "BMI": bmi_manual,

        "Height": height,

        "Weight": weight,

        "Waist": waist,

        "TSH": tsh,

        "Free_T3": free_t3,

        "Free_T4": free_t4,

        "TPO": tpo,

        "HDL": hdl,

        "Total_Cholesterol": total_cholesterol,

        "SYS_BP": sys_bp,

        "DIA_BP": dia_bp,

        "FT3_FT4_Ratio": ft3_ft4_ratio,

        "Diabetes": 2 if diabetes == "Yes" else 1,

        "Smoking_100_Cigarettes":
            2 if smoking == "Yes / Previous or current" else 1,

        "Thyroid_Diagnosis":
            1 if thyroid_diagnosis == "Yes" else 0
    }

    return pd.DataFrame([data])


# ==============================================================================
# CVD MODEL PREDICTION
# ==============================================================================

def calculate_cvd_prediction(user_df):

    if model is None:

        return None, None, None

    try:

        X = user_df.copy()

        # ----------------------------------------------------------
        # CASE 1: MODEL PIPELINE
        # ----------------------------------------------------------

        if hasattr(model, "predict_proba"):

            # If the model has feature names, try to match them.
            if hasattr(model, "feature_names_in_"):

                expected = list(model.feature_names_in_)

                for col in expected:

                    if col not in X.columns:

                        X[col] = 0

                X = X[expected]

            probability = float(
                model.predict_proba(X)[0][1]
            )

            prediction = int(
                model.predict(X)[0]
            )

            return probability, prediction, "model"

        return None, None, None

    except Exception as error:

        st.error(
            "The existing CVD model could not process the "
            "dashboard inputs."
        )

        st.code(str(error))

        return None, None, None


# ==============================================================================
# RISK CATEGORY
# ==============================================================================

def determine_risk_category(probability):

    if probability is None:
        return "Unavailable"

    # IMPORTANT:
    # If the original model already has a specific risk-category
    # mapping, that mapping should be used instead.
    #
    # These thresholds are ONLY a dashboard display fallback.
    # They are NOT a new prediction model.

    if probability < 0.25:
        return "Low"

    elif probability < 0.50:
        return "Moderate"

    elif probability < 0.75:
        return "High"

    else:
        return "Very High"


# ==============================================================================
# THYROID INTERPRETATION
# ==============================================================================

def thyroid_cvd_interpretation():

    return """
    **Validated thyroid-CVD research finding**

    In the analyzed thyroid-problem population:

    - TSH was not independently associated with model-predicted CVD.
    - Free T3 was not independently associated with model-predicted CVD.
    - Free T4 showed an unadjusted association but did not remain
      independently significant after adjustment.
    - TPO was not independently associated with model-predicted CVD.
    - FT3/FT4 ratio showed an unadjusted association and an association
      with continuous CVD probability, but did not remain independently
      significant after adjustment.

    Therefore, these findings should be interpreted as **associations,
    not proof that thyroid abnormalities cause cardiovascular disease**.
    """


# ==============================================================================
# DISPLAY RESULTS
# ==============================================================================

if run_assessment:

    # --------------------------------------------------------------------------
    # VALIDATE INPUTS
    # --------------------------------------------------------------------------

    if free_t4 <= 0:

        st.error(
            "Free T4 must be greater than zero to calculate the FT3/FT4 ratio."
        )

        st.stop()

    if age < 18:

        st.warning(
            "This dashboard research workflow was developed using an "
            "adult dataset. Results for users under 18 should not be "
            "interpreted using this workflow."
        )

    # --------------------------------------------------------------------------
    # CREATE MODEL INPUT
    # --------------------------------------------------------------------------

    user_df = create_model_input()

    # --------------------------------------------------------------------------
    # RISK FACTORS
    # --------------------------------------------------------------------------

    risk_factors = detect_user_risk_factors()

    primary_risk_factor = get_primary_risk_factor(
        risk_factors
    )

    recommendations = generate_prevention_recommendations(
        risk_factors
    )

    # --------------------------------------------------------------------------
    # CVD MODEL
    # --------------------------------------------------------------------------

    probability, prediction, prediction_source = (
        calculate_cvd_prediction(user_df)
    )

    # ==========================================================================
    # RESULT SECTION
    # ==========================================================================

    st.header("6. Cardiovascular Risk Assessment")

    if probability is not None:

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "CVD Probability",
                f"{probability * 100:.2f}%"
            )

        with col2:

            if prediction == 1:

                st.error(
                    "Model-Predicted CVD"
                )

            else:

                st.success(
                    "No Model-Predicted CVD"
                )

        with col3:

            risk_category = determine_risk_category(
                probability
            )

            st.metric(
                "Risk Category",
                risk_category
            )

    else:

        st.error(
            """
            The original trained CVD model could not be loaded.

            The dashboard can still display the user's cardiovascular
            risk factors and thyroid interpretation, but it cannot
            calculate a new CVD probability until the original model
            file is provided.
            """
        )

        risk_category = "Unavailable"


    # ==========================================================================
    # THYROID STATUS
    # ==========================================================================

    st.header("7. Thyroid Assessment")

    col1, col2 = st.columns(2)

    with col1:

        if thyroid_problem:

            st.info(
                "User-reported thyroid diagnosis: YES"
            )

        else:

            st.info(
                "No thyroid diagnosis reported."
            )

    with col2:

        st.metric(
            "FT3 / FT4 Ratio",
            f"{ft3_ft4_ratio:.3f}"
        )


    # ==========================================================================
    # BIOMARKER VALUES
    # ==========================================================================

    st.subheader("Thyroid Biomarker Profile")

    biomarker_df = pd.DataFrame({

        "Biomarker": [
            "TSH",
            "Free T3",
            "Free T4",
            "TPO",
            "FT3 / FT4 Ratio"
        ],

        "User Value": [
            tsh,
            free_t3,
            free_t4,
            tpo,
            ft3_ft4_ratio
        ]
    })

    st.dataframe(
        biomarker_df,
        use_container_width=True,
        hide_index=True
    )


    st.info(
        """
        Thyroid laboratory values should be interpreted using the
        reference ranges of the laboratory that performed the test,
        together with symptoms, diagnosis, medications, and clinical
        history.
        """
    )


    # ==========================================================================
    # RISK FACTORS
    # ==========================================================================

    st.header("8. Cardiovascular Risk Factors")

    if risk_factors:

        st.warning(
            f"{len(risk_factors)} major modifiable risk factor(s) flagged."
        )

        for factor in risk_factors:

            st.write(
                f"🔴 **{factor}**"
            )

        st.subheader("Primary Modifiable Risk Factor")

        st.error(
            primary_risk_factor
        )

    else:

        st.success(
            "No major modifiable cardiovascular risk factors were "
            "flagged by the dashboard rules."
        )


    # ==========================================================================
    # INPUT SUMMARY
    # ==========================================================================

    st.header("9. Cardiovascular Measurement Summary")

    measurement_df = pd.DataFrame({

        "Measurement": [
            "Age",
            "BMI",
            "Waist",
            "Systolic BP",
            "Diastolic BP",
            "HDL",
            "Total Cholesterol"
        ],

        "Value": [
            age,
            bmi_manual,
            waist,
            sys_bp,
            dia_bp,
            hdl,
            total_cholesterol
        ]
    })

    st.dataframe(
        measurement_df,
        use_container_width=True,
        hide_index=True
    )


    # ==========================================================================
    # THYROID-CVD RESEARCH INTERPRETATION
    # ==========================================================================

    st.header("10. Thyroid-CVD Research Interpretation")

    st.markdown(
        thyroid_cvd_interpretation()
    )


    # ==========================================================================
    # PERSONALIZED INTERPRETATION
    # ==========================================================================

    st.header("11. Individual Risk Interpretation")

    if probability is not None:

        if prediction == 1:

            st.warning(
                f"""
                The existing CVD model classifies this user as
                **model-predicted CVD**, with an estimated probability
                of **{probability * 100:.2f}%**.

                This is a model-based risk assessment and should not
                be interpreted as a confirmed diagnosis.
                """
            )

        else:

            st.success(
                f"""
                The existing CVD model does not classify this user
                as model-predicted CVD. The estimated probability is
                **{probability * 100:.2f}%**.

                This does not mean cardiovascular risk is absent.
                Individual risk factors should still be addressed.
                """
            )


    # ==========================================================================
    # PREVENTION
    # ==========================================================================

    st.header("12. Prevention and Risk-Management Guidance")

    if recommendations:

        for recommendation in recommendations:

            st.write(
                f"✓ {recommendation}"
            )


    # ==========================================================================
    # RISK FACTOR ACTION TABLE
    # ==========================================================================

    st.subheader("Recommended Focus Areas")

    action_rows = []

    if "High blood pressure" in risk_factors:

        action_rows.append({
            "Risk Factor": "High blood pressure",
            "Priority": "High",
            "Suggested Action":
                "Monitor BP and discuss management with a healthcare professional."
        })

    if "Smoking-related risk" in risk_factors:

        action_rows.append({
            "Risk Factor": "Smoking-related risk",
            "Priority": "High",
            "Suggested Action":
                "Consider smoking cessation and professional support."
        })

    if "Obesity" in risk_factors:

        action_rows.append({
            "Risk Factor": "Obesity",
            "Priority": "High",
            "Suggested Action":
                "Consider gradual weight management and regular physical activity."
        })

    if "High cholesterol" in risk_factors:

        action_rows.append({
            "Risk Factor": "High cholesterol",
            "Priority": "High",
            "Suggested Action":
                "Discuss lipid management and lifestyle measures with a clinician."
        })

    if "Low HDL" in risk_factors:

        action_rows.append({
            "Risk Factor": "Low HDL",
            "Priority": "Moderate",
            "Suggested Action":
                "Focus on physical activity, healthy diet, weight management, and avoiding tobacco."
        })

    if action_rows:

        action_df = pd.DataFrame(action_rows)

        st.dataframe(
            action_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No specific modifiable risk factor action was triggered."
        )


    # ==========================================================================
    # MEDICAL SAFETY
    # ==========================================================================

    st.header("13. Clinical Safety Message")

    st.error(
        """
        **This dashboard is not a medical diagnostic tool.**

        The displayed CVD probability is based on the existing trained
        research model and should be interpreted as a risk estimate.

        Thyroid biomarker findings represent statistical associations
        identified in the research dataset and do not prove causation.

        Please discuss abnormal thyroid results, blood pressure,
        cholesterol, diabetes, smoking history, or elevated
        cardiovascular risk with a qualified healthcare professional.

        Do not start, stop, or change medication based solely on this
        dashboard.

        Seek urgent medical attention for severe or sudden symptoms
        such as chest pain, severe difficulty breathing, fainting,
        or other emergency symptoms.
        """
    )


    # ==========================================================================
    # TECHNICAL INFORMATION
    # ==========================================================================

    with st.expander("Technical / Research Information"):

        st.write(
            "**Prediction source:**",
            prediction_source
        )

        st.write(
            "**CVD probability used as predictor:** NO"
        )

        st.write(
            "**New CVD model created:** NO"
        )

        st.write(
            "**Original Cardiovascular_Disease modified:** NO"
        )

        st.write(
            "**Thyroid-CVD causal claim:** NO"
        )

        st.write(
            "**Dashboard risk factors:**",
            ", ".join(risk_factors)
            if risk_factors
            else "None"
        )

        st.write(
            "**Primary modifiable risk factor:**",
            primary_risk_factor
        )


# ==============================================================================
# FOOTER
# ==============================================================================

st.divider()

st.caption(
    """
    Thyroid-CVD Integrated Risk Assessment Research Dashboard |
    Research and educational use only |
    Model output does not constitute medical diagnosis
    """
)
