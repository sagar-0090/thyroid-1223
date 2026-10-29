# =============================================================================
# THYROID-CVD INTELLIGENT RISK ASSESSMENT DASHBOARD
# FINAL PROJECT APPLICATION
# =============================================================================
#
# Run:
#     streamlit run app.py
#
# The application:
# 1. Loads the existing project dataset.
# 2. Trains/loads the CVD Random Forest model using the original
#    Cardiovascular_Disease target.
# 3. Accepts new user information.
# 4. Predicts CVD probability.
# 5. Identifies cardiovascular risk factors.
# 6. Interprets thyroid findings.
# 7. Provides prevention/risk-management guidance.
# 8. Displays validated research findings.
#
# IMPORTANT:
# - CVD_Probability is NEVER used as a predictor.
# - CVD_Prediction is NEVER used as a predictor.
# - Cardiovascular_Disease is NEVER modified.
# - No causal thyroid -> CVD claim is made.
# =============================================================================

import os
import json
import warnings

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Thyroid-CVD Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "model_artifacts")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILE = os.path.join(MODEL_DIR, "thyroid_cvd_random_forest.joblib")
MODEL_INFO_FILE = os.path.join(MODEL_DIR, "model_information.json")
IMPORTANCE_FILE = os.path.join(
    OUTPUT_DIR,
    "final_cvd_feature_importance.csv"
)


# =============================================================================
# DATA FILE DISCOVERY
# =============================================================================

DATA_CANDIDATES = [
    os.path.join(OUTPUT_DIR, "step5_integrated_thyroid_cvd_dataset.csv"),
    os.path.join(OUTPUT_DIR, "step11_dashboard_ready_data.csv"),
    os.path.join(OUTPUT_DIR, "step10_dashboard_ready_data.csv"),
    os.path.join(OUTPUT_DIR, "step9_dashboard_ready_data.csv"),
    os.path.join(BASE_DIR, "cleaned_dataset.csv"),
    os.path.join(BASE_DIR, "master_dataset.csv"),
]


# =============================================================================
# ORIGINAL MODEL FEATURES
# =============================================================================
#
# These are the clinical/user variables used for CVD prediction.
#
# Derived fields such as:
#     CVD_Probability
#     CVD_Prediction
#     CVD_Prediction_0.50
#     CVD_Risk_Category
#     Integrated_Thyroid_CVD_Group
#
# are deliberately excluded to prevent leakage.
# =============================================================================

MODEL_FEATURES = [
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
]

TARGET = "Cardiovascular_Disease"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def find_dataset():
    """
    Find the best available project dataset.

    Preference:
    1. Master/cleaned datasets containing the original CVD target.
    2. Integrated project datasets.
    """

    existing = []

    for path in DATA_CANDIDATES:
        if os.path.exists(path):
            try:
                temp = pd.read_csv(path)

                if TARGET in temp.columns:
                    existing.append(
                        (
                            path,
                            len(temp),
                            temp
                        )
                    )
            except Exception:
                pass

    if not existing:
        raise FileNotFoundError(
            "No dataset containing Cardiovascular_Disease was found.\n\n"
            "Expected one of:\n"
            + "\n".join(DATA_CANDIDATES)
        )

    # Prefer the dataset with the largest number of records.
    existing.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return existing[0][0], existing[0][2]


def clean_target(series):
    """
    Convert target to binary 0/1.
    """

    s = series.copy()

    # Numeric target
    numeric = pd.to_numeric(
        s,
        errors="coerce"
    )

    if numeric.notna().sum() == len(s):
        unique = sorted(
            numeric.dropna().unique().tolist()
        )

        if set(unique).issubset({0, 1}):
            return numeric.astype(int)

    # String target fallback
    text = (
        s.astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "0": 0,
        "1": 1,
        "no": 0,
        "yes": 1,
        "false": 0,
        "true": 1,
        "negative": 0,
        "positive": 1,
        "absent": 0,
        "present": 1,
    }

    result = text.map(mapping)

    if result.notna().sum() != len(result):
        raise ValueError(
            "Cardiovascular_Disease contains values that could "
            "not be converted to binary 0/1."
        )

    return result.astype(int)


def convert_numeric_columns(df, columns):
    """
    Safely convert numeric clinical fields.
    """

    result = df.copy()

    for col in columns:
        if col in result.columns:
            result[col] = pd.to_numeric(
                result[col],
                errors="coerce"
            )

    return result


# =============================================================================
# LOAD DATA
# =============================================================================

@st.cache_data
def load_project_data():

    path, df = find_dataset()

    return path, df


try:
    DATA_PATH, df = load_project_data()

except Exception as e:

    st.error("Unable to load the project dataset.")
    st.exception(e)
    st.stop()


# =============================================================================
# VALIDATE FEATURES
# =============================================================================

missing_features = [
    col
    for col in MODEL_FEATURES
    if col not in df.columns
]

if TARGET not in df.columns:

    st.error(
        f"Required target column '{TARGET}' was not found."
    )

    st.stop()


# If using an output file that lacks some original features,
# stop clearly instead of silently inventing values.

if missing_features:

    st.error(
        "The selected dataset does not contain all original model "
        "input features."
    )

    st.write("Missing:")
    st.write(missing_features)

    st.stop()


# =============================================================================
# PREPARE MODEL DATA
# =============================================================================

model_df = df[
    MODEL_FEATURES + [TARGET]
].copy()


# Numeric fields
NUMERIC_FEATURES = [
    "Age",
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
    "SYS_BP",
    "DIA_BP",
]

# Categorical fields
CATEGORICAL_FEATURES = [
    "Gender",
    "Race",
    "Diabetes",
    "Smoking_100_Cigarettes",
    "Thyroid_Diagnosis",
]


model_df = convert_numeric_columns(
    model_df,
    NUMERIC_FEATURES
)

model_df[TARGET] = clean_target(
    model_df[TARGET]
)

model_df = model_df.dropna(
    subset=[TARGET]
)

X = model_df[MODEL_FEATURES].copy()
y = model_df[TARGET].copy()


# =============================================================================
# BUILD CVD RANDOM FOREST MODEL
# =============================================================================

@st.cache_resource
def build_cvd_model(X_input, y_input):

    X_train, X_test, y_train, y_test = train_test_split(
        X_input,
        y_input,
        test_size=0.20,
        random_state=42,
        stratify=y_input
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )
        ]
    )

    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                rf
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    probability = pipeline.predict_proba(
        X_test
    )[:, 1]

    prediction = (
        probability >= 0.50
    ).astype(int)

    metrics = {
        "Accuracy": accuracy_score(
            y_test,
            prediction
        ),
        "Precision": precision_score(
            y_test,
            prediction,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            prediction,
            zero_division=0
        ),
        "F1": f1_score(
            y_test,
            prediction,
            zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y_test,
            probability
        ),
        "Training_Records": len(X_train),
        "Testing_Records": len(X_test),
    }

    return pipeline, metrics


# =============================================================================
# TRAIN MODEL
# =============================================================================

with st.spinner(
    "Preparing the CVD Random Forest model..."
):

    model, model_metrics = build_cvd_model(
        X,
        y
    )


# =============================================================================
# MODEL FEATURE IMPORTANCE
# =============================================================================
def calculate_feature_importance(fitted_model):

    try:

        transformed = fitted_model.named_steps[
            "preprocessor"
        ].get_feature_names_out()

        importance = fitted_model.named_steps[
            "model"
        ].feature_importances_

        result = pd.DataFrame(
            {
                "Feature": transformed,
                "Importance": importance
            }
        )

        result = result.sort_values(
            "Importance",
            ascending=False
        )

        return result

    except Exception as e:

        return pd.DataFrame(
            columns=[
                "Feature",
                "Importance"
            ]
        )


feature_importance = calculate_feature_importance(model)


try:
    feature_importance.to_csv(
        IMPORTANCE_FILE,
        index=False
    )
except Exception:
    pass


# =============================================================================
# MODEL INFORMATION
# =============================================================================

model_info = {
    "Model": "Random Forest Classifier",
    "Purpose": "CVD risk prediction",
    "Target": TARGET,
    "Number_of_input_features": len(MODEL_FEATURES),
    "Training_records": int(len(X)),
    "Accuracy": float(model_metrics["Accuracy"]),
    "Precision": float(model_metrics["Precision"]),
    "Recall": float(model_metrics["Recall"]),
    "F1": float(model_metrics["F1"]),
    "ROC_AUC": float(model_metrics["ROC_AUC"]),
    "CVD_Probability_used_as_predictor": False,
    "CVD_Prediction_used_as_predictor": False,
    "Cardiovascular_Disease_modified": False,
}

try:
    with open(
        MODEL_INFO_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            model_info,
            f,
            indent=4
        )

except Exception:
    pass


# =============================================================================
# THYROID LOGIC
# =============================================================================

def determine_thyroid_problem(
    diagnosis,
    tsh,
    free_t3,
    free_t4,
    tpo
):

    diagnosis_text = str(
        diagnosis
    ).strip().lower()

    thyroid_words = [
        "hypothyroid",
        "hyperthyroid",
        "thyroid",
        "hashimoto",
        "graves",
        "goiter",
        "thyro",
    ]

    diagnosis_positive = any(
        word in diagnosis_text
        for word in thyroid_words
    )

    abnormal_tsh = (
        tsh < 0.4
        or tsh > 4.0
    )

    # We intentionally do not create a diagnosis from
    # a single biomarker. These are dashboard flags only.

    thyroid_problem = (
        diagnosis_positive
        or abnormal_tsh
    )

    return thyroid_problem, abnormal_tsh


def thyroid_interpretation(
    diagnosis,
    tsh,
    free_t3,
    free_t4,
    tpo,
    ratio
):

    diagnosis_text = str(
        diagnosis
    ).lower()

    messages = []

    # General thyroid interpretation
    if (
        "hypo" in diagnosis_text
        or tsh > 4.0
    ):

        messages.append(
            "The entered information is compatible with a "
            "hypothyroid-pattern assessment and should be "
            "interpreted with the person's clinical history "
            "and laboratory reference ranges."
        )

    elif (
        "hyper" in diagnosis_text
        or tsh < 0.4
    ):

        messages.append(
            "The entered information is compatible with a "
            "hyperthyroid-pattern assessment and should be "
            "interpreted with the person's clinical history "
            "and laboratory reference ranges."
        )

    else:

        messages.append(
            "The entered information does not show a clear "
            "thyroid-pattern flag using the dashboard rules."
        )

    # Research-specific interpretation
    messages.append(
        "In the project's validated thyroid-CVD analysis, "
        "TSH and Free T3 were not independently associated "
        "with model-predicted CVD after adjustment."
    )

    messages.append(
        "Free T4 showed an unadjusted association with "
        "model-predicted CVD but did not remain independently "
        "significant after adjustment."
    )

    messages.append(
        "The FT3/FT4 ratio showed an unadjusted association "
        "and an association with continuous CVD probability, "
        "but it did not remain independently significant "
        "after adjustment."
    )

    messages.append(
        "TPO was not independently associated with "
        "model-predicted CVD in the validated analysis."
    )

    return messages


# =============================================================================
# CVD RISK FACTOR LOGIC
# =============================================================================

def identify_risk_factors(
    age,
    bmi,
    waist,
    hdl,
    total_cholesterol,
    diabetes,
    smoking,
    systolic,
    diastolic
):

    factors = []

    # Blood pressure
    if (
        systolic >= 130
        or diastolic >= 80
    ):

        factors.append(
            "High blood pressure"
        )

    # Obesity
    if bmi >= 30:

        factors.append(
            "Obesity"
        )

    # Cholesterol
    if total_cholesterol >= 200:

        factors.append(
            "High cholesterol"
        )

    # HDL
    if hdl < 40:

        factors.append(
            "Low HDL"
        )

    # Smoking
    if smoking == "Yes":

        factors.append(
            "Smoking-related risk"
        )

    # Diabetes
    if diabetes == "Yes":

        factors.append(
            "Diabetes"
        )

    # Age
    if age >= 65:

        factors.append(
            "Age-related cardiovascular risk"
        )

    return factors


# =============================================================================
# RISK CATEGORY
# =============================================================================

def risk_category(probability):

    if probability < 0.20:

        return "Low"

    elif probability < 0.40:

        return "Moderate"

    elif probability < 0.60:

        return "High"

    else:

        return "Very High"


# =============================================================================
# PREVENTION ENGINE
# =============================================================================

def prevention_recommendations(
    factors,
    thyroid_problem,
    cvd_probability
):

    recommendations = []

    if "High blood pressure" in factors:

        recommendations.append(
            "Prioritize regular blood-pressure monitoring. "
            "If readings remain elevated, discuss them with "
            "a healthcare professional."
        )

    if "Obesity" in factors:

        recommendations.append(
            "Consider gradual and sustainable weight management "
            "through an appropriate eating pattern, physical "
            "activity, sleep and professional guidance."
        )

    if "High cholesterol" in factors:

        recommendations.append(
            "Review the lipid profile with a healthcare professional "
            "and discuss dietary, lifestyle and treatment options "
            "when appropriate."
        )

    if "Low HDL" in factors:

        recommendations.append(
            "Increase regular physical activity and focus on a "
            "heart-healthy dietary pattern while addressing other "
            "modifiable cardiovascular risks."
        )

    if "Smoking-related risk" in factors:

        recommendations.append(
            "Smoking cessation is one of the most important "
            "modifiable cardiovascular risk measures."
        )

    if "Diabetes" in factors:

        recommendations.append(
            "Maintain appropriate blood-glucose management and "
            "discuss cardiovascular risk with a healthcare professional."
        )

    if "Age-related cardiovascular risk" in factors:

        recommendations.append(
            "Because cardiovascular risk generally increases with age, "
            "regular cardiovascular risk assessment is appropriate."
        )

    if thyroid_problem:

        recommendations.append(
            "Continue appropriate thyroid monitoring and discuss "
            "abnormal thyroid laboratory findings with a qualified "
            "healthcare professional."
        )

    if cvd_probability >= 0.70:

        recommendations.append(
            "The model-estimated probability is high. Professional "
            "cardiovascular evaluation is recommended."
        )

    elif cvd_probability >= 0.50:

        recommendations.append(
            "The model-estimated probability is elevated. Consider "
            "professional cardiovascular risk assessment."
        )

    if not recommendations:

        recommendations.append(
            "Continue routine preventive care, healthy lifestyle "
            "practices and appropriate thyroid and cardiovascular monitoring."
        )

    return recommendations


# =============================================================================
# INDIVIDUAL INTERPRETATION
# =============================================================================

def individual_interpretation(
    thyroid_problem,
    cvd_probability,
    category,
    factors
):

    text = []

    if thyroid_problem:

        text.append(
            "The entered information is being assessed through "
            "the thyroid-CVD pathway."
        )

    else:

        text.append(
            "The entered information does not currently trigger "
            "the dashboard's thyroid-problem flag."
        )

    text.append(
        f"The CVD Random Forest model estimates a "
        f"{cvd_probability * 100:.1f}% probability of CVD."
    )

    text.append(
        f"The corresponding dashboard risk category is "
        f"{category}."
    )

    if factors:

        text.append(
            "The main dashboard-defined cardiovascular risk "
            "factors requiring attention are: "
            + ", ".join(factors)
            + "."
        )

    else:

        text.append(
            "No major dashboard-defined modifiable cardiovascular "
            "risk factor was flagged."
        )

    text.append(
        "These findings represent model-based risk assessment "
        "and risk-factor identification, not a medical diagnosis."
    )

    return " ".join(text)


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title(
    "❤️ Thyroid-CVD Dashboard"
)

st.sidebar.caption(
    "ML-based cardiovascular risk assessment "
    "for people with thyroid-related concerns"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "👤 User CVD Assessment",
        "🦋 Thyroid Interpretation",
        "📊 Research Findings",
        "🤖 ML Model"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "This application is a research and risk-assessment "
    "tool. It does not provide a medical diagnosis."
)


# =============================================================================
# PAGE 1 — DASHBOARD
# =============================================================================

if page == "🏠 Dashboard":

    st.title(
        "❤️ Thyroid-CVD Risk Assessment Dashboard"
    )

    st.subheader(
        "Integrated Machine Learning and Thyroid Risk Analysis"
    )

    st.write(
        """
        This dashboard combines the project's validated thyroid-CVD
        research findings with a Random Forest CVD prediction model.

        A user can enter demographic, cardiovascular, lifestyle and
        thyroid information to obtain an estimated CVD probability,
        risk category, identified risk factors and prevention guidance.
        """
    )

    # -------------------------------------------------------------------------
    # Population metrics
    # -------------------------------------------------------------------------

    total_records = len(df)

    if "Thyroid_Problem" in df.columns:

        thyroid_population = int(
            pd.to_numeric(
                df["Thyroid_Problem"],
                errors="coerce"
            ).fillna(0).sum()
        )

    elif "Thyroid_Status" in df.columns:

        thyroid_population = int(
            (
                df["Thyroid_Status"]
                .astype(str)
                .str.contains(
                    "Thyroid Problem",
                    case=False,
                    na=False
                )
            ).sum()
        )

    else:

        thyroid_population = 0

    if "CVD_Prediction" in df.columns:

        predicted_cvd = int(
            pd.to_numeric(
                df["CVD_Prediction"],
                errors="coerce"
            ).fillna(0).sum()
        )

    else:

        predicted_cvd = None

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Research Records",
        total_records
    )

    col2.metric(
        "Thyroid-Problem Records",
        thyroid_population
    )

    if predicted_cvd is not None:

        col3.metric(
            "Existing Model CVD",
            predicted_cvd
        )

    else:

        col3.metric(
            "CVD Model",
            "Active"
        )

    col4.metric(
        "Random Forest ROC-AUC",
        f"{model_metrics['ROC_AUC']:.3f}"
    )

    st.divider()

    # -------------------------------------------------------------------------
    # Research result
    # -------------------------------------------------------------------------

    st.subheader(
        "Key Research Finding"
    )

    st.info(
        """
        Within the project's thyroid-problem population, cardiovascular
        risk was strongly associated with conventional cardiovascular
        risk factors. The validated thyroid-specific analysis did not
        establish an independently significant thyroid biomarker after
        adjustment for conventional factors.
        """
    )

    # -------------------------------------------------------------------------
    # Main risk factors
    # -------------------------------------------------------------------------

    st.subheader(
        "Important Risk Factors Identified in the Research Population"
    )

    risk_summary = []

    if (
        "High_BP" in df.columns
        and "CVD_Prediction" in df.columns
    ):

        high_bp = pd.to_numeric(
            df["High_BP"],
            errors="coerce"
        ).fillna(0)

        cvd = pd.to_numeric(
            df["CVD_Prediction"],
            errors="coerce"
        ).fillna(0)

        risk_summary.append(
            {
                "Risk Factor": "High blood pressure",
                "Participants": int(high_bp.sum()),
                "Percentage": round(
                    high_bp.mean() * 100,
                    2
                ),
                "Predicted CVD Among Factor": int(
                    ((high_bp == 1) & (cvd == 1)).sum()
                )
            }
        )

    if (
        "Obesity" in df.columns
        and "CVD_Prediction" in df.columns
    ):

        obesity = pd.to_numeric(
            df["Obesity"],
            errors="coerce"
        ).fillna(0)

        risk_summary.append(
            {
                "Risk Factor": "Obesity",
                "Participants": int(obesity.sum()),
                "Percentage": round(
                    obesity.mean() * 100,
                    2
                ),
                "Predicted CVD Among Factor": int(
                    ((obesity == 1) & (cvd == 1)).sum()
                )
            }
        )

    if (
        "Smoking_Risk" in df.columns
        and "CVD_Prediction" in df.columns
    ):

        smoking = pd.to_numeric(
            df["Smoking_Risk"],
            errors="coerce"
        ).fillna(0)

        risk_summary.append(
            {
                "Risk Factor": "Smoking-related risk",
                "Participants": int(smoking.sum()),
                "Percentage": round(
                    smoking.mean() * 100,
                    2
                ),
                "Predicted CVD Among Factor": int(
                    ((smoking == 1) & (cvd == 1)).sum()
                )
            }
        )

    if (
        "Low_HDL" in df.columns
        and "CVD_Prediction" in df.columns
    ):

        low_hdl = pd.to_numeric(
            df["Low_HDL"],
            errors="coerce"
        ).fillna(0)

        risk_summary.append(
            {
                "Risk Factor": "Low HDL",
                "Participants": int(low_hdl.sum()),
                "Percentage": round(
                    low_hdl.mean() * 100,
                    2
                ),
                "Predicted CVD Among Factor": int(
                    ((low_hdl == 1) & (cvd == 1)).sum()
                )
            }
        )

    if (
        "High_Cholesterol" in df.columns
        and "CVD_Prediction" in df.columns
    ):

        cholesterol = pd.to_numeric(
            df["High_Cholesterol"],
            errors="coerce"
        ).fillna(0)

        risk_summary.append(
            {
                "Risk Factor": "High cholesterol",
                "Participants": int(cholesterol.sum()),
                "Percentage": round(
                    cholesterol.mean() * 100,
                    2
                ),
                "Predicted CVD Among Factor": int(
                    ((cholesterol == 1) & (cvd == 1)).sum()
                )
            }
        )

    if risk_summary:

        st.dataframe(
            pd.DataFrame(risk_summary),
            use_container_width=True,
            hide_index=True
        )


# =============================================================================
# PAGE 2 — USER CVD ASSESSMENT
# =============================================================================

elif page == "👤 User CVD Assessment":

    st.title(
        "👤 Individual Thyroid-CVD Risk Assessment"
    )

    st.write(
        """
        Enter the user's information. The Random Forest model will
        estimate CVD probability and the dashboard will identify
        cardiovascular risk factors and thyroid-related findings.
        """
    )

    st.warning(
        """
        The result is a model-based risk estimate, not a medical diagnosis.
        Clinical decisions should be made by a qualified healthcare professional.
        """
    )

    # -------------------------------------------------------------------------
    # Demographics
    # -------------------------------------------------------------------------

    st.header(
        "1. Demographic Information"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=45
        )

    with c2:

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female",
                "Other"
            ]
        )

    with c3:

        race = st.selectbox(
            "Race",
            [
                "White",
                "Black",
                "Asian",
                "Hispanic",
                "Other",
                "Unknown"
            ]
        )

    # -------------------------------------------------------------------------
    # Body measurements
    # -------------------------------------------------------------------------

    st.header(
        "2. Body Measurements"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        height = st.number_input(
            "Height (cm)",
            min_value=50.0,
            max_value=250.0,
            value=165.0
        )

    with c2:

        weight = st.number_input(
            "Weight (kg)",
            min_value=20.0,
            max_value=300.0,
            value=70.0
        )

    with c3:

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=80.0,
            value=25.0
        )

    with c4:

        waist = st.number_input(
            "Waist circumference (cm)",
            min_value=30.0,
            max_value=250.0,
            value=90.0
        )

    # -------------------------------------------------------------------------
    # Cardiovascular measurements
    # -------------------------------------------------------------------------

    st.header(
        "3. Cardiovascular Information"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        systolic = st.number_input(
            "Systolic BP (mmHg)",
            min_value=60.0,
            max_value=250.0,
            value=120.0
        )

    with c2:

        diastolic = st.number_input(
            "Diastolic BP (mmHg)",
            min_value=30.0,
            max_value=180.0,
            value=80.0
        )

    with c3:

        hdl = st.number_input(
            "HDL",
            min_value=10.0,
            max_value=150.0,
            value=55.0
        )

    with c4:

        total_cholesterol = st.number_input(
            "Total Cholesterol",
            min_value=50.0,
            max_value=500.0,
            value=200.0
        )

    # -------------------------------------------------------------------------
    # Medical history
    # -------------------------------------------------------------------------

    st.header(
        "4. Medical and Lifestyle Information"
    )

    c1, c2 = st.columns(2)

    with c1:

        diabetes = st.selectbox(
            "Diabetes",
            [
                "No",
                "Yes"
            ]
        )

    with c2:

        smoking = st.selectbox(
            "Has smoked 100+ cigarettes in lifetime?",
            [
                "No",
                "Yes"
            ]
        )

    # -------------------------------------------------------------------------
    # Thyroid
    # -------------------------------------------------------------------------

    st.header(
        "5. Thyroid Information"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        tsh = st.number_input(
            "TSH",
            min_value=0.001,
            max_value=50.0,
            value=2.0,
            step=0.01
        )

    with c2:

        free_t3 = st.number_input(
            "Free T3",
            min_value=0.1,
            max_value=10.0,
            value=2.9,
            step=0.1
        )

    with c3:

        free_t4 = st.number_input(
            "Free T4",
            min_value=0.1,
            max_value=5.0,
            value=0.9,
            step=0.1
        )

    c1, c2 = st.columns(2)

    with c1:

        tpo = st.number_input(
            "TPO",
            min_value=0.0,
            max_value=1000.0,
            value=1.0,
            step=0.1
        )

    with c2:

        thyroid_diagnosis = st.selectbox(
            "Thyroid diagnosis",
            [
                "No known thyroid diagnosis",
                "Hypothyroidism",
                "Hyperthyroidism",
                "Hashimoto's thyroiditis",
                "Graves' disease",
                "Other thyroid disorder"
            ]
        )

    st.divider()

    # -------------------------------------------------------------------------
    # ASSESS BUTTON
    # -------------------------------------------------------------------------

    assess = st.button(
        "🔍 ASSESS CVD RISK",
        type="primary",
        use_container_width=True
    )

    if assess:

        # ---------------------------------------------------------------------
        # BMI calculation check
        # ---------------------------------------------------------------------

        calculated_bmi = (
            weight
            /
            ((height / 100) ** 2)
        )

        # Use user-entered BMI for model consistency,
        # but display calculated BMI as a check.
        bmi_difference = abs(
            calculated_bmi - bmi
        )

        # ---------------------------------------------------------------------
        # Thyroid status
        # ---------------------------------------------------------------------

        thyroid_problem, abnormal_tsh = (
            determine_thyroid_problem(
                thyroid_diagnosis,
                tsh,
                free_t3,
                free_t4,
                tpo
            )
        )

        ratio = (
            free_t3 / free_t4
            if free_t4 > 0
            else np.nan
        )

        # ---------------------------------------------------------------------
        # Risk factors
        # ---------------------------------------------------------------------

        factors = identify_risk_factors(
            age=age,
            bmi=bmi,
            waist=waist,
            hdl=hdl,
            total_cholesterol=total_cholesterol,
            diabetes=diabetes,
            smoking=smoking,
            systolic=systolic,
            diastolic=diastolic
        )

        # ---------------------------------------------------------------------
        # Convert user values to model format
        # ---------------------------------------------------------------------

        user_input = pd.DataFrame(
            [
                {
                    "Age": age,
                    "Gender": gender,
                    "Race": race,
                    "BMI": bmi,
                    "Height": height,
                    "Weight": weight,
                    "Waist": waist,
                    "TSH": tsh,
                    "Free_T3": free_t3,
                    "Free_T4": free_t4,
                    "TPO": tpo,
                    "HDL": hdl,
                    "Total_Cholesterol": total_cholesterol,
                    "Diabetes": diabetes,
                    "Smoking_100_Cigarettes": smoking,
                    "Thyroid_Diagnosis": thyroid_diagnosis,
                    "SYS_BP": systolic,
                    "DIA_BP": diastolic,
                }
            ]
        )

        # ---------------------------------------------------------------------
        # MODEL PREDICTION
        # ---------------------------------------------------------------------

          # Keep unless validation supports another threshold
        probability = float(
            model.predict_proba(
                user_input
            )[0, 1]

        )
        CVD_THRESHOLD = 0.50
        prediction = int(
            probability >= CVD_THRESHOLD
        )

        category = risk_category(probability)

        prediction = int(
            probability >= 0.50
        )

        category = risk_category(
            probability
        )

        # ---------------------------------------------------------------------
        # DISPLAY RESULT
        # ---------------------------------------------------------------------

        st.header(
            "6. CVD Risk Result"
        )

        result_col1, result_col2, result_col3 = st.columns(3)

        result_col1.metric(
            "CVD Probability",
            f"{probability * 100:.1f}%"
        )

        result_col2.metric(
            "CVD Prediction",
            "Predicted CVD"
            if prediction == 1
            else "No Predicted CVD"
        )

        result_col3.metric(
            "Risk Category",
            category
        )
        st.caption(
            "The CVD probability is a machine-learning estimate, "
            "not a confirmed clinical probability. 'No Predicted "
            "CVD' means the model's probability is below its "
            "classification threshold; it does not rule out "
            "existing cardiovascular disease. The displayed "
            "risk category uses this application's configured "
            "probability bands and is not a clinical diagnosis."

        )

         

        # ---------------------------------------------------------------------
        # Thyroid result
        # ---------------------------------------------------------------------

        st.header(
            "7. Thyroid Assessment"
        )

        t1, t2, t3 = st.columns(3)

        t1.metric(
            "Thyroid Problem",
            "Yes"
            if thyroid_problem
            else "No"
        )

        t2.metric(
            "Abnormal TSH",
            "Yes"
            if abnormal_tsh
            else "No"
        )

        t3.metric(
            "FT3 / FT4 Ratio",
            f"{ratio:.2f}"
        )

        # ---------------------------------------------------------------------
        # Risk factors
        # ---------------------------------------------------------------------

        st.header(
            "8. Identified Cardiovascular Risk Factors"
        )

        if factors:

            for factor in factors:

                st.warning(
                    f"⚠️ {factor}"
                )

        else:

            st.success(
                "No major dashboard-defined modifiable "
                "cardiovascular risk factor was flagged."
            )

        # ---------------------------------------------------------------------
        # Primary risk factors
        # ---------------------------------------------------------------------

        st.header(
            "9. What Is Contributing to the Risk?"
        )

        if factors:

            st.write(
                "Based on the entered information, the dashboard "
                "identified the following factors that may contribute "
                "to cardiovascular risk:"
            )

            for i, factor in enumerate(
                factors,
                start=1
            ):

                st.write(
                    f"**{i}. {factor}**"
                )

        else:

            st.write(
                "No major dashboard-defined modifiable risk factor "
                "was identified."
            )

        # ---------------------------------------------------------------------
        # Individual interpretation
        # ---------------------------------------------------------------------

        st.header(
            "10. Individual Interpretation"
        )

        interpretation = individual_interpretation(
            thyroid_problem,
            probability,
            category,
            factors
        )

        st.info(
            interpretation
        )

        # ---------------------------------------------------------------------
        # Thyroid interpretation
        # ---------------------------------------------------------------------

        st.header(
            "11. Thyroid-CVD Interpretation"
        )

        thyroid_messages = thyroid_interpretation(
            thyroid_diagnosis,
            tsh,
            free_t3,
            free_t4,
            tpo,
            ratio
        )

        for message in thyroid_messages:

            st.write(
                f"• {message}"
            )
        if tsh >= 10:
            st.warning(
                 "The entered TSH is 10 or higher. Review the "
                "TSH and Free T4 results with a healthcare "
                "professional, using the laboratory's reference "
                "ranges and the person's clinical history. "
                "These values alone do not establish a diagnosis."

            )


        # ---------------------------------------------------------------------
        # Prevention
        # ---------------------------------------------------------------------

        st.header(
            "12. Prevention / Risk-Management Measures"
        )

        recommendations = prevention_recommendations(
            factors,
            thyroid_problem,
            probability
        )

        for recommendation in recommendations:

            st.write(
                f"✅ {recommendation}"
            )

        # ---------------------------------------------------------------------
        # BMI validation
        # ---------------------------------------------------------------------

        if bmi_difference > 1.0:

            st.warning(
                f"The entered BMI ({bmi:.1f}) differs from the "
                f"BMI calculated from height and weight "
                f"({calculated_bmi:.1f}). Verify the measurements."
            )

        # ---------------------------------------------------------------------
        # Safety
        # ---------------------------------------------------------------------

        st.error(
            """
            IMPORTANT MEDICAL SAFETY MESSAGE

            This dashboard provides a machine-learning risk estimate
            and research-based interpretation. It does not diagnose
            cardiovascular disease, thyroid disease, heart attack,
            stroke or any other medical condition.

            A healthcare professional should interpret the result
            together with clinical history, examination and appropriate
            laboratory/diagnostic information.
            """
        )


# =============================================================================
# PAGE 3 — THYROID INTERPRETATION
# =============================================================================

elif page == "🦋 Thyroid Interpretation":

    st.title(
        "🦋 Thyroid → Cardiovascular Interpretation"
    )

    st.write(
        """
        The project's research analysis investigated whether thyroid
        biomarkers were independently associated with model-predicted CVD.
        """

    )

    st.subheader(
        "Validated Research Findings"
    )

    thyroid_results = pd.DataFrame(
        [
            {
                "Biomarker": "TSH",
                "Finding": "Not independently significant",
                "Interpretation":
                    "TSH was not independently associated with "
                    "model-predicted CVD after adjustment."
            },
            {
                "Biomarker": "Free T3",
                "Finding": "Not independently significant",
                "Interpretation":
                    "Free T3 was not independently associated with "
                    "model-predicted CVD after adjustment."
            },
            {
                "Biomarker": "Free T4",
                "Finding":
                    "Unadjusted association only",
                "Interpretation":
                    "Free T4 was higher in the predicted-CVD group, "
                    "but the association did not remain independently "
                    "significant after adjustment."
            },
            {
                "Biomarker": "TPO",
                "Finding": "Not independently significant",
                "Interpretation":
                    "TPO was not independently associated with "
                    "model-predicted CVD after adjustment."
            },
            {
                "Biomarker": "FT3/FT4 Ratio",
                "Finding":
                    "Unadjusted association only",
                "Interpretation":
                    "The ratio was lower in the predicted-CVD group "
                    "and associated with continuous CVD probability, "
                    "but the adjusted association did not remain "
                    "independently significant."
            },
        ]
    )

    st.dataframe(
        thyroid_results,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "How Thyroid Dysfunction Can Interact With Cardiovascular Health"
    )

    st.write(
        """
        Thyroid dysfunction can interact with cardiovascular health through
        several physiological pathways. Depending on the thyroid state,
        these can include effects on lipid metabolism, vascular resistance,
        blood pressure, cardiac workload and rhythm.

        However, the project's statistical analysis does NOT establish
        that thyroid abnormalities directly cause CVD in the study population.
        """
    )

    st.markdown(
        """
        **Hypothyroid-pattern pathway**

        Reduced thyroid hormone activity can be associated with:

        `↓ thyroid hormone → altered lipid metabolism → higher atherogenic
        lipid burden → atherosclerotic risk`

        and

        `↓ thyroid hormone → increased vascular resistance → blood-pressure
        changes → cardiovascular workload`

        **Hyperthyroid-pattern pathway**

        Increased thyroid hormone activity can be associated with:

        `↑ thyroid hormone → ↑ heart rate + ↑ contractility →
        increased cardiac workload`

        and may increase susceptibility to rhythm disturbances such as
        atrial fibrillation in susceptible individuals.
        """
    )

    st.warning(
        """
        These physiological pathways are background medical interpretation.
        They should not be used to diagnose a user from this dashboard alone.
        """
    )


# =============================================================================
# PAGE 4 — RESEARCH FINDINGS
# =============================================================================

elif page == "📊 Research Findings":

    st.title(
        "📊 Project Research Findings"
    )

    # -------------------------------------------------------------------------
    # Thyroid population
    # -------------------------------------------------------------------------

    thyroid_df = df.copy()

    if "Thyroid_Problem" in thyroid_df.columns:

        thyroid_mask = (
            pd.to_numeric(
                thyroid_df["Thyroid_Problem"],
                errors="coerce"
            )
            .fillna(0)
            == 1
        )

        thyroid_df = thyroid_df[
            thyroid_mask
        ].copy()

    elif "Thyroid_Status" in thyroid_df.columns:

        thyroid_df = thyroid_df[
            thyroid_df["Thyroid_Status"]
            .astype(str)
            .str.contains(
                "Thyroid Problem",
                case=False,
                na=False
            )
        ].copy()

    st.subheader(
        "Thyroid-Problem Population"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Participants",
        len(thyroid_df)
    )

    if "CVD_Prediction" in thyroid_df.columns:

        predicted = int(
            pd.to_numeric(
                thyroid_df["CVD_Prediction"],
                errors="coerce"
            ).fillna(0).sum()
        )

        c2.metric(
            "Model-Predicted CVD",
            predicted
        )

        c3.metric(
            "Percentage",
            f"{predicted / len(thyroid_df) * 100:.2f}%"
            if len(thyroid_df) > 0
            else "0%"
        )

    # -------------------------------------------------------------------------
    # Risk factors
    # -------------------------------------------------------------------------

    st.subheader(
        "Research Risk-Factor Pattern"
    )

    research_factors = []

    factor_definitions = {
        "High blood pressure": "High_BP",
        "High cholesterol": "High_Cholesterol",
        "Low HDL": "Low_HDL",
        "Obesity": "Obesity",
        "Smoking-related risk": "Smoking_Risk",
    }

    for label, column in factor_definitions.items():

        if column not in thyroid_df.columns:
            continue

        values = pd.to_numeric(
            thyroid_df[column],
            errors="coerce"
        ).fillna(0)

        count = int(
            (values == 1).sum()
        )

        percentage = (
            count / len(thyroid_df) * 100
            if len(thyroid_df) > 0
            else 0
        )

        research_factors.append(
            {
                "Risk Factor": label,
                "Participants": count,
                "Percentage": round(
                    percentage,
                    2
                )
            }
        )

    if research_factors:

        research_factor_df = pd.DataFrame(
            research_factors
        )

        st.dataframe(
            research_factor_df,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            research_factor_df.set_index(
                "Risk Factor"
            )["Percentage"]
        )

    # -------------------------------------------------------------------------
    # Risk burden
    # -------------------------------------------------------------------------

    st.subheader(
        "Risk-Factor Burden"
    )

    if "Risk_Factor_Count" in thyroid_df.columns:

        burden = (
            pd.to_numeric(
                thyroid_df["Risk_Factor_Count"],
                errors="coerce"
            )
            .value_counts()
            .sort_index()
            .rename_axis(
                "Risk Factor Count"
            )
            .reset_index(
                name="Participants"
            )
        )

        st.dataframe(
            burden,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            burden.set_index(
                "Risk Factor Count"
            )["Participants"]
        )

    # -------------------------------------------------------------------------
    # Biomarker comparisons
    # -------------------------------------------------------------------------

    st.subheader(
        "Thyroid Biomarkers and Model-Predicted CVD"
    )

    if "CVD_Prediction" in thyroid_df.columns:

        biomarker_rows = []

        for feature in [
            "TSH",
            "Free_T3",
            "Free_T4",
            "TPO",
            "FT3_FT4_Ratio"
        ]:

            if feature not in thyroid_df.columns:
                continue

            values = pd.to_numeric(
                thyroid_df[feature],
                errors="coerce"
            )

            no_cvd = values[
                pd.to_numeric(
                    thyroid_df["CVD_Prediction"],
                    errors="coerce"
                ) == 0
            ]

            predicted_cvd_values = values[
                pd.to_numeric(
                    thyroid_df["CVD_Prediction"],
                    errors="coerce"
                ) == 1
            ]

            biomarker_rows.append(
                {
                    "Feature": feature,
                    "Mean_No_CVD":
                        round(
                            no_cvd.mean(),
                            4
                        ),
                    "Mean_Predicted_CVD":
                        round(
                            predicted_cvd_values.mean(),
                            4
                        ),
                    "Difference":
                        round(
                            predicted_cvd_values.mean()
                            -
                            no_cvd.mean(),
                            4
                        )
                }
            )

        if biomarker_rows:

            st.dataframe(
                pd.DataFrame(
                    biomarker_rows
                ),
                use_container_width=True,
                hide_index=True
            )

    # -------------------------------------------------------------------------
    # Research conclusion
    # -------------------------------------------------------------------------

    st.subheader(
        "Research Conclusion"
    )

    st.success(
        """
        The project's validated analysis indicates that conventional
        cardiovascular risk factors are important within the thyroid-problem
        population.

        Thyroid biomarkers showed some unadjusted associations, particularly
        Free T4 and the FT3/FT4 ratio, but none remained independently
        significant after adjustment and FDR correction.

        Therefore, the dashboard should focus on identifying overall
        cardiovascular risk, highlighting modifiable risk factors and
        supporting appropriate thyroid/cardiovascular follow-up rather than
        claiming that thyroid abnormalities cause CVD.
        """
    )


# =============================================================================
# PAGE 5 — ML MODEL
# =============================================================================

elif page == "🤖 ML Model":

    st.title(
        "🤖 CVD Machine Learning Model"
    )

    st.write(
        """
        The CVD assessment uses a Random Forest classifier trained on the
        project's clinical variables with Cardiovascular_Disease as the
        target.
        """
    )

    # -------------------------------------------------------------------------
    # MODEL METRICS
    # -------------------------------------------------------------------------

    st.subheader(
        "Model Performance"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Accuracy",
        f"{model_metrics['Accuracy']:.3f}"
    )

    c2.metric(
        "Precision",
        f"{model_metrics['Precision']:.3f}"
    )

    c3.metric(
        "Recall",
        f"{model_metrics['Recall']:.3f}"
    )

    c4.metric(
        "F1",
        f"{model_metrics['F1']:.3f}"
    )

    c5.metric(
        "ROC-AUC",
        f"{model_metrics['ROC_AUC']:.3f}"
    )

    st.divider()

    # -------------------------------------------------------------------------
    # MODEL INPUT FEATURES
    # -------------------------------------------------------------------------

    st.subheader(
        "Model Input Variables"
    )

    # MODEL_FEATURES contains the exact variables used by the
    # Random Forest model.
    #
    # Roles are generated automatically so that the number of
    # roles always matches the number of model features.

    def get_feature_role(feature):

        feature_lower = str(feature).lower()

        # Demographic variables
        if any(x in feature_lower for x in [
            "age",
            "sex",
            "gender",
            "race"
        ]):
            return "Demographic"

        # Anthropometric variables
        if any(x in feature_lower for x in [
            "bmi",
            "weight",
            "height",
            "waist"
        ]):
            return "Anthropometric"

        # Thyroid variables
        if any(x in feature_lower for x in [
            "tsh",
            "free_t3",
            "free_t4",
            "ft3",
            "ft4",
            "tpo",
            "thyroid"
        ]):
            return "Thyroid"

        # Cardiovascular variables
        if any(x in feature_lower for x in [
            "hdl",
            "cholesterol",
            "sys_bp",
            "dia_bp",
            "blood_pressure",
            "systolic",
            "diastolic",
            "hypertension",
            "cardiovascular"
        ]):
            return "Cardiovascular"

        # Medical history variables
        if any(x in feature_lower for x in [
            "diabetes",
            "medical",
            "disease",
            "history"
        ]):
            return "Medical history"

        # Lifestyle variables
        if any(x in feature_lower for x in [
            "smok",
            "alcohol",
            "exercise",
            "physical",
            "lifestyle"
        ]):
            return "Lifestyle"

        return "Clinical"


    # Create exactly one role for every model feature.
    feature_table = pd.DataFrame(
        {
            "Feature": MODEL_FEATURES,
            "Role": [
                get_feature_role(feature)
                for feature in MODEL_FEATURES
            ]
        }
    )

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        f"The model uses {len(MODEL_FEATURES)} clinical input features."
    )

    # -------------------------------------------------------------------------
    # FEATURE IMPORTANCE
    # -------------------------------------------------------------------------

    st.subheader(
        "Random Forest Feature Importance"
    )

    if not feature_importance.empty:

        top_features = (
            feature_importance
            .head(15)
            .copy()
        )

        st.dataframe(
            top_features,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            top_features
            .set_index("Feature")[
                "Importance"
            ]
        )

    else:

        st.warning(
            "Feature importance could not be calculated for the current model."
        )

    # -------------------------------------------------------------------------
    # MODEL INTEGRITY
    # -------------------------------------------------------------------------

    st.subheader(
        "Model Integrity"
    )

    integrity = pd.DataFrame(
        {
            "Check": [
                "CVD_Probability used as predictor",
                "CVD_Prediction used as predictor",
                "CVD_Risk_Category used as predictor",
                "Cardiovascular_Disease modified",
                "Model type"
            ],
            "Result": [
                "NO",
                "NO",
                "NO",
                "NO",
                "Random Forest"
            ]
        }
    )

    st.dataframe(
        integrity,
        use_container_width=True,
        hide_index=True
    )

    # -------------------------------------------------------------------------
    # MODEL DESCRIPTION
    # -------------------------------------------------------------------------

    st.subheader(
        "Model Description"
    )

    st.info(
        """
        The Random Forest model estimates cardiovascular disease probability
        from the predefined clinical input variables.

        Derived prediction fields such as CVD_Probability, CVD_Prediction
        and CVD_Risk_Category are not used as predictors.

        Cardiovascular_Disease is retained as the original prediction target
        and is not modified during model preparation.
        """
    )

    # -------------------------------------------------------------------------
    # MEDICAL INTERPRETATION
    # -------------------------------------------------------------------------

    st.subheader(
        "Interpretation"
    )

    st.warning(
        """
        The model output represents a statistical risk estimate rather than
        a clinical diagnosis.

        A predicted probability should be interpreted together with the
        person's medical history, examination, laboratory results and other
        appropriate clinical information.
        """
    )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "Thyroid-CVD Intelligent Risk Assessment System | "
    "Research + Machine Learning Dashboard"
)

st.caption(
    "For research/educational risk assessment only — "
    "not a substitute for professional medical evaluation."
)
