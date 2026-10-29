from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
TARGET = "Cardiovascular_Disease"

# Use explicit feature names; never expose encoded category codes in the form.
MODEL_FEATURES = [
    "Age", "Gender", "Race", "BMI", "Height", "Weight", "Waist",
    "TSH", "Free_T3", "Free_T4", "TPO", "HDL", "Total_Cholesterol",
    "Diabetes", "Smoking_100_Cigarettes", "Thyroid_Diagnosis",
    "SYS_BP", "DIA_BP"
]
NUMERIC_FEATURES = [
    "Age", "BMI", "Height", "Weight", "Waist", "TSH", "Free_T3",
    "Free_T4", "TPO", "HDL", "Total_Cholesterol", "SYS_BP", "DIA_BP"
]
CATEGORICAL_FEATURES = [
    "Gender", "Race", "Diabetes", "Smoking_100_Cigarettes", "Thyroid_Diagnosis"
]

DATA_CANDIDATES = [
    OUTPUT_DIR / "step5_integrated_thyroid_cvd_dataset.csv",
    OUTPUT_DIR / "step11_dashboard_ready_data.csv",
    OUTPUT_DIR / "step10_dashboard_ready_data.csv",
    OUTPUT_DIR / "step9_dashboard_ready_data.csv",
    BASE_DIR / "cleaned_dataset.csv",
    BASE_DIR / "master_dataset.csv",
    OUTPUT_DIR / "cleaned_dataset.csv",
]

app = Flask(__name__)
_state = {"data": None, "path": None, "model": None, "metrics": None, "error": None}


def normalize_target(series):
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().all() and set(values.unique()).issubset({0, 1}):
        return values.astype(int)
    mapping = {
        "0": 0, "1": 1, "no": 0, "yes": 1, "false": 0, "true": 1,
        "negative": 0, "positive": 1, "absent": 0, "present": 1
    }
    result = series.astype(str).str.strip().str.lower().map(mapping)
    if result.isna().any():
        raise ValueError("CVD target must contain binary 0/1 or yes/no labels.")
    return result.astype(int)


def find_dataset():
    candidates = []
    for path in DATA_CANDIDATES:
        if path.exists():
            try:
                df = pd.read_csv(path)
                if TARGET in df.columns:
                    candidates.append((len(df), path, df))
            except Exception as exc:
                warnings.warn(f"Could not read {path}: {exc}")
    if not candidates:
        raise FileNotFoundError(
            "No candidate CSV containing Cardiovascular_Disease was found. "
            "Check DATA_CANDIDATES in server.py and your output folder."
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def clean_category(series, field):
    """Normalize obvious numeric category encodings only when a mapping is defensible."""
    s = series.copy()
    # Standardize common yes/no labels and boolean-like numbers.
    text = s.astype(str).str.strip()
    lower = text.str.lower()
    yes_no = {
        "yes": "Yes", "true": "Yes", "1": "Yes", "present": "Yes", "positive": "Yes",
        "no": "No", "false": "No", "0": "No", "absent": "No", "negative": "No"
    }
    if field in {"Diabetes", "Smoking_100_Cigarettes"}:
        return lower.map(yes_no).fillna(text)
    if field == "Gender":
        gender_map = {
            "f": "Female", "female": "Female", "woman": "Female",
            "m": "Male", "male": "Male", "man": "Male",
            "0": "Female", "1": "Male"
        }
        return lower.map(gender_map).fillna(text)
    # Do not guess meanings for encoded race/diagnosis categories.
    return text.replace({"nan": np.nan, "None": np.nan})


def prepare_data(df):
    data = df[MODEL_FEATURES + [TARGET]].copy()
    for col in NUMERIC_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        data[col] = clean_category(data[col], col)
    data[TARGET] = normalize_target(data[TARGET])
    data = data.dropna(subset=[TARGET])
    return data


def build_model():
    path, raw = find_dataset()
    missing = [c for c in MODEL_FEATURES + [TARGET] if c not in raw.columns]
    if missing:
        raise ValueError(
            f"Dataset {path} is missing required columns: {missing}. "
            "Check that the selected dataset has the expected 18 features and target."
        )
    data = prepare_data(raw)
    if len(data) < 10 or data[TARGET].nunique() < 2:
        raise ValueError("The selected dataset needs at least 10 usable rows and both CVD target classes.")

    X, y = data[MODEL_FEATURES], data[TARGET]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES)
    ])
    pipe = Pipeline([
        ("preprocessor", prep),
        ("model", RandomForestClassifier(
            n_estimators=400, random_state=42, class_weight="balanced",
            min_samples_leaf=2, n_jobs=-1
        ))
    ])
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "records": int(len(raw)),
        "test_records": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probs)) if y_test.nunique() == 2 else None,
        "dataset": str(path.relative_to(BASE_DIR)) if path.is_relative_to(BASE_DIR) else str(path)
    }
    _state.update(data=data, path=path, model=pipe, metrics=metrics, error=None)
    return _state


def ensure_model():
    if _state["model"] is None and _state["error"] is None:
        try:
            build_model()
        except Exception as exc:
            _state["error"] = str(exc)
    return _state


def humanize_categories(series, field):
    values = series.dropna().astype(str).str.strip()
    values = [v for v in values.unique().tolist() if v and v.lower() != "nan"]
    if field == "Gender":
        return sorted(set(clean_category(pd.Series(values), field).tolist()))
    if field in {"Diabetes", "Smoking_100_Cigarettes"}:
        return sorted(set(clean_category(pd.Series(values), field).tolist()))
    return sorted(values)


def get_metadata():
    state = ensure_model()
    if state["error"]:
        raise RuntimeError(state["error"])
    df = state["data"]
    options = {col: humanize_categories(df[col], col) for col in CATEGORICAL_FEATURES}
    # Human-readable defaults for categorical values if present; no numeric-code labels for unknown encodings.
    return {
        "options": options,
        "metrics": state["metrics"],
        "categorical_fields": CATEGORICAL_FEATURES,
        "numeric_fields": NUMERIC_FEATURES,
        "feature_count": len(MODEL_FEATURES)
    }


def parse_num(payload, key, min_value=None, max_value=None):
    try:
        val = float(payload[key])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"Enter a valid value for {key.replace('_', ' ')}.")
    if not np.isfinite(val):
        raise ValueError(f"{key.replace('_', ' ')} must be a valid finite number.")
    if min_value is not None and val < min_value:
        raise ValueError(f"{key.replace('_', ' ')} must be at least {min_value}.")
    if max_value is not None and val > max_value:
        raise ValueError(f"{key.replace('_', ' ')} must be no more than {max_value}.")
    return val


def app_risk_category(prob):
    if prob < 0.30:
        return "Low"
    if prob < 0.50:
        return "Moderate"
    if prob < 0.70:
        return "High"
    return "Very High"


def screening_flags(row, prob):
    flags = []
    if row["Age"] >= 45:
        flags.append("Age is a factor to discuss with a healthcare professional.")
    if row["BMI"] >= 30:
        flags.append("BMI is in the obesity range.")
    if row["HDL"] < 40:
        flags.append("HDL is below 40 in the entered dataset units.")
    if row["Total_Cholesterol"] >= 200:
        flags.append("Total cholesterol is 200 or higher in the entered dataset units.")
    if row["Diabetes"] == "Yes":
        flags.append("Diabetes is marked as present.")
    if row["Smoking_100_Cigarettes"] == "Yes":
        flags.append("Smoking history is marked as present.")
    if row["SYS_BP"] >= 130 or row["DIA_BP"] >= 80:
        flags.append("Blood pressure is elevated relative to common adult screening thresholds.")
    if not flags:
        flags.append("No risk factors were flagged by these simple rules; this does not mean there is no risk.")
    diagnosis = row["Thyroid_Diagnosis"].lower()
    thyroid_flag = any(k in diagnosis for k in ("hypothyroid", "hyperthyroid", "thyroid", "hashimoto", "graves", "goiter", "thyro"))
    tsh_flag = row["TSH"] < 0.4 or row["TSH"] > 4.0
    return flags, thyroid_flag, tsh_flag


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/assessment")
def assessment_page():
    return render_template("assessment.html")


@app.route("/research")
def research_page():
    return render_template("research.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/metadata")
def metadata():
    try:
        return jsonify(get_metadata())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/assess", methods=["POST"])
def assess():
    state = ensure_model()
    if state["error"]:
        return jsonify({"error": state["error"]}), 500
    payload = request.get_json(silent=True) or {}
    try:
        row = {
            "Age": parse_num(payload, "Age", 18, 120),
            "Gender": str(payload.get("Gender", "")).strip(),
            "Race": str(payload.get("Race", "")).strip(),
            "BMI": parse_num(payload, "BMI", 10, 80),
            "Height": parse_num(payload, "Height", 80, 250),
            "Weight": parse_num(payload, "Weight", 20, 350),
            "Waist": parse_num(payload, "Waist", 20, 250),
            "TSH": parse_num(payload, "TSH", 0, 100),
            "Free_T3": parse_num(payload, "Free_T3", 0, 50),
            "Free_T4": parse_num(payload, "Free_T4", 0, 50),
            "TPO": parse_num(payload, "TPO", 0, 10000),
            "HDL": parse_num(payload, "HDL", 0, 500),
            "Total_Cholesterol": parse_num(payload, "Total_Cholesterol", 0, 1000),
            "Diabetes": str(payload.get("Diabetes", "")).strip(),
            "Smoking_100_Cigarettes": str(payload.get("Smoking_100_Cigarettes", "")).strip(),
            "Thyroid_Diagnosis": str(payload.get("Thyroid_Diagnosis", "")).strip(),
            "SYS_BP": parse_num(payload, "SYS_BP", 50, 300),
            "DIA_BP": parse_num(payload, "DIA_BP", 20, 200),
        }
        for field in CATEGORICAL_FEATURES:
            if not row[field]:
                raise ValueError(f"Choose {field.replace('_', ' ')}.")
        allowed = get_metadata()["options"]
        for field in CATEGORICAL_FEATURES:
            if row[field] not in allowed[field]:
                raise ValueError(f"Choose a valid {field.replace('_', ' ')} option from the dataset.")
        if row["DIA_BP"] >= row["SYS_BP"]:
            raise ValueError("Diastolic pressure should be lower than systolic pressure.")
        X = pd.DataFrame([row], columns=MODEL_FEATURES)
        prob = float(state["model"].predict_proba(X)[0, 1])
        prediction = int(prob >= 0.5)
        factors, thyroid_flag, tsh_flag = screening_flags(row, prob)
        recommendations = []
        if prob >= 0.7:
            recommendations.append("The model-estimated probability is high; seek a professional cardiovascular evaluation.")
        elif prob >= 0.5:
            recommendations.append("Consider a professional cardiovascular risk assessment.")
        else:
            recommendations.append("Continue routine preventive care and discuss personal risk factors with a clinician.")
        if thyroid_flag or tsh_flag:
            recommendations.append("Thyroid-related inputs triggered a screening flag; discuss the results with a qualified healthcare professional.")
        return jsonify({
            "probability": round(prob, 4),
            "probability_percent": round(prob * 100, 1),
            "prediction": prediction,
            "prediction_label": "Model class 1" if prediction else "Model class 0",
            "risk_category": app_risk_category(prob),
            "risk_factors": factors,
            "thyroid_flag": thyroid_flag,
            "tsh_flag": tsh_flag,
            "recommendations": recommendations,
            "notice": "Research and educational use only. This output is not a diagnosis or a clinically validated individual risk estimate."
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Assessment failed")
        return jsonify({"error": f"Assessment failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
