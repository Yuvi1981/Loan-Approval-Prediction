"""
Inference module for loan-telemetry.
Provides single-application inference pipeline with schema validation,
domain feature engineering, real-time SHAP explainability, operational
telemetry logging, and sub-500ms response guarantees.
"""

import json
import logging
import time
from typing import Any

import joblib
import pandas as pd

from src.config import get_path
from src.explain import LoanExplainer
from src.features import add_features
from src.telemetry import log_prediction

logger = logging.getLogger(__name__)

# Global singletons for fast re-use
_MODEL_PIPELINE = None
_EXPLAINER = None
_MODEL_META = None


def load_model_and_meta() -> tuple[Any, dict[str, Any]]:
    """Load model pipeline and metadata from disk with caching."""
    global _MODEL_PIPELINE, _MODEL_META

    if _MODEL_PIPELINE is None:
        model_path = get_path("best_model")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}. Run training first.")
        _MODEL_PIPELINE = joblib.load(model_path)

    if _MODEL_META is None:
        meta_path = get_path("model_meta")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                _MODEL_META = json.load(f)
        else:
            _MODEL_META = {"version": "1.0.0", "default_threshold": 0.5}

    return _MODEL_PIPELINE, _MODEL_META


def get_explainer() -> LoanExplainer:
    """Initialize or retrieve cached SHAP LoanExplainer."""
    global _EXPLAINER
    if _EXPLAINER is None:
        model, _ = load_model_and_meta()
        # Optionally load a small background slice from processed train data
        try:
            train_path = get_path("processed_train")
            bg_data = pd.read_csv(train_path).head(40) if train_path.exists() else None
        except Exception:
            bg_data = None
        _EXPLAINER = LoanExplainer(model, background_data=bg_data)
    return _EXPLAINER


def validate_input(raw_input: dict[str, Any]) -> dict[str, Any]:
    """Validate and sanitize raw input fields for a loan application."""
    validated = raw_input.copy()

    # Defaults for missing entries
    defaults = {
        "ApplicantIncome": 4000.0,
        "CoapplicantIncome": 0.0,
        "LoanAmount": 120.0,
        "Loan_Amount_Term": 360.0,
        "Credit_History": 1.0,
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "Property_Area": "Urban",
    }

    for key, def_val in defaults.items():
        if key not in validated or validated[key] is None or validated[key] == "":
            validated[key] = def_val

    # Coerce numerics safely
    for num_key in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]:
        try:
            validated[num_key] = float(validated[num_key])
        except (ValueError, TypeError):
            validated[num_key] = defaults[num_key]

    return validated


def predict_one(
    raw_input: dict[str, Any],
    threshold: float | None = None,
    log_to_telemetry: bool = True,
    true_label: int | None = None,
) -> dict[str, Any]:
    """Execute full prediction pipeline for a single loan application.

    Steps:
    1. Validate input schema and types.
    2. Engineer financial features (Total_Income, EMI, Balance_Income, etc.).
    3. Run model pipeline inference and measure latency.
    4. Compute top-5 SHAP factor explanations.
    5. Log prediction details to SQLite telemetry database.

    Args:
        raw_input: Dictionary with application fields.
        threshold: Decision threshold (defaults to config or model metadata).
        log_to_telemetry: Whether to record event in SQLite database.
        true_label: Optional actual ground truth label.

    Returns:
        Dict[str, Any]: Decision, probability, latency, top SHAP factors, and prediction ID.
    """
    # 1. Validation
    valid_dict = validate_input(raw_input)
    df_raw = pd.DataFrame([valid_dict])

    # 2. Model & Metadata Loading (ensures cached in memory before timing)
    model, meta = load_model_and_meta()
    model_version = meta.get("version", "1.0.0")
    explainer = get_explainer()

    # Start timing actual inference pipeline
    start_time = time.perf_counter()

    # 3. Feature Engineering
    df_featured = add_features(df_raw)

    if threshold is None:
        threshold = float(meta.get("default_threshold", 0.50))

    # Exclude sensitive attributes (Gender, Married) and IDs from model input
    exclude_cols = ["Loan_ID", "Loan_Status", "Gender", "Married"]
    model_input_df = df_featured[[c for c in df_featured.columns if c not in exclude_cols]]

    # 4. Predict probability and decision
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(model_input_df)[0, 1])
    else:
        prob = float(model.predict(model_input_df)[0])

    decision = 1 if prob >= threshold else 0

    # 5. SHAP Explanations
    top_factors = explainer.explain_one(model_input_df, top_k=5)

    # Latency (inference + SHAP + features)
    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    # 6. Telemetry Logging
    pred_id = None
    if log_to_telemetry:
        try:
            pred_id = log_prediction(
                inputs=valid_dict,
                probability=prob,
                decision=decision,
                threshold=threshold,
                latency_ms=latency_ms,
                model_version=model_version,
                top_factors=top_factors,
                true_label=true_label,
            )
        except Exception as e:
            logger.error("Failed to log prediction to telemetry: %s", e)

    return {
        "prediction_id": pred_id,
        "decision": decision,
        "status_label": "Approved" if decision == 1 else "Rejected",
        "probability": round(prob, 4),
        "threshold": threshold,
        "top_factors": top_factors,
        "latency_ms": latency_ms,
        "model_version": model_version,
        "model_name": meta.get("model_name", "BestModel"),
    }


if __name__ == "__main__":
    sample_app = {
        "ApplicantIncome": 5849,
        "CoapplicantIncome": 0,
        "LoanAmount": 128,
        "Loan_Amount_Term": 360,
        "Credit_History": 1.0,
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "Property_Area": "Urban",
    }
    res = predict_one(sample_app)
    print("Sample Prediction Result:")
    print(json.dumps(res, indent=2))
