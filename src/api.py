"""
FastAPI Backend Server for loan-telemetry.
Exposes REST APIs for real-time loan prediction, MLflow model comparisons,
live inference telemetry, statistical drift surveillance, and fairness auditing,
while serving the modern frontend web application.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import get_path
from src.predict import load_model_and_meta, predict_one
from src.telemetry import (
    clear_telemetry,
    read_predictions,
    summary_stats,
    update_true_label,
)

logger = logging.getLogger("loan_telemetry_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Loan Approval Prediction & ML Telemetry API",
    description="Production-grade ML API providing loan screening, local SHAP explainability, telemetry logging, drift surveillance, and fairness auditing.",
    version="1.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
class LoanApplicationRequest(BaseModel):
    ApplicantIncome: float = Field(..., example=5000.0, description="Monthly applicant income in USD")
    CoapplicantIncome: float = Field(0.0, example=1500.0, description="Monthly co-applicant income in USD")
    LoanAmount: float = Field(..., example=140.0, description="Loan amount in thousands USD")
    Loan_Amount_Term: float = Field(360.0, example=360.0, description="Loan duration in months")
    Credit_History: float = Field(1.0, example=1.0, description="Credit criteria met: 1.0 = Yes, 0.0 = No")
    Dependents: str = Field("0", example="0", description="Number of dependents ('0', '1', '2', '3+')")
    Education: str = Field("Graduate", example="Graduate", description="'Graduate' or 'Not Graduate'")
    Self_Employed: str = Field("No", example="No", description="'Yes' or 'No'")
    Property_Area: str = Field("Urban", example="Semiurban", description="'Urban', 'Semiurban', or 'Rural'")
    Gender: Optional[str] = Field("Male", example="Male", description="Gender (audited for fairness, not model input)")
    Married: Optional[str] = Field("Yes", example="Yes", description="Marital status (audited for fairness, not model input)")
    threshold: Optional[float] = Field(None, example=0.50, description="Custom decision threshold (0.10 - 0.90)")


@app.get("/api/health")
def health_check():
    """System health check and champion model status."""
    try:
        _, meta = load_model_and_meta()
        model_name = meta.get("model_name", "XGBoost")
        version = meta.get("version", "1.0.0")
    except Exception:
        model_name = "Not Loaded"
        version = "Unknown"

    return {
        "status": "healthy",
        "service": "loan-telemetry-backend",
        "active_model": model_name,
        "model_version": version,
    }


@app.post("/api/predict")
def predict_endpoint(payload: LoanApplicationRequest):
    """Screen a single loan application, compute probability, top SHAP factors, and log telemetry."""
    try:
        input_dict = payload.dict()
        threshold = input_dict.pop("threshold", None)
        result = predict_one(input_dict, threshold=threshold, log_to_telemetry=True)
        return {
            "success": True,
            "data": result,
        }
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
def get_models_leaderboard():
    """Retrieve full model benchmark leaderboard and champion model specifications."""
    meta_path = Path("models/model_meta.json")
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Model metadata not found. Run training first.")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Leaderboard table
    leaderboard = [
        {"model": "XGBoost (Champion)", "type": "Gradient Boosting", "cv_f1": 0.8524, "accuracy": 0.8618, "precision": 0.8469, "recall": 0.9765, "f1": 0.9071, "roc_auc": 0.8359},
        {"model": "StackingClassifier", "type": "Stacking Ensemble", "cv_f1": 0.8622, "accuracy": 0.8618, "precision": 0.8542, "recall": 0.9647, "f1": 0.9061, "roc_auc": 0.8647},
        {"model": "DecisionTree", "type": "Baseline Tree", "cv_f1": 0.8582, "accuracy": 0.8618, "precision": 0.8617, "recall": 0.9529, "f1": 0.9050, "roc_auc": 0.7913},
        {"model": "LogisticRegression", "type": "Linear Baseline", "cv_f1": 0.8626, "accuracy": 0.8537, "precision": 0.8317, "recall": 0.9882, "f1": 0.9032, "roc_auc": 0.8443},
        {"model": "GaussianNaiveBayes", "type": "Probabilistic Baseline", "cv_f1": 0.8378, "accuracy": 0.8537, "precision": 0.8317, "recall": 0.9882, "f1": 0.9032, "roc_auc": 0.8437},
        {"model": "VotingClassifier_Soft", "type": "Soft Ensemble", "cv_f1": 0.8626, "accuracy": 0.8537, "precision": 0.8317, "recall": 0.9882, "f1": 0.9032, "roc_auc": 0.8734},
        {"model": "RandomForest_ClassWeighted", "type": "Imbalance (Balanced)", "cv_f1": 0.8486, "accuracy": 0.8537, "precision": 0.8681, "recall": 0.9294, "f1": 0.8977, "roc_auc": 0.8709},
        {"model": "RandomForest", "type": "Bagging Ensemble", "cv_f1": 0.8672, "accuracy": 0.8455, "precision": 0.8367, "recall": 0.9647, "f1": 0.8962, "roc_auc": 0.8789},
        {"model": "XGBoost_ClassWeighted", "type": "Imbalance (Pos Weight)", "cv_f1": 0.8273, "accuracy": 0.8537, "precision": 0.8851, "recall": 0.9059, "f1": 0.8953, "roc_auc": 0.8601},
        {"model": "RandomForest_Tuned", "type": "Tuned GridSearchCV", "cv_f1": 0.8731, "accuracy": 0.8293, "precision": 0.8265, "recall": 0.9529, "f1": 0.8852, "roc_auc": 0.8799},
        {"model": "KNN", "type": "Distance Baseline", "cv_f1": 0.8497, "accuracy": 0.8293, "precision": 0.8478, "recall": 0.9176, "f1": 0.8814, "roc_auc": 0.8209},
        {"model": "RandomForest_SMOTE", "type": "Imbalance (SMOTE)", "cv_f1": 0.8441, "accuracy": 0.7967, "precision": 0.8409, "recall": 0.8706, "f1": 0.8555, "roc_auc": 0.8396},
    ]

    return {
        "champion": meta,
        "leaderboard": leaderboard,
    }


@app.get("/api/telemetry")
def get_telemetry_data(limit: int = Query(50, ge=5, le=500)):
    """Retrieve aggregate operational statistics and recent transaction logs."""
    stats = summary_stats()
    df_logs = read_predictions(limit=limit)

    records = []
    if not df_logs.empty:
        for _, row in df_logs.iterrows():
            inputs_parsed = {}
            factors_parsed = []
            try:
                inputs_parsed = json.loads(row["inputs_json"])
            except Exception:
                pass
            try:
                factors_parsed = json.loads(row["top_factors_json"])
            except Exception:
                pass

            records.append({
                "id": int(row["id"]),
                "timestamp": row["ts"],
                "decision": int(row["decision"]),
                "decision_label": "Approved" if row["decision"] == 1 else "Rejected",
                "probability": float(row["probability"]),
                "threshold": float(row["threshold"]),
                "latency_ms": float(row["latency_ms"]),
                "model_version": row["model_version"],
                "true_label": int(row["true_label"]) if pd.notna(row["true_label"]) else None,
                "inputs": inputs_parsed,
                "top_factors": factors_parsed,
            })

    # Sanitize stats for JSON compliance
    for k, v in stats.items():
        if isinstance(v, float) and (pd.isna(v) or np.isinf(v)):
            stats[k] = 0.0

    return {
        "stats": stats,
        "recent_transactions": records,
    }


@app.get("/api/telemetry/stats")
def get_telemetry_stats():
    """Retrieve operational telemetry aggregate summary metrics."""
    stats = summary_stats()
    for k, v in stats.items():
        if isinstance(v, float) and (pd.isna(v) or np.isinf(v)):
            stats[k] = 0.0
    return stats


def sanitize_nan(obj: Any) -> Any:
    """Recursively convert float NaN/Inf to None or 0.0 for strict JSON compliance."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_nan(v) for v in obj]
    return obj


@app.get("/api/drift")
def get_drift_data():
    """Retrieve statistical drift surveillance metrics (PSI, KS-test, and performance drop)."""
    drift_path = Path("reports/drift_demo.json")
    if not drift_path.exists():
        raise HTTPException(status_code=404, detail="Drift report not found. Run simulation first.")

    with open(drift_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return sanitize_nan(data)


@app.get("/api/fairness")
def get_fairness_data():
    """Retrieve Fairlearn algorithmic bias and demographic disparity audit results."""
    fairness_path = Path("reports/fairness.json")
    if not fairness_path.exists():
        raise HTTPException(status_code=404, detail="Fairness report not found. Run fairness audit first.")

    with open(fairness_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return sanitize_nan(data)


@app.post("/api/simulate-traffic")
def trigger_traffic_simulation(count: int = Query(25, ge=5, le=100)):
    """Generate and log a batch of realistic test applications."""
    import random

    from scripts.generate_traffic import generate_sample_application

    generated = 0
    for i in range(count):
        app_data = generate_sample_application(random_seed=5000 + i)
        thresh = random.choice([0.45, 0.50, 0.55])
        res = predict_one(app_data, threshold=thresh, log_to_telemetry=True)
        if random.random() < 0.60 and res["prediction_id"] is not None:
            true_lbl = 1 if (app_data["Credit_History"] == 1.0 and random.random() < 0.85) else 0
            update_true_label(res["prediction_id"], true_lbl)
        generated += 1

    return {
        "success": True,
        "generated_count": generated,
        "updated_stats": summary_stats(),
    }


@app.get("/api/data-quality")
def get_data_quality():
    """Retrieve comprehensive data quality auditing report."""
    dq_path = Path("reports/data_quality.json")
    if dq_path.exists():
        with open(dq_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return sanitize_nan(data)

    # Fallback to computing on train.csv
    try:
        from src.data_quality import run_checks
        raw_train_path = get_path("raw_train")
        if raw_train_path.exists():
            df_train = pd.read_csv(raw_train_path)
            res = run_checks(df_train, output_path=dq_path)
            return sanitize_nan(res)
    except Exception as e:
        logger.error("Failed to run data quality checks: %s", e)

    raise HTTPException(status_code=404, detail="Data quality report not found.")


@app.post("/api/telemetry/reset")
def reset_telemetry():
    """Clear all logged prediction records from the SQLite database."""
    try:
        clear_telemetry()
        return {
            "success": True,
            "message": "Telemetry database reset successfully.",
            "stats": summary_stats(),
        }
    except Exception as e:
        logger.error("Failed to reset telemetry: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/telemetry/export")
def export_telemetry():
    """Export all telemetry predictions as a CSV download."""
    try:
        df = read_predictions(limit=5000)
        csv_str = df.to_csv(index=False)
        return Response(
            content=csv_str,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=loan_telemetry_export.csv"},
        )
    except Exception as e:
        logger.error("Failed to export telemetry: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# Mount Static Files for Frontend
frontend_dir = Path("frontend")
frontend_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
