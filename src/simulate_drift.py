"""
Drift simulation and reporting script for loan-telemetry.
Simulates demographic and economic shifts (income increase, credit score degradation,
loan amount inflation), evaluates PSI/KS drift, computes the corresponding ML performance
drop, and generates reports/drift_demo.json and visualization charts.
"""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import get_path
from src.drift import generate_drift_report
from src.evaluate import evaluate_predictions
from src.features import add_features

logger = logging.getLogger(__name__)


def create_drifted_data(test_df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Create a realistic shifted dataset simulating economic and demographic drift."""
    rng = np.random.RandomState(random_state)
    drifted = test_df.copy()

    # 1. Income inflation: ApplicantIncome shifted up by 1.6x with noise
    if "ApplicantIncome" in drifted.columns:
        drifted["ApplicantIncome"] = drifted["ApplicantIncome"] * rng.uniform(1.4, 1.8, size=len(drifted))

    # 2. Loan Amount shift: increased loan requests by 1.4x
    if "LoanAmount" in drifted.columns:
        drifted["LoanAmount"] = drifted["LoanAmount"] * rng.uniform(1.2, 1.6, size=len(drifted))

    # 3. Macroeconomic stress: 45% of good credit histories flip to 0.0
    if "Credit_History" in drifted.columns:
        good_mask = drifted["Credit_History"] == 1.0
        flip_indices = drifted[good_mask].sample(frac=0.45, random_state=random_state).index
        drifted.loc[flip_indices, "Credit_History"] = 0.0

    # 4. Property area shift towards Rural
    if "Property_Area" in drifted.columns:
        drifted["Property_Area"] = rng.choice(
            ["Rural", "Semiurban", "Urban"],
            size=len(drifted),
            p=[0.60, 0.20, 0.20],
        )

    # Recompute engineered features to reflect changes
    drifted = add_features(drifted)
    return drifted


def run_drift_simulation() -> dict[str, Any]:
    """Execute drift simulation, score model, produce reports and diagnostic charts."""
    train_path = get_path("processed_train")
    test_path = get_path("processed_test")
    model_path = get_path("best_model")

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Processed datasets not found. Run training first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    model = joblib.load(model_path)

    # 1. Generate drifted dataset
    drifted_df = create_drifted_data(test_df)

    # 2. Compute statistical drift report (PSI and KS-test)
    features_to_check = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Total_Income",
        "EMI",
        "Balance_Income",
        "Loan_to_Income",
        "Log_Total_Income",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area",
    ]
    report_df = generate_drift_report(train_df, drifted_df, features_to_check=features_to_check)

    # 3. Model performance evaluation: Baseline vs Drifted
    exclude_cols = ["Loan_ID", "Loan_Status", "Gender", "Married"]
    X_test_base = test_df[[c for c in test_df.columns if c not in exclude_cols]]
    y_test_base = test_df["Loan_Status"]

    X_test_drift = drifted_df[[c for c in drifted_df.columns if c not in exclude_cols]]
    y_test_drift = drifted_df["Loan_Status"]

    base_probs = model.predict_proba(X_test_base)[:, 1]
    drift_probs = model.predict_proba(X_test_drift)[:, 1]

    base_eval = evaluate_predictions(y_test_base, base_probs, threshold=0.5)
    drift_eval = evaluate_predictions(y_test_drift, drift_probs, threshold=0.5)

    delta_acc = round(drift_eval["accuracy"] - base_eval["accuracy"], 4)
    delta_f1 = round(drift_eval["f1"] - base_eval["f1"], 4)
    delta_auc = round(drift_eval["roc_auc"] - base_eval["roc_auc"], 4)

    # 4. Save results to reports/drift_demo.json
    results = {
        "drift_summary": {
            "monitored_samples": len(drifted_df),
            "features_evaluated": len(report_df),
            "features_with_drift": int(report_df["drift_detected"].sum()),
            "max_psi": float(report_df["psi"].max()),
            "top_drifted_feature": str(report_df.iloc[0]["feature"]),
        },
        "performance_impact": {
            "baseline": {
                "accuracy": base_eval["accuracy"],
                "precision": base_eval["precision"],
                "recall": base_eval["recall"],
                "f1": base_eval["f1"],
                "roc_auc": base_eval["roc_auc"],
            },
            "drifted": {
                "accuracy": drift_eval["accuracy"],
                "precision": drift_eval["precision"],
                "recall": drift_eval["recall"],
                "f1": drift_eval["f1"],
                "roc_auc": drift_eval["roc_auc"],
            },
            "deltas": {
                "accuracy_drop": delta_acc,
                "f1_drop": delta_f1,
                "roc_auc_drop": delta_auc,
            },
        },
        "feature_drift_details": report_df.to_dict(orient="records"),
    }

    out_json = Path("reports/drift_demo.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved drift demo report to %s", out_json)

    # 5. Generate Visual Diagnostic Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: PSI per feature with colored status thresholds
    colors = [
        "#d62728" if psi >= 0.25 else "#ff7f0e" if psi >= 0.10 else "#2ca02c"
        for psi in report_df["psi"]
    ]
    axes[0].barh(report_df["feature"], report_df["psi"], color=colors)
    axes[0].axvline(0.10, color="orange", linestyle="--", label="Moderate Drift (0.10)")
    axes[0].axvline(0.25, color="red", linestyle="--", label="Significant Drift (0.25)")
    axes[0].set_xlabel("Population Stability Index (PSI)", fontweight="bold")
    axes[0].set_title("Feature PSI Drift Analysis", fontweight="bold")
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle=":", alpha=0.5)

    # Right: Metric drop comparison
    metrics_names = ["Accuracy", "F1-Score", "ROC-AUC"]
    base_vals = [base_eval["accuracy"], base_eval["f1"], base_eval["roc_auc"]]
    drift_vals = [drift_eval["accuracy"], drift_eval["f1"], drift_eval["roc_auc"]]

    x = np.arange(len(metrics_names))
    width = 0.35
    axes[1].bar(x - width / 2, base_vals, width, label="Baseline (Reference)", color="#1f77b4")
    axes[1].bar(x + width / 2, drift_vals, width, label="Drifted (Shifted)", color="#d62728")
    axes[1].set_ylabel("Metric Score", fontweight="bold")
    axes[1].set_title("Model Performance Degradation", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics_names, fontweight="bold")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(loc="lower left")
    axes[1].grid(True, linestyle=":", alpha=0.5)

    for i in range(len(metrics_names)):
        axes[1].text(x[i] - width / 2, base_vals[i] + 0.02, f"{base_vals[i]:.2f}", ha="center", fontsize=9)
        axes[1].text(x[i] + width / 2, drift_vals[i] + 0.02, f"{drift_vals[i]:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    chart_path = Path("reports/figures/drift_demo.png")
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    logger.info("Saved drift visualization chart to %s", chart_path)
    return results


if __name__ == "__main__":
    res = run_drift_simulation()
    print("Drift Simulation Completed Successfully!")
    print(f"Top Drifted Feature: {res['drift_summary']['top_drifted_feature']} (PSI: {res['drift_summary']['max_psi']})")
    print(f"Accuracy Drop: {res['performance_impact']['deltas']['accuracy_drop']}")
    print(f"F1 Drop: {res['performance_impact']['deltas']['f1_drop']}")
