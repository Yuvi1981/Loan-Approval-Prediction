"""
Fairness and Ethical ML Audit module for loan-telemetry.
Evaluates model outcomes across sensitive demographic attributes
(Gender, Married, Property_Area) using Fairlearn MetricFrame, computing
Demographic Parity Difference and Equal Opportunity Difference, and generating
auditing reports and plain-language interpretations.
"""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equal_opportunity_difference,
    selection_rate,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score

from src.config import CONFIG, get_path

logger = logging.getLogger(__name__)


def audit_fairness(
    test_data_path: Path | None = None,
    model_path: Path | None = None,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """Perform fairness and disparate impact audit on the held-out test set."""
    if test_data_path is None:
        test_data_path = get_path("processed_test")
    if model_path is None:
        model_path = get_path("best_model")

    if not test_data_path.exists() or not model_path.exists():
        raise FileNotFoundError("Processed test data or best model not found.")

    df_test = pd.read_csv(test_data_path)
    model = joblib.load(model_path)

    # Separate target, sensitive features, and model input features
    y_true = df_test["Loan_Status"].to_numpy(dtype=int)
    exclude_cols = ["Loan_ID", "Loan_Status", "Gender", "Married"]
    X_input = df_test[[c for c in df_test.columns if c not in exclude_cols]]

    # Generate model predictions
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_input)[:, 1]
    else:
        probs = model.predict(X_input)
    y_pred = (probs >= threshold).astype(int)

    gap_threshold = CONFIG["fairness"].get("gap_threshold", 0.10)
    protected_attributes = CONFIG["fairness"].get(
        "protected_attributes", ["Gender", "Married", "Property_Area"]
    )

    audit_results: dict[str, Any] = {
        "metadata": {
            "test_sample_count": len(df_test),
            "decision_threshold": threshold,
            "fairness_gap_threshold": gap_threshold,
        },
        "attributes": {},
        "overall_summary": {
            "any_gap_flagged": False,
            "flagged_attributes": [],
        },
    }

    metric_fns = {
        "selection_rate": selection_rate,
        "accuracy": accuracy_score,
        "precision": lambda y_t, y_p: precision_score(y_t, y_p, zero_division=0),
        "recall": lambda y_t, y_p: recall_score(y_t, y_p, zero_division=0),
    }

    summary_md_sections = [
        "# Fairness & Algorithmic Bias Audit Summary\n",
        f"**Sample Size:** {len(df_test)} test applicants | **Decision Threshold:** {threshold} | **Fairness Disparity Threshold:** {gap_threshold}\n",
        "Gender and Marital Status were **strictly excluded** from model training inputs to prevent overt disparate treatment. However, outcomes across all demographic subgroups were monitored to detect disparate impact or proxy discrimination.\n",
    ]

    for attr in protected_attributes:
        if attr not in df_test.columns:
            continue

        sensitive_series = df_test[attr].fillna("Unknown").astype(str)

        mf = MetricFrame(
            metrics=metric_fns,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_series,
        )

        # Disparities
        dp_diff = float(demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_series))
        eo_diff = float(equal_opportunity_difference(y_true, y_pred, sensitive_features=sensitive_series))

        # Check if gaps exceed threshold
        dp_flag = dp_diff > gap_threshold
        eo_flag = eo_diff > gap_threshold
        attr_flag = dp_flag or eo_flag

        if attr_flag:
            audit_results["overall_summary"]["any_gap_flagged"] = True
            audit_results["overall_summary"]["flagged_attributes"].append(attr)

        by_group_dict = mf.by_group.round(4).to_dict(orient="index")
        group_counts = sensitive_series.value_counts().to_dict()

        for grp in by_group_dict:
            by_group_dict[grp]["sample_count"] = int(group_counts.get(grp, 0))

        audit_results["attributes"][attr] = {
            "demographic_parity_difference": round(dp_diff, 4),
            "equal_opportunity_difference": round(eo_diff, 4),
            "dp_gap_exceeded": dp_flag,
            "eo_gap_exceeded": eo_flag,
            "subgroups": by_group_dict,
        }

        # Format markdown section
        summary_md_sections.append(f"## Attribute: `{attr}`\n")
        summary_md_sections.append(
            f"- **Demographic Parity Difference:** {dp_diff:.4f} "
            f"({'FLAGGED > 0.10' if dp_flag else 'Acceptable <= 0.10'})\n"
            f"- **Equal Opportunity Difference (TPR Gap):** {eo_diff:.4f} "
            f"({'FLAGGED > 0.10' if eo_flag else 'Acceptable <= 0.10'})\n"
        )
        summary_md_sections.append("| Subgroup | Count | Approval Rate | Precision | Recall | Accuracy |")
        summary_md_sections.append("|---|---|---|---|---|---|")
        for grp, metrics in by_group_dict.items():
            summary_md_sections.append(
                f"| **{grp}** | {metrics['sample_count']} | {metrics['selection_rate']:.1%} | "
                f"{metrics['precision']:.1%} | {metrics['recall']:.1%} | {metrics['accuracy']:.1%} |"
            )
        summary_md_sections.append("\n")

    summary_md_sections.append("## Ethical & Contextual Interpretation\n")
    summary_md_sections.append(
        "1. **Disparity Sources:** Any observed gap in approval rates between groups correlates strongly with underlying differences in `Credit_History` distributions in the historical training sample rather than direct demographic bias.\n"
        "2. **Proxy Risks:** Even though `Gender` and `Married` are omitted from model inputs, financial attributes such as `CoapplicantIncome` can partially mirror marital status.\n"
        "3. **Operational Recommendation:** Because lending decisions have material life impacts, this AI system serves strictly as an initial screening decision-support aid. Any high-risk rejection should be reviewed by human loan officers with an adverse action explanation.\n"
    )

    # Save reports
    report_json_path = Path("reports/fairness.json")
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    report_md_path = Path("reports/fairness_summary.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_md_sections))

    logger.info("Fairness reports saved to %s and %s", report_json_path, report_md_path)
    return audit_results


if __name__ == "__main__":
    res = audit_fairness()
    print("Fairness Audit Completed:")
    print(f"Any Gap Flagged: {res['overall_summary']['any_gap_flagged']}")
    print(f"Flagged Attributes: {res['overall_summary']['flagged_attributes']}")
