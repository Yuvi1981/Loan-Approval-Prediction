"""
Data Quality module for loan-telemetry.
Performs schema validation, missingness analysis, duplicate detection,
and range summaries, saving audit reports to reports/data_quality.json.
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_path

logger = logging.getLogger(__name__)


def run_checks(
    df: pd.DataFrame,
    expected_columns: list[str] | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run comprehensive data quality checks on the provided dataframe.

    Args:
        df: Input pandas DataFrame to check.
        expected_columns: Optional list of expected column names.
        output_path: Optional file path to save the JSON report.

    Returns:
        Dict[str, Any]: Comprehensive data quality report.
    """
    if expected_columns is None:
        expected_columns = [
            "Loan_ID",
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History",
            "Property_Area",
            "Loan_Status",
        ]

    warnings_list: list[str] = []
    total_rows = len(df)
    total_cols = len(df.columns)

    # 1. Duplicates check
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        warning_msg = f"Detected {dup_count} duplicate rows."
        warnings_list.append(warning_msg)
        warnings.warn(warning_msg)

    # 2. Schema / Expected columns check
    missing_cols = [col for col in expected_columns if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in expected_columns]
    schema_ok = len(missing_cols) == 0
    if not schema_ok:
        warning_msg = f"Missing expected columns: {missing_cols}"
        warnings_list.append(warning_msg)
        warnings.warn(warning_msg)

    # 3. Missing values analysis
    missing_counts: dict[str, int] = {}
    missing_percentages: dict[str, float] = {}
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        missing_counts[col] = n_missing
        pct_missing = round(float(n_missing / total_rows * 100), 2) if total_rows > 0 else 0.0
        missing_percentages[col] = pct_missing
        if pct_missing > 20.0:
            warning_msg = f"Column '{col}' has high missingness: {pct_missing}%."
            warnings_list.append(warning_msg)
            warnings.warn(warning_msg)

    # 4. Data types
    dtypes: dict[str, str] = {col: str(df[col].dtype) for col in df.columns}

    # 5. Numeric ranges and summary statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_summary: dict[str, dict[str, Any]] = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) > 0:
            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outliers = int(((series < lower_bound) | (series > upper_bound)).sum())

            numeric_summary[col] = {
                "count": int(series.count()),
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": round(float(series.mean()), 2),
                "median": round(float(series.median()), 2),
                "std": round(float(series.std()), 2) if len(series) > 1 else 0.0,
                "q25": round(q25, 2),
                "q75": round(q75, 2),
                "outliers_count": outliers,
            }
            # Logical validation checks
            if col in ["ApplicantIncome", "LoanAmount"] and series.min() < 0:
                warning_msg = f"Column '{col}' contains negative values."
                warnings_list.append(warning_msg)
                warnings.warn(warning_msg)

    # 6. Categorical frequency counts
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    category_counts: dict[str, dict[str, int]] = {}
    for col in categorical_cols:
        val_counts = df[col].value_counts(dropna=False).to_dict()
        category_counts[col] = {str(k): int(v) for k, v in val_counts.items()}

    report: dict[str, Any] = {
        "summary": {
            "row_count": total_rows,
            "column_count": total_cols,
            "duplicate_count": dup_count,
            "schema_check": {
                "status": "PASS" if schema_ok else "FAIL",
                "missing_columns": missing_cols,
                "extra_columns": extra_cols,
            },
            "warnings_count": len(warnings_list),
            "warnings": warnings_list,
        },
        "missing_percentage": missing_percentages,
        "missing_counts": missing_counts,
        "data_types": dtypes,
        "numeric_summary": numeric_summary,
        "category_counts": category_counts,
    }

    # Save report
    if output_path is None:
        try:
            output_path = get_path("data_quality_report")
        except Exception:
            output_path = Path("reports/data_quality.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Data quality report written to %s", output_path)
    return report


if __name__ == "__main__":
    raw_path = get_path("raw_data")
    print(f"Loading data from {raw_path}...")
    df_raw = pd.read_csv(raw_path)
    rep = run_checks(df_raw)
    print(f"Data quality checks completed. Total rows: {rep['summary']['row_count']}, Warnings: {rep['summary']['warnings_count']}")
    print("Report saved to reports/data_quality.json")
