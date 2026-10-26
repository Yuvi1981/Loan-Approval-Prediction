"""
Data Drift Detection module for loan-telemetry.
Implements Population Stability Index (PSI) and Kolmogorov-Smirnov (KS-test)
to monitor statistical distribution shifts between baseline training data
and incoming prediction batches.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.config import CONFIG

logger = logging.getLogger(__name__)


def calculate_psi(
    expected: np.ndarray | pd.Series | list[Any],
    actual: np.ndarray | pd.Series | list[Any],
    bins: int = 10,
    is_categorical: bool = False,
    epsilon: float = 1e-4,
) -> float:
    """Calculate the Population Stability Index (PSI) between baseline (expected) and new (actual).

    PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))

    Args:
        expected: Baseline/reference sample (e.g. training set).
        actual: Target/monitoring sample (e.g. recent applications).
        bins: Number of quantiles for numeric binning (default: 10).
        is_categorical: Whether the feature is categorical.
        epsilon: Small smoothing constant to prevent zero-division and log(0).

    Returns:
        float: Population Stability Index.
    """
    exp_series = pd.Series(expected).dropna()
    act_series = pd.Series(actual).dropna()

    if len(exp_series) == 0 or len(act_series) == 0:
        return 0.0

    if is_categorical:
        # Categorical: count per distinct level
        all_categories = sorted(
            list(set(exp_series.astype(str)).union(set(act_series.astype(str))))
        )
        exp_counts = exp_series.astype(str).value_counts()
        act_counts = act_series.astype(str).value_counts()

        exp_dist = np.array(
            [exp_counts.get(cat, 0) / len(exp_series) for cat in all_categories]
        )
        act_dist = np.array(
            [act_counts.get(cat, 0) / len(act_series) for cat in all_categories]
        )
    else:
        # Numeric: define quantiles on the expected series
        exp_arr = exp_series.to_numpy(dtype=float)
        act_arr = act_series.to_numpy(dtype=float)

        # Handle degenerate cases (all identical values)
        if np.all(exp_arr == exp_arr[0]):
            return 0.0 if np.all(act_arr == exp_arr[0]) else 1.0

        percentiles = np.linspace(0, 100, bins + 1)
        bin_edges = np.percentile(exp_arr, percentiles)
        bin_edges = np.unique(bin_edges)  # remove duplicate edges

        if len(bin_edges) <= 1:
            return 0.0

        # Adjust outer edges slightly to include min and max
        bin_edges[0] -= 1e-5
        bin_edges[-1] += 1e-5

        exp_counts, _ = np.histogram(exp_arr, bins=bin_edges)
        act_counts, _ = np.histogram(act_arr, bins=bin_edges)

        exp_dist = exp_counts / len(exp_arr)
        act_dist = act_counts / len(act_arr)

    # Apply epsilon smoothing and re-normalize
    exp_dist = np.where(exp_dist == 0, epsilon, exp_dist)
    act_dist = np.where(act_dist == 0, epsilon, act_dist)
    exp_dist = exp_dist / np.sum(exp_dist)
    act_dist = act_dist / np.sum(act_dist)

    psi_val = np.sum((act_dist - exp_dist) * np.log(act_dist / exp_dist))
    return float(max(0.0, round(psi_val, 4)))


def calculate_ks_test(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
) -> tuple[float, float, bool]:
    """Calculate Two-Sample Kolmogorov-Smirnov test for continuous features.

    Args:
        expected: Baseline numeric values.
        actual: New sample numeric values.

    Returns:
        Tuple[statistic, p_value, is_drift_flagged]:
            statistic: Maximum vertical distance between empirical CDFs.
            p_value: Probability of observing the distance under null hypothesis.
            is_drift_flagged: True if p_value < 0.05.
    """
    exp_arr = pd.Series(expected).dropna().to_numpy(dtype=float)
    act_arr = pd.Series(actual).dropna().to_numpy(dtype=float)

    if len(exp_arr) == 0 or len(act_arr) == 0:
        return 0.0, 1.0, False

    res = stats.ks_2samp(exp_arr, act_arr)
    stat = round(float(res.statistic), 4)
    p_val = round(float(res.pvalue), 4)
    is_drift = bool(p_val < CONFIG["drift"].get("ks_p_value_threshold", 0.05))

    return stat, p_val, is_drift


def generate_drift_report(
    train_df: pd.DataFrame,
    new_df: pd.DataFrame,
    features_to_check: list[str] | None = None,
) -> pd.DataFrame:
    """Generate comprehensive feature drift report comparing reference data with incoming data.

    Args:
        train_df: Reference baseline DataFrame.
        new_df: Monitored/incoming DataFrame.
        features_to_check: Optional list of features to evaluate.

    Returns:
        pd.DataFrame: Drift report table.
    """
    if features_to_check is None:
        # Check all overlapping non-ID columns
        features_to_check = [
            c
            for c in train_df.columns
            if c in new_df.columns and c not in ["Loan_ID", "Loan_Status"]
        ]

    psi_stable = CONFIG["drift"].get("psi_stable_threshold", 0.10)
    psi_mod = CONFIG["drift"].get("psi_moderate_threshold", 0.25)

    records = []
    for feat in features_to_check:
        is_cat = (
            train_df[feat].dtype == "object"
            or str(train_df[feat].dtype).startswith("cat")
            or train_df[feat].nunique() <= 5
        )

        psi_val = calculate_psi(
            train_df[feat],
            new_df[feat],
            bins=CONFIG["drift"].get("num_bins", 10),
            is_categorical=is_cat,
        )

        if not is_cat:
            ks_stat, ks_pval, ks_flag = calculate_ks_test(train_df[feat], new_df[feat])
        else:
            ks_stat, ks_pval, ks_flag = np.nan, np.nan, False

        # Determine PSI status
        if psi_val < psi_stable:
            status = "Stable"
            flag = False
        elif psi_val < psi_mod:
            status = "Moderate Drift"
            flag = True
        else:
            status = "Significant Drift"
            flag = True

        records.append(
            {
                "feature": feat,
                "type": "Categorical" if is_cat else "Numeric",
                "psi": psi_val,
                "psi_status": status,
                "ks_statistic": ks_stat,
                "ks_p_value": ks_pval,
                "drift_detected": flag or ks_flag,
            }
        )

    report_df = pd.DataFrame(records).sort_values(by="psi", ascending=False).reset_index(drop=True)
    return report_df
