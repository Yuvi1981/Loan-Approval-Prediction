"""
Evaluation and plotting module for loan-telemetry.
Computes classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC),
confusion matrix, threshold sweep analysis, and generates diagnostic plots.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Calculate key classification metrics given true labels and predicted probabilities.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class (1).
        threshold: Decision threshold for positive classification.

    Returns:
        Dict[str, Any]: Dictionary containing accuracy, precision, recall, f1,
            roc_auc, and confusion matrix.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob_arr >= threshold).astype(int)

    acc = float(accuracy_score(y_true_arr, y_pred))
    prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    except ValueError:
        auc = 0.5

    cm = confusion_matrix(y_true_arr, y_pred).tolist()

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "threshold": threshold,
        "confusion_matrix": cm,
    }


def compute_threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute precision, recall, F1, and accuracy across a range of decision thresholds.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted positive class probabilities.
        thresholds: Array of thresholds to evaluate.

    Returns:
        pd.DataFrame: Metric curves indexed by threshold.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 37)

    records = []
    for t in thresholds:
        metrics = evaluate_predictions(y_true, y_prob, threshold=float(t))
        records.append(
            {
                "threshold": round(float(t), 3),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )

    return pd.DataFrame(records)


def plot_confusion_matrix_fig(
    cm: list[list[int]],
    class_names: list[str] | None = None,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Generate a clean matplotlib figure for a confusion matrix.

    Args:
        cm: 2x2 confusion matrix [[TN, FP], [FN, TP]].
        class_names: Class labels (default: ['Rejected', 'Approved']).
        title: Chart title.

    Returns:
        plt.Figure: Rendered figure.
    """
    if class_names is None:
        class_names = ["Rejected (0)", "Approved (1)"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    thresh = np.max(cm) / 2.0
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            ax.text(
                j,
                i,
                format(cm[i][j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i][j] > thresh else "black",
                fontweight="bold",
                fontsize=11,
            )

    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_xlabel("Predicted Label", fontweight="bold")
    plt.tight_layout()
    return fig


def plot_roc_curve_fig(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
) -> plt.Figure:
    """Generate a ROC curve matplotlib figure.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted positive class probabilities.
        model_name: Label for the curve.

    Returns:
        plt.Figure: Rendered ROC plot.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    try:
        auc_val = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc_val = 0.5

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        fpr,
        tpr,
        color="#1f77b4",
        lw=2,
        label=f"{model_name} (AUC = {auc_val:.3f})",
    )
    ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall)", fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC)", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    return fig


def plot_threshold_sweep_fig(sweep_df: pd.DataFrame) -> plt.Figure:
    """Plot Precision, Recall, and F1 across thresholds.

    Args:
        sweep_df: DataFrame output of compute_threshold_sweep.

    Returns:
        plt.Figure: Rendered threshold trade-off curve.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision", color="#2ca02c", lw=2)
    ax.plot(sweep_df["threshold"], sweep_df["recall"], label="Recall", color="#d62728", lw=2)
    ax.plot(sweep_df["threshold"], sweep_df["f1"], label="F1-Score", color="#1f77b4", lw=2, linestyle="--")
    ax.set_xlabel("Decision Threshold", fontweight="bold")
    ax.set_ylabel("Metric Score", fontweight="bold")
    ax.set_title("Decision Threshold Trade-Off Analysis", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    return fig
