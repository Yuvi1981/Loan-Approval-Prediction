"""
Prediction Telemetry Store for loan-telemetry.
Provides persistent SQLite logging of all loan prediction inference requests,
probabilities, decisions, latency, top SHAP factors, and true outcomes,
along with query and statistical aggregation helpers.
"""

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_path

logger = logging.getLogger(__name__)


def get_db_path() -> Path:
    """Resolve database path from configuration."""
    try:
        db_path = get_path("telemetry_db")
    except Exception:
        db_path = Path("logs/telemetry.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def init_db(db_path: Path | None = None) -> None:
    """Initialize SQLite predictions table and schema if not present."""
    if db_path is None:
        db_path = get_db_path()

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                model_version TEXT NOT NULL,
                inputs_json TEXT NOT NULL,
                probability REAL NOT NULL,
                decision INTEGER NOT NULL,
                threshold REAL NOT NULL,
                latency_ms REAL NOT NULL,
                top_factors_json TEXT,
                true_label INTEGER
            );
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(ts);"
        )
        conn.commit()


def clear_telemetry(db_path: Path | None = None) -> None:
    """Clear all records from the predictions telemetry table."""
    if db_path is None:
        db_path = get_db_path()
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions;")
        conn.commit()


def log_prediction(
    inputs: dict[str, Any],
    probability: float,
    decision: int,
    threshold: float,
    latency_ms: float,
    model_version: str = "1.0.0",
    top_factors: list[dict[str, Any]] | None = None,
    true_label: int | None = None,
    db_path: Path | None = None,
) -> int:
    """Log a single prediction event to SQLite database.

    Args:
        inputs: Input features provided for prediction.
        probability: Model output probability of loan approval.
        decision: Final decision (1 for Approved, 0 for Rejected).
        threshold: Decision threshold used.
        latency_ms: Inference runtime latency in milliseconds.
        model_version: Model identifier or version tag.
        top_factors: List of top contributing factors.
        true_label: Optional ground truth label if known.
        db_path: Optional override for database file path.

    Returns:
        int: The primary key (ID) of the logged record.
    """
    if db_path is None:
        db_path = get_db_path()

    init_db(db_path)
    now_utc = datetime.now(UTC).isoformat()

    inputs_serialized = json.dumps(inputs, default=str)
    factors_serialized = json.dumps(top_factors, default=str) if top_factors else "[]"

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO predictions (
                ts, model_version, inputs_json, probability,
                decision, threshold, latency_ms, top_factors_json, true_label
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                now_utc,
                model_version,
                inputs_serialized,
                float(probability),
                int(decision),
                float(threshold),
                float(latency_ms),
                factors_serialized,
                true_label,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def update_true_label(
    prediction_id: int,
    true_label: int,
    db_path: Path | None = None,
) -> bool:
    """Update ground truth label for an existing prediction record."""
    if db_path is None:
        db_path = get_db_path()

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE predictions SET true_label = ? WHERE id = ?;",
            (int(true_label), int(prediction_id)),
        )
        conn.commit()
        return cursor.rowcount > 0


def read_predictions(
    since: str | None = None,
    limit: int | None = None,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Read logged predictions from SQLite into a pandas DataFrame."""
    if db_path is None:
        db_path = get_db_path()

    if not db_path.exists():
        init_db(db_path)
        return pd.DataFrame()

    query = "SELECT * FROM predictions"
    params: list[Any] = []
    if since is not None:
        query += " WHERE ts >= ?"
        params.append(since)
    query += " ORDER BY id DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(int(limit))

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)

    return df


def summary_stats(db_path: Path | None = None) -> dict[str, Any]:
    """Compute aggregate telemetry operational statistics."""
    df = read_predictions(db_path=db_path)
    if df.empty:
        return {
            "total_predictions": 0,
            "approval_rate": 0.0,
            "rejection_rate": 0.0,
            "mean_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "mean_probability": 0.0,
            "ground_truth_count": 0,
            "current_accuracy": None,
        }

    total = len(df)
    approvals = int((df["decision"] == 1).sum())
    app_rate = round(float(approvals / total), 4)

    latencies = df["latency_ms"].dropna()
    mean_lat = round(float(latencies.mean()), 2) if len(latencies) > 0 else 0.0
    p95_lat = round(float(np.percentile(latencies, 95)), 2) if len(latencies) > 0 else 0.0

    # Accuracy on labeled data
    labeled = df.dropna(subset=["true_label"])
    curr_acc = None
    if len(labeled) > 0:
        correct = (labeled["decision"] == labeled["true_label"]).sum()
        curr_acc = round(float(correct / len(labeled)), 4)

    return {
        "total_predictions": total,
        "approval_rate": app_rate,
        "rejection_rate": round(1.0 - app_rate, 4),
        "mean_latency_ms": mean_lat,
        "p95_latency_ms": p95_lat,
        "mean_probability": round(float(df["probability"].mean()), 4),
        "ground_truth_count": len(labeled),
        "current_accuracy": curr_acc,
    }


if __name__ == "__main__":
    init_db()
    stats = summary_stats()
    print("Telemetry database initialized. Current stats:", stats)
