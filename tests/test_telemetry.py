"""
Unit tests for telemetry module.
Validates SQLite table initialization, log and read round-trip,
true outcome update, and summary statistics calculation.
"""

from pathlib import Path

import pytest

from src.telemetry import (
    init_db,
    log_prediction,
    read_predictions,
    summary_stats,
    update_true_label,
)


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "test_telemetry.db"
    init_db(db_file)
    return db_file


def test_log_and_read_round_trip(temp_db: Path):
    sample_inputs = {"ApplicantIncome": 5000, "Credit_History": 1.0}
    top_factors = [{"feature": "Credit_History", "impact": 0.35}]

    pred_id = log_prediction(
        inputs=sample_inputs,
        probability=0.88,
        decision=1,
        threshold=0.50,
        latency_ms=12.5,
        model_version="1.0.0",
        top_factors=top_factors,
        db_path=temp_db,
    )

    assert pred_id is not None
    assert pred_id >= 1

    df = read_predictions(db_path=temp_db)
    assert len(df) == 1
    assert df.iloc[0]["probability"] == pytest.approx(0.88, abs=1e-3)
    assert df.iloc[0]["decision"] == 1
    assert df.iloc[0]["threshold"] == 0.50
    assert df.iloc[0]["model_version"] == "1.0.0"


def test_update_true_label(temp_db: Path):
    pred_id = log_prediction(
        inputs={"ApplicantIncome": 3000},
        probability=0.20,
        decision=0,
        threshold=0.50,
        latency_ms=10.0,
        db_path=temp_db,
    )

    success = update_true_label(prediction_id=pred_id, true_label=0, db_path=temp_db)
    assert success is True

    df = read_predictions(db_path=temp_db)
    assert df.iloc[0]["true_label"] == 0


def test_summary_stats(temp_db: Path):
    # Log 2 predictions
    p1 = log_prediction(
        inputs={}, probability=0.9, decision=1, threshold=0.5, latency_ms=10, db_path=temp_db
    )
    p2 = log_prediction(
        inputs={}, probability=0.1, decision=0, threshold=0.5, latency_ms=20, db_path=temp_db
    )
    update_true_label(p1, 1, db_path=temp_db)
    update_true_label(p2, 0, db_path=temp_db)

    stats = summary_stats(db_path=temp_db)
    assert stats["total_predictions"] == 2
    assert stats["approval_rate"] == 0.5
    assert stats["current_accuracy"] == 1.0
    assert stats["mean_latency_ms"] == 15.0
