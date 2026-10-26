"""
Smoke test for end-to-end prediction, explainability, and telemetry.
"""

from pathlib import Path

from src.predict import predict_one
from src.telemetry import read_predictions


def test_predict_one_smoke(tmp_path: Path):
    sample = {
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 2000,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360,
        "Credit_History": 1.0,
        "Dependents": "1",
        "Education": "Graduate",
        "Self_Employed": "No",
        "Property_Area": "Semiurban",
    }

    res = predict_one(sample, threshold=0.50, log_to_telemetry=True)

    assert "decision" in res
    assert res["decision"] in [0, 1]
    assert 0.0 <= res["probability"] <= 1.0
    assert "top_factors" in res
    assert isinstance(res["top_factors"], list)
    assert res["latency_ms"] < 500.0  # NFR: latency under 500ms
    assert res["prediction_id"] is not None

    df_logs = read_predictions(limit=1)
    assert not df_logs.empty
    assert df_logs.iloc[0]["id"] == res["prediction_id"]
