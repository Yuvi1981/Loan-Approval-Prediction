"""
Traffic Generation script for loan-telemetry.
Simulates real-world loan application traffic by submitting 200 applications
through the prediction pipeline and logging them to SQLite for dashboard demos.
"""

import logging
import random
from typing import Any, Dict

from src.predict import predict_one
from src.telemetry import summary_stats, update_true_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_sample_application(random_seed: int = None) -> Dict[str, Any]:
    """Generate a single realistic synthetic loan application."""
    if random_seed is not None:
        random.seed(random_seed)

    # Education
    education = random.choices(["Graduate", "Not Graduate"], weights=[0.8, 0.2])[0]

    # Income distributions
    if education == "Graduate":
        app_income = round(random.lognormvariate(8.3, 0.6), -1)  # ~$4,000 avg
    else:
        app_income = round(random.lognormvariate(8.0, 0.5), -1)  # ~$3,000 avg

    # Coapplicant
    has_coapplicant = random.random() < 0.55
    coapp_income = round(random.lognormvariate(7.2, 0.7), -1) if has_coapplicant else 0.0

    # Loan Amount
    total_inc = app_income + coapp_income
    loan_amt = round(min(700.0, max(20.0, (total_inc * 0.025) + random.gauss(20, 25))), -1)

    # Loan term
    term = random.choices([360, 180, 240, 120, 480], weights=[0.85, 0.05, 0.04, 0.03, 0.03])[0]

    # Credit History
    credit_hist = random.choices([1.0, 0.0], weights=[0.82, 0.18])[0]

    # Dependents & Self Employed
    dependents = random.choices(["0", "1", "2", "3+"], weights=[0.55, 0.17, 0.17, 0.11])[0]
    self_employed = random.choices(["No", "Yes"], weights=[0.86, 0.14])[0]

    # Sensitive attributes (tracked for fairness, excluded from model)
    gender = random.choices(["Male", "Female"], weights=[0.80, 0.20])[0]
    married = random.choices(["Yes", "No"], weights=[0.65, 0.35])[0]
    property_area = random.choices(["Semiurban", "Urban", "Rural"], weights=[0.38, 0.33, 0.29])[0]

    return {
        "ApplicantIncome": float(app_income),
        "CoapplicantIncome": float(coapp_income),
        "LoanAmount": float(loan_amt),
        "Loan_Amount_Term": float(term),
        "Credit_History": float(credit_hist),
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "Gender": gender,
        "Married": married,
        "Property_Area": property_area,
    }


def simulate_traffic(n_samples: int = 200) -> None:
    """Generate and submit n_samples predictions through the ML system."""
    logger.info("Starting simulation of %d loan applications...", n_samples)

    pred_ids = []
    for i in range(n_samples):
        app = generate_sample_application(random_seed=1000 + i)
        # Randomize threshold slightly around 0.50 for realistic operations
        threshold = random.choice([0.45, 0.50, 0.55])
        res = predict_one(app, threshold=threshold, log_to_telemetry=True)
        pred_ids.append(res["prediction_id"])

        # For ~60% of applications, assign a true repayment outcome later (simulating post-settlement monitoring)
        if random.random() < 0.60 and res["prediction_id"] is not None:
            # High probability of default if Credit_History was 0
            if app["Credit_History"] == 0.0:
                true_label = 1 if random.random() < 0.15 else 0
            else:
                true_label = 1 if random.random() < 0.85 else 0
            update_true_label(res["prediction_id"], true_label)

    logger.info("Successfully generated %d predictions.", len(pred_ids))
    stats = summary_stats()
    print("\n========== POST-TRAFFIC TELEMETRY SUMMARY ==========")
    print(f"Total Logged Predictions : {stats['total_predictions']}")
    print(f"Approval Rate            : {stats['approval_rate']:.1%}")
    print(f"Mean Inference Latency   : {stats['mean_latency_ms']} ms")
    print(f"P95 Inference Latency    : {stats['p95_latency_ms']} ms")
    print(f"Labeled Ground Truths    : {stats['ground_truth_count']}")
    if stats["current_accuracy"] is not None:
        print(f"Operational Accuracy     : {stats['current_accuracy']:.1%}")
    print("====================================================\n")


if __name__ == "__main__":
    simulate_traffic(n_samples=200)
