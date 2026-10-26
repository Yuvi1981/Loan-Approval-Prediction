"""
Unit tests for feature engineering module.
Validates financial formulas, division by zero guards, and log transforms.
"""

import numpy as np
import pandas as pd

from src.features import add_features


def test_add_features_formulas():
    df = pd.DataFrame(
        {
            "ApplicantIncome": [5000, 3000],
            "CoapplicantIncome": [2000, 1000],
            "LoanAmount": [140, 100],  # thousands
            "Loan_Amount_Term": [360, 180],  # months
        }
    )

    featured = add_features(df)

    # 1. Total_Income = ApplicantIncome + CoapplicantIncome
    assert list(featured["Total_Income"]) == [7000.0, 4000.0]

    # 2. EMI = LoanAmount / Loan_Amount_Term
    expected_emi_0 = 140.0 / 360.0
    expected_emi_1 = 100.0 / 180.0
    np.testing.assert_allclose(featured["EMI"], [expected_emi_0, expected_emi_1])

    # 3. Balance_Income = Total_Income - EMI * 1000
    expected_bal_0 = 7000.0 - (expected_emi_0 * 1000.0)
    expected_bal_1 = 4000.0 - (expected_emi_1 * 1000.0)
    np.testing.assert_allclose(featured["Balance_Income"], [expected_bal_0, expected_bal_1])

    # 4. Loan_to_Income = LoanAmount * 1000 / Total_Income
    expected_lti_0 = (140.0 * 1000.0) / 7000.0
    expected_lti_1 = (100.0 * 1000.0) / 4000.0
    np.testing.assert_allclose(featured["Loan_to_Income"], [expected_lti_0, expected_lti_1])

    # 5. Log_Total_Income = log1p(Total_Income)
    expected_log_0 = np.log1p(7000.0)
    expected_log_1 = np.log1p(4000.0)
    np.testing.assert_allclose(featured["Log_Total_Income"], [expected_log_0, expected_log_1])


def test_add_features_zero_income_and_zero_term_guard():
    # Applicant with zero income and zero term
    df = pd.DataFrame(
        {
            "ApplicantIncome": [0],
            "CoapplicantIncome": [0],
            "LoanAmount": [100],
            "Loan_Amount_Term": [0],
        }
    )

    featured = add_features(df)

    # Total_Income is 0
    assert featured["Total_Income"].iloc[0] == 0.0

    # No infs in any generated column
    for col in ["Total_Income", "EMI", "Balance_Income", "Loan_to_Income", "Log_Total_Income"]:
        val = featured[col].iloc[0]
        assert not np.isinf(val), f"Column {col} generated infinity!"

    # Log_Total_Income should be log1p(0) = 0.0
    assert featured["Log_Total_Income"].iloc[0] == 0.0
