"""
Unit tests for data preprocessing module.
Validates IQR outlier capper, dependents cleaner, missing value handling,
and robustness against unseen categories.
"""

import numpy as np
import pandas as pd

from src.preprocessing import (
    IQROutlierCapper,
    build_preprocessor,
    clean_dependents,
    extract_features_and_target,
)


def test_clean_dependents():
    df = pd.DataFrame({"Dependents": ["0", "1", "2", "3+", None, np.nan, " 3+ "]})
    cleaned = clean_dependents(df)
    expected_values = ["0", "1", "2", "3", np.nan, np.nan, "3"]

    for actual, expected in zip(cleaned["Dependents"], expected_values):
        if pd.isna(expected):
            assert pd.isna(actual)
        else:
            assert actual == expected


def test_iqr_outlier_capper():
    X = np.array([[10.0], [12.0], [11.0], [13.0], [12.0], [100.0], [-50.0]])
    capper = IQROutlierCapper(factor=1.5)
    capper.fit(X)

    X_trans = capper.transform(X)
    assert X_trans.max() < 100.0
    assert X_trans.min() > -50.0
    assert len(X_trans) == len(X)


def test_preprocessor_no_nans_and_unseen_category():
    train_data = pd.DataFrame(
        {
            "ApplicantIncome": [5000, 3000, np.nan, 8000],
            "CoapplicantIncome": [0, 1500, 2000, 0],
            "LoanAmount": [120, np.nan, 150, 200],
            "Loan_Amount_Term": [360, 360, 180, 360],
            "Credit_History": [1.0, 0.0, np.nan, 1.0],
            "Dependents": ["0", "1", "3+", "2"],
            "Education": ["Graduate", "Not Graduate", "Graduate", "Graduate"],
            "Self_Employed": ["No", "Yes", "No", np.nan],
            "Property_Area": ["Urban", "Rural", "Semiurban", "Urban"],
        }
    )
    cleaned_train = clean_dependents(train_data)

    num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
    cat_cols = ["Dependents", "Education", "Self_Employed", "Property_Area"]

    preprocessor = build_preprocessor(num_cols, cat_cols)
    transformed_train = preprocessor.fit_transform(cleaned_train)

    # Check no NaNs
    assert not np.isnan(transformed_train).any()

    # Test with unseen category and new missing values in test set
    test_data = pd.DataFrame(
        {
            "ApplicantIncome": [np.nan, 4500],
            "CoapplicantIncome": [1000, 0],
            "LoanAmount": [130, np.nan],
            "Loan_Amount_Term": [360, 360],
            "Credit_History": [1.0, 0.0],
            "Dependents": ["UnknownCategory", "0"],
            "Education": ["Doctorate", "Graduate"],  # 'Doctorate' is unseen
            "Self_Employed": ["Maybe", "No"],         # 'Maybe' is unseen
            "Property_Area": ["Suburban", "Urban"],    # 'Suburban' is unseen
        }
    )
    cleaned_test = clean_dependents(test_data)
    transformed_test = preprocessor.transform(cleaned_test)

    # Should not crash and contain no NaNs
    assert transformed_test.shape[0] == 2
    assert not np.isnan(transformed_test).any()


def test_extract_features_and_target_excludes_sensitive():
    sample_df = pd.DataFrame(
        {
            "Loan_ID": ["LP001", "LP002"],
            "Gender": ["Male", "Female"],
            "Married": ["Yes", "No"],
            "Dependents": ["0", "1"],
            "Education": ["Graduate", "Not Graduate"],
            "ApplicantIncome": [5000, 4000],
            "Loan_Status": ["Y", "N"],
        }
    )

    X, y, sensitive_df = extract_features_and_target(sample_df)

    # Loan_ID, Gender, Married must be excluded from model inputs X
    assert "Loan_ID" not in X.columns
    assert "Gender" not in X.columns
    assert "Married" not in X.columns
    assert "Loan_Status" not in X.columns
    assert list(y) == [1, 0]

    # Sensitive attributes must be in sensitive_df
    assert "Gender" in sensitive_df.columns
    assert "Married" in sensitive_df.columns
