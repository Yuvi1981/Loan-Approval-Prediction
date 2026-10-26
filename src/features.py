"""
Feature Engineering and Selection module for loan-telemetry.
Computes domain financial features (Total_Income, EMI, Balance_Income,
Loan_to_Income, Log_Total_Income) with zero-division guards and provides
correlation, chi-square, and tree-based importance utilities.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer domain-specific financial features for loan assessment.

    Features created:
    - Total_Income = ApplicantIncome + CoapplicantIncome
    - EMI = LoanAmount / Loan_Amount_Term
    - Balance_Income = Total_Income - (EMI * 1000)
    - Loan_to_Income = (LoanAmount * 1000) / Total_Income
    - Log_Total_Income = log1p(Total_Income)

    Includes zero-division safeguards to prevent infinite values.

    Args:
        df: Input DataFrame with raw applicant and loan columns.

    Returns:
        pd.DataFrame: DataFrame containing original plus engineered features.
    """
    df_out = df.copy()

    # Ensure numeric columns are cast to float
    app_inc = pd.to_numeric(df_out.get("ApplicantIncome", 0.0), errors="coerce").fillna(0.0)
    coapp_inc = pd.to_numeric(df_out.get("CoapplicantIncome", 0.0), errors="coerce").fillna(0.0)
    loan_amt = pd.to_numeric(df_out.get("LoanAmount", np.nan), errors="coerce")
    term = pd.to_numeric(df_out.get("Loan_Amount_Term", np.nan), errors="coerce")

    # 1. Total_Income
    total_inc = app_inc + coapp_inc
    df_out["Total_Income"] = total_inc

    # 2. EMI (LoanAmount in thousands / Loan_Amount_Term in months)
    # Safe division: if term is 0 or NaN, set EMI to NaN (handled downstream by imputer)
    safe_term = term.replace(0, np.nan)
    emi = loan_amt / safe_term
    df_out["EMI"] = emi

    # 3. Balance_Income = Total_Income - (EMI * 1000)
    # If EMI is NaN, Balance_Income is computed after median EMI or left as NaN for imputer
    df_out["Balance_Income"] = total_inc - (emi * 1000.0)

    # 4. Loan_to_Income = (LoanAmount * 1000) / Total_Income
    # Safeguard against zero total income (replace <= 0 with 1.0 for ratio calculation)
    safe_income = np.where(total_inc > 0, total_inc, 1.0)
    loan_to_income = (loan_amt * 1000.0) / safe_income
    # Replace any accidental infs with NaN or 0
    df_out["Loan_to_Income"] = np.where(np.isinf(loan_to_income), np.nan, loan_to_income)

    # 5. Log_Total_Income = log1p(Total_Income)
    df_out["Log_Total_Income"] = np.log1p(np.maximum(0.0, total_inc))

    return df_out


def compute_correlations(
    df: pd.DataFrame, method: str = "pearson"
) -> pd.DataFrame:
    """Compute pairwise feature correlation matrix for numeric columns.

    Args:
        df: Input DataFrame.
        method: 'pearson', 'kendall', or 'spearman'.

    Returns:
        pd.DataFrame: Correlation matrix.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr(method=method)


def compute_chi_square_scores(
    X_cat: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    """Compute Chi-Square statistic and p-values for non-negative categorical features.

    Args:
        X_cat: Categorical features encoded as non-negative integers or one-hot.
        y: Binary target series.

    Returns:
        pd.DataFrame: Table of features with chi2 statistic and p-value.
    """
    valid_mask = (~X_cat.isna().any(axis=1)) & (~y.isna())
    chi2_stat, p_val = chi2(X_cat[valid_mask], y[valid_mask])
    res = pd.DataFrame(
        {"feature": X_cat.columns, "chi2_stat": chi2_stat, "p_value": p_val}
    )
    return res.sort_values(by="chi2_stat", ascending=False).reset_index(drop=True)


def compute_rf_feature_importances(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train a quick Random Forest classifier and extract Gini feature importances.

    Args:
        X: Preprocessed feature matrix.
        y: Target array.
        feature_names: Optional column labels.
        random_state: Seed for reproducibility.

    Returns:
        pd.DataFrame: Table of feature names and normalized importance scores.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf.fit(X, y)
    importances = rf.feature_importances_

    names = feature_names if feature_names is not None else [f"f_{i}" for i in range(X.shape[1])]
    df_imp = pd.DataFrame({"feature": names, "importance": importances})
    return df_imp.sort_values(by="importance", ascending=False).reset_index(drop=True)


def run_pca_experiment(
    X: np.ndarray, n_components: int = 5
) -> tuple[PCA, np.ndarray, np.ndarray]:
    """Perform PCA dimensionality reduction experiment.

    Args:
        X: Standardized numeric feature array.
        n_components: Number of principal components.

    Returns:
        Tuple[PCA, explained_variance_ratio, transformed_X]
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return pca, pca.explained_variance_ratio_, X_pca
