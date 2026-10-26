"""
Preprocessing module for loan-telemetry.
Contains IQR outlier-capping transformer, dependents cleaner, and
ColumnTransformer pipeline to preprocess numeric and categorical data
strictly without data leakage.
"""


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class IQROutlierCapper(BaseEstimator, TransformerMixin):
    """Caps numeric outliers based on Interquartile Range (IQR).

    Values outside [Q1 - factor*IQR, Q3 + factor*IQR] are clipped to the boundaries.
    Boundaries are calculated on the training data only during fit().
    """

    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.lower_bounds_: np.ndarray | None = None
        self.upper_bounds_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "IQROutlierCapper":
        X_arr = np.asarray(X, dtype=np.float64)
        q25 = np.nanpercentile(X_arr, 25, axis=0)
        q75 = np.nanpercentile(X_arr, 75, axis=0)
        iqr = q75 - q25

        lower = q25 - self.factor * iqr
        upper = q75 + self.factor * iqr

        # Do not clip features with zero IQR (e.g. binary indicators like Credit_History)
        zero_iqr = iqr <= 0
        lower[zero_iqr] = -np.inf
        upper[zero_iqr] = np.inf

        self.lower_bounds_ = lower
        self.upper_bounds_ = upper
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise RuntimeError("IQROutlierCapper must be fitted before transform.")
        X_arr = np.asarray(X, dtype=np.float64).copy()
        return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f"num_{i}" for i in range(len(self.lower_bounds_))], dtype=object)
        return np.asarray(input_features, dtype=object)


def clean_dependents(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Dependents column by converting '3+' to 3 and preserving missing values.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with cleaned Dependents column.
    """
    df_out = df.copy()
    if "Dependents" in df_out.columns:
        # Convert strings '3+' to '3'
        df_out["Dependents"] = (
            df_out["Dependents"]
            .astype(str)
            .str.replace("+", "", regex=False)
            .str.strip()
        )
        # Handle nan strings produced by astype(str)
        df_out.loc[df_out["Dependents"].isin(["nan", "None", "<NA>"]), "Dependents"] = np.nan
    return df_out


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Build a ColumnTransformer for numeric and categorical pipelines.

    Numeric: median imputer -> IQR outlier capper -> StandardScaler.
    Categorical: most-frequent imputer -> OneHotEncoder(handle_unknown='ignore').

    Args:
        numeric_features: List of column names for numeric features.
        categorical_features: List of column names for categorical features.

    Returns:
        ColumnTransformer: Configured preprocessor.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("outlier_capper", IQROutlierCapper(factor=1.5)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def extract_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series | None, pd.DataFrame]:
    """Extract model feature dataframe (excluding Gender, Married, Loan_ID),
    target series (if present), and sensitive attribute audit dataframe.

    Args:
        df: Raw or feature-engineered input DataFrame.

    Returns:
        Tuple[X, y, sensitive_df]:
            X: Features for model training/inference (without sensitive columns).
            y: Binary target (1 = Approved, 0 = Rejected), or None.
            sensitive_df: Protected attributes (Gender, Married, Property_Area) for auditing.
    """
    df_clean = clean_dependents(df)

    # Sensitive attributes kept for fairness audit
    sensitive_cols = ["Gender", "Married", "Property_Area"]
    present_sensitive = [c for c in sensitive_cols if c in df_clean.columns]
    sensitive_df = df_clean[present_sensitive].copy()

    # Target
    y: pd.Series | None = None
    if "Loan_Status" in df_clean.columns:
        # Convert 'Y' -> 1, 'N' -> 0
        y = (df_clean["Loan_Status"] == "Y").astype(int)

    # Exclude Loan_ID and sensitive attributes from model input
    exclude_cols = ["Loan_ID", "Loan_Status", "Gender", "Married"]
    model_cols = [c for c in df_clean.columns if c not in exclude_cols]
    X = df_clean[model_cols].copy()

    return X, y, sensitive_df
