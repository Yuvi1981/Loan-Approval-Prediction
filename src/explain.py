"""
Explainability module for loan-telemetry.
Provides local (single prediction) and global SHAP explanations with
TreeExplainer and KernelExplainer support and graceful degradation fallback.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


class LoanExplainer:
    """SHAP explainer wrapper for lending decisions."""

    def __init__(self, pipeline: Any, background_data: pd.DataFrame | None = None):
        """Initialize the explainer with a trained pipeline and optional background data."""
        self.pipeline = pipeline
        self.background_data = background_data
        self.preprocessor = None
        self.classifier = None
        self.explainer = None
        self.feature_names: list[str] = []

        self._setup_explainer()

    def _setup_explainer(self) -> None:
        """Extract classifier and preprocessor, determine feature names, and initialize SHAP."""
        try:
            if hasattr(self.pipeline, "named_steps"):
                self.preprocessor = self.pipeline.named_steps.get("preprocessor")
                self.classifier = self.pipeline.named_steps.get("classifier")
            else:
                self.classifier = self.pipeline

            # Determine transformed feature names
            if self.preprocessor is not None and hasattr(self.preprocessor, "get_feature_names_out"):
                try:
                    self.feature_names = list(self.preprocessor.get_feature_names_out())
                except Exception:
                    pass

            # Pre-transform a small background sample if available
            X_bg_trans = None
            if self.background_data is not None and self.preprocessor is not None:
                bg_sample = self.background_data.head(50)
                try:
                    X_bg_trans = self.preprocessor.transform(bg_sample)
                except Exception as e:
                    logger.warning("Could not transform background sample: %s", e)

            # Choose appropriate SHAP explainer
            classifier_type = type(self.classifier).__name__.lower()
            if any(tree_name in classifier_type for tree_name in ["forest", "tree", "xgb", "boost", "gbm"]):
                try:
                    self.explainer = shap.TreeExplainer(self.classifier)
                    logger.info("Initialized SHAP TreeExplainer for %s", classifier_type)
                except Exception:
                    if X_bg_trans is not None:
                        self.explainer = shap.Explainer(self.classifier.predict_proba, X_bg_trans)
            else:
                if X_bg_trans is not None and hasattr(self.classifier, "predict_proba"):
                    self.explainer = shap.KernelExplainer(
                        lambda x: self.classifier.predict_proba(x)[:, 1],
                        shap.kmeans(X_bg_trans, min(10, len(X_bg_trans))),
                    )
                    logger.info("Initialized SHAP KernelExplainer")
        except Exception as e:
            logger.warning("SHAP initialization encountered an error: %s. Graceful fallback active.", e)
            self.explainer = None

    def explain_one(
        self,
        raw_row: pd.DataFrame,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Explain a single prediction by returning the top-k contributing factors.

        Args:
            raw_row: Single-row DataFrame containing model input features.
            top_k: Number of top driving factors to return.

        Returns:
            List[Dict[str, Any]]: List of top factors with name, input value, SHAP impact, and direction.
        """
        if self.explainer is None:
            return []

        try:
            # Transform input row
            if self.preprocessor is not None:
                X_trans = self.preprocessor.transform(raw_row)
            else:
                X_trans = np.asarray(raw_row)

            shap_vals = self.explainer.shap_values(X_trans)

            # Extract positive class SHAP values
            if isinstance(shap_vals, list) and len(shap_vals) == 2:
                # Binary classification [class 0, class 1]
                vals = shap_vals[1][0]
            elif isinstance(shap_vals, np.ndarray):
                if shap_vals.ndim == 3 and shap_vals.shape[2] == 2:
                    vals = shap_vals[0, :, 1]
                elif shap_vals.ndim == 2:
                    vals = shap_vals[0]
                else:
                    vals = shap_vals.flatten()
            else:
                vals = np.array(shap_vals).flatten()

            names = (
                self.feature_names
                if len(self.feature_names) == len(vals)
                else [f"Feature_{i}" for i in range(len(vals))]
            )

            # Clean feature names (e.g. remove 'num__' or 'cat__')
            clean_names = [
                n.replace("num__", "").replace("cat__", "").replace("remainder__", "")
                for n in names
            ]

            # Pair names, values, and impacts
            factors = []
            for name, shap_val, trans_val in zip(clean_names, vals, X_trans[0]):
                factors.append(
                    {
                        "feature": name,
                        "value": round(float(trans_val), 2),
                        "impact": round(float(shap_val), 4),
                        "abs_impact": abs(float(shap_val)),
                        "direction": "Positive (+)" if shap_val > 0 else "Negative (-)",
                        "description": (
                            "Pushes towards Approval"
                            if shap_val > 0
                            else "Pushes towards Rejection"
                        ),
                    }
                )

            # Sort by absolute impact
            factors.sort(key=lambda x: x["abs_impact"], reverse=True)
            return factors[:top_k]

        except Exception as e:
            logger.warning("SHAP explanation failed: %s. Returning fallback.", e)
            return []

    def global_importance(self, X_sample: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
        """Compute mean absolute SHAP feature importance across a sample.

        Args:
            X_sample: DataFrame of input features.
            top_k: Number of features to return.

        Returns:
            pd.DataFrame: Table of feature names and mean absolute SHAP impact.
        """
        if self.explainer is None:
            return pd.DataFrame(columns=["feature", "mean_abs_shap"])

        try:
            if self.preprocessor is not None:
                X_trans = self.preprocessor.transform(X_sample)
            else:
                X_trans = np.asarray(X_sample)

            shap_vals = self.explainer.shap_values(X_trans)
            if isinstance(shap_vals, list) and len(shap_vals) == 2:
                vals = np.abs(shap_vals[1]).mean(axis=0)
            elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                vals = np.abs(shap_vals[:, :, 1]).mean(axis=0)
            else:
                vals = np.abs(shap_vals).mean(axis=0)

            names = (
                self.feature_names
                if len(self.feature_names) == len(vals)
                else [f"Feature_{i}" for i in range(len(vals))]
            )
            clean_names = [
                n.replace("num__", "").replace("cat__", "") for n in names
            ]

            df_imp = pd.DataFrame({"feature": clean_names, "mean_abs_shap": vals})
            return df_imp.sort_values(by="mean_abs_shap", ascending=False).head(top_k).reset_index(drop=True)
        except Exception as e:
            logger.warning("Global SHAP computation failed: %s", e)
            return pd.DataFrame(columns=["feature", "mean_abs_shap"])
