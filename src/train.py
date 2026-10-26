"""
Model training, cross-validation, tuning, and MLflow tracking module for loan-telemetry.
Trains baseline and advanced classifiers, runs stratified CV, performs hyperparameter
tuning, evaluates class-imbalance treatments (class-weight, SMOTE), logs runs to MLflow,
and persists the top-performing model pipeline and metadata.
"""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import CONFIG, get_path
from src.data_quality import run_checks
from src.evaluate import (
    compute_threshold_sweep,
    evaluate_predictions,
    plot_confusion_matrix_fig,
    plot_roc_curve_fig,
    plot_threshold_sweep_fig,
)
from src.features import add_features
from src.preprocessing import build_preprocessor, extract_features_and_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_cv_metrics(
    pipeline: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """Perform Stratified K-Fold CV and return mean and std for key metrics."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    accs, precs, recs, f1s, aucs = [], [], [], [], []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_val)[:, 1]
        else:
            y_prob = pipeline.predict(X_val)

        res = evaluate_predictions(y_val.to_numpy(), y_prob, threshold=0.5)
        accs.append(res["accuracy"])
        precs.append(res["precision"])
        recs.append(res["recall"])
        f1s.append(res["f1"])
        aucs.append(res["roc_auc"])

    return {
        "cv_accuracy_mean": round(float(np.mean(accs)), 4),
        "cv_accuracy_std": round(float(np.std(accs)), 4),
        "cv_precision_mean": round(float(np.mean(precs)), 4),
        "cv_recall_mean": round(float(np.mean(recs)), 4),
        "cv_f1_mean": round(float(np.mean(f1s)), 4),
        "cv_f1_std": round(float(np.std(f1s)), 4),
        "cv_auc_mean": round(float(np.mean(aucs)), 4),
    }


def train_and_track_all_models() -> dict[str, Any]:
    """Train all baseline and advanced models, track runs in MLflow, and persist the best model."""
    seed = CONFIG["project"]["random_seed"]
    np.random.seed(seed)

    # 1. Load Data & Quality Check
    raw_path = get_path("raw_data")
    logger.info("Loading raw data from %s...", raw_path)
    df_raw = pd.read_csv(raw_path)

    # Run data quality checks
    dq_report = run_checks(df_raw)
    logger.info("Data quality checks passed. Total rows: %d", dq_report["summary"]["row_count"])

    # 2. Feature Engineering
    logger.info("Engineering financial domain features...")
    df_featured = add_features(df_raw)

    # 3. Separate features, target, and sensitive auditing attributes
    X_all, y_all, sensitive_all = extract_features_and_target(df_featured)
    if y_all is None:
        raise ValueError("Target column 'Loan_Status' missing from dataset.")

    # 4. Stratified 80/20 Split
    test_size = CONFIG["data"].get("test_size", 0.20)
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X_all,
        y_all,
        sensitive_all,
        test_size=test_size,
        stratify=y_all,
        random_state=seed,
    )

    logger.info("Train shape: %s | Test shape: %s", X_train.shape, X_test.shape)

    # Save processed splits for reproducibility and audit
    processed_train_path = get_path("processed_train")
    processed_test_path = get_path("processed_test")
    processed_train_path.parent.mkdir(parents=True, exist_ok=True)

    train_export = pd.concat([X_train, sens_train, y_train.rename("Loan_Status")], axis=1)
    test_export = pd.concat([X_test, sens_test, y_test.rename("Loan_Status")], axis=1)
    train_export.to_csv(processed_train_path, index=False)
    test_export.to_csv(processed_test_path, index=False)

    # Identify numeric and categorical columns for preprocessing
    num_cols = [
        c
        for c in CONFIG["data"]["model_features"]["numeric"]
        if c in X_train.columns
    ]
    cat_cols = [
        c
        for c in CONFIG["data"]["model_features"]["categorical"]
        if c in X_train.columns
    ]

    # Preprocessor
    preprocessor = build_preprocessor(num_cols, cat_cols)

    # Configure MLflow
    mlflow_tracking_uri = CONFIG["paths"].get("mlflow_tracking_uri", "./mlruns")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    experiment_name = "loan-approval-telemetry"
    mlflow.set_experiment(experiment_name)

    # Dictionary of Candidate Models
    # Baselines
    models_to_run: dict[str, tuple[BaseEstimator, dict[str, Any], bool]] = {
        "LogisticRegression": (
            LogisticRegression(max_iter=1000, random_state=seed),
            {"model_type": "baseline", "penalty": "l2", "C": 1.0},
            False,
        ),
        "DecisionTree": (
            DecisionTreeClassifier(max_depth=5, random_state=seed),
            {"model_type": "baseline", "max_depth": 5, "criterion": "gini"},
            False,
        ),
        "KNN": (
            KNeighborsClassifier(n_neighbors=5),
            {"model_type": "baseline", "n_neighbors": 5},
            False,
        ),
        "GaussianNaiveBayes": (
            GaussianNB(),
            {"model_type": "baseline"},
            False,
        ),
        # Advanced Ensembles
        "RandomForest": (
            RandomForestClassifier(n_estimators=100, max_depth=6, random_state=seed),
            {"model_type": "advanced_ensemble", "n_estimators": 100, "max_depth": 6},
            False,
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                eval_metric="logloss",
                random_state=seed,
            ),
            {"model_type": "advanced_ensemble", "n_estimators": 100, "max_depth": 4, "learning_rate": 0.05},
            False,
        ),
        # Imbalance treatments
        "RandomForest_ClassWeighted": (
            RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                class_weight="balanced",
                random_state=seed,
            ),
            {"model_type": "imbalance_treatment", "treatment": "class_weight_balanced"},
            False,
        ),
        "RandomForest_SMOTE": (
            RandomForestClassifier(n_estimators=100, max_depth=6, random_state=seed),
            {"model_type": "imbalance_treatment", "treatment": "SMOTE_resampling"},
            True,  # Uses SMOTE in imblearn pipeline
        ),
        "XGBoost_ClassWeighted": (
            XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                scale_pos_weight=0.5,
                eval_metric="logloss",
                random_state=seed,
            ),
            {"model_type": "imbalance_treatment", "treatment": "scale_pos_weight"},
            False,
        ),
    }

    # Complex Ensembles: Soft Voting and Stacking
    lr_base = LogisticRegression(max_iter=1000, random_state=seed)
    rf_base = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=seed)
    xgb_base = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=seed)

    voting_clf = VotingClassifier(
        estimators=[("lr", lr_base), ("rf", rf_base), ("xgb", xgb_base)],
        voting="soft",
    )
    models_to_run["VotingClassifier_Soft"] = (
        voting_clf,
        {"model_type": "ensemble", "voting": "soft", "estimators": "LR,RF,XGB"},
        False,
    )

    stacking_clf = StackingClassifier(
        estimators=[("rf", rf_base), ("xgb", xgb_base)],
        final_estimator=LogisticRegression(max_iter=1000, random_state=seed),
        cv=5,
    )
    models_to_run["StackingClassifier"] = (
        stacking_clf,
        {"model_type": "ensemble", "meta_learner": "LogisticRegression"},
        False,
    )

    # Tracking Results
    results_records: list[dict[str, Any]] = []
    trained_pipelines: dict[str, BaseEstimator] = {}

    artifacts_dir = Path("reports/figures")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    cv_folds = CONFIG["data"].get("cv_folds", 5)

    for model_name, (estimator, params, use_smote) in models_to_run.items():
        logger.info("Training and evaluating: %s (SMOTE=%s)...", model_name, use_smote)

        # Build clean pipeline
        if use_smote:
            pipeline = ImbPipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("smote", SMOTE(random_state=seed)),
                    ("classifier", estimator),
                ]
            )
        else:
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", estimator),
                ]
            )

        # 5-fold CV evaluation
        cv_metrics = compute_cv_metrics(pipeline, X_train, y_train, cv_folds=cv_folds, random_state=seed)

        # Fit on full training set
        pipeline.fit(X_train, y_train)
        trained_pipelines[model_name] = pipeline

        # Predict on held-out test set
        if hasattr(pipeline, "predict_proba"):
            y_test_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_test_prob = pipeline.predict(X_test)

        test_metrics = evaluate_predictions(y_test.to_numpy(), y_test_prob, threshold=0.5)

        # Generate plots
        cm_fig = plot_confusion_matrix_fig(
            test_metrics["confusion_matrix"],
            title=f"{model_name} Confusion Matrix",
        )
        cm_path = artifacts_dir / f"{model_name}_cm.png"
        cm_fig.savefig(cm_path)
        plt.close(cm_fig)

        roc_fig = plot_roc_curve_fig(y_test.to_numpy(), y_test_prob, model_name=model_name)
        roc_path = artifacts_dir / f"{model_name}_roc.png"
        roc_fig.savefig(roc_path)
        plt.close(roc_fig)

        # Log run in MLflow
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(params)
            mlflow.log_param("use_smote", use_smote)
            mlflow.log_param("cv_folds", cv_folds)
            mlflow.log_param("random_seed", seed)

            # Log CV metrics
            for k, v in cv_metrics.items():
                mlflow.log_metric(k, v)

            # Log Test metrics
            mlflow.log_metric("test_accuracy", test_metrics["accuracy"])
            mlflow.log_metric("test_precision", test_metrics["precision"])
            mlflow.log_metric("test_recall", test_metrics["recall"])
            mlflow.log_metric("test_f1", test_metrics["f1"])
            mlflow.log_metric("test_roc_auc", test_metrics["roc_auc"])

            # Log plot artifacts
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(roc_path))

            # Log pipeline model artifact
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

        record = {
            "model": model_name,
            "cv_f1_mean": cv_metrics["cv_f1_mean"],
            "cv_f1_std": cv_metrics["cv_f1_std"],
            "cv_accuracy_mean": cv_metrics["cv_accuracy_mean"],
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_roc_auc": test_metrics["roc_auc"],
        }
        results_records.append(record)

    # 5. Hyperparameter Tuning for Best Tree Models (RandomForest and XGBoost)
    logger.info("Conducting Hyperparameter Tuning (RandomizedSearchCV + GridSearchCV)...")

    # Tune Random Forest
    rf_param_grid = {
        "classifier__n_estimators": [50, 100, 150],
        "classifier__max_depth": [3, 5, 8],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2],
    }
    rf_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=seed)),
        ]
    )
    rf_search = GridSearchCV(
        rf_pipe,
        rf_param_grid,
        scoring="f1",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)
    best_rf_pipe = rf_search.best_estimator_

    rf_tuned_prob = best_rf_pipe.predict_proba(X_test)[:, 1]
    rf_tuned_metrics = evaluate_predictions(y_test.to_numpy(), rf_tuned_prob, threshold=0.5)

    with mlflow.start_run(run_name="RandomForest_Tuned"):
        mlflow.log_params(rf_search.best_params_)
        mlflow.log_metric("best_cv_f1", round(float(rf_search.best_score_), 4))
        mlflow.log_metric("test_accuracy", rf_tuned_metrics["accuracy"])
        mlflow.log_metric("test_precision", rf_tuned_metrics["precision"])
        mlflow.log_metric("test_recall", rf_tuned_metrics["recall"])
        mlflow.log_metric("test_f1", rf_tuned_metrics["f1"])
        mlflow.log_metric("test_roc_auc", rf_tuned_metrics["roc_auc"])
        mlflow.sklearn.log_model(best_rf_pipe, artifact_path="model")

    results_records.append(
        {
            "model": "RandomForest_Tuned",
            "cv_f1_mean": round(float(rf_search.best_score_), 4),
            "cv_f1_std": 0.0,
            "cv_accuracy_mean": 0.0,
            "test_accuracy": rf_tuned_metrics["accuracy"],
            "test_precision": rf_tuned_metrics["precision"],
            "test_recall": rf_tuned_metrics["recall"],
            "test_f1": rf_tuned_metrics["f1"],
            "test_roc_auc": rf_tuned_metrics["roc_auc"],
        }
    )
    trained_pipelines["RandomForest_Tuned"] = best_rf_pipe

    # Results Comparison Table
    results_df = pd.DataFrame(results_records).sort_values(
        by=["test_f1", "test_accuracy"], ascending=False
    ).reset_index(drop=True)

    print("\n==================== MODEL COMPARISON LEADERBOARD ====================")
    print(results_df.to_string(index=False))
    print("=======================================================================\n")

    # Select Best Model
    best_model_name = results_df.iloc[0]["model"]
    best_pipeline = trained_pipelines[best_model_name]
    logger.info("Top-performing model selected: %s", best_model_name)

    # Save Best Model Artifact
    best_model_path = get_path("best_model")
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, best_model_path)
    logger.info("Saved best model artifact to %s", best_model_path)

    # Generate and save threshold sweep analysis for the best model
    best_probs = best_pipeline.predict_proba(X_test)[:, 1]
    sweep_df = compute_threshold_sweep(y_test.to_numpy(), best_probs)
    sweep_fig = plot_threshold_sweep_fig(sweep_df)
    sweep_path = Path("reports/figures/threshold_sweep.png")
    sweep_fig.savefig(sweep_path)
    plt.close(sweep_fig)

    # Compute training data statistics for drift baselines
    num_stats: dict[str, dict[str, float]] = {}
    for col in num_cols:
        series = X_train[col].dropna()
        num_stats[col] = {
            "mean": round(float(series.mean()), 3),
            "std": round(float(series.std()), 3),
            "median": round(float(series.median()), 3),
            "q25": round(float(series.quantile(0.25)), 3),
            "q75": round(float(series.quantile(0.75)), 3),
        }

    # Save Model Metadata
    model_meta = {
        "model_name": best_model_name,
        "version": "1.0.0",
        "random_seed": seed,
        "selected_metrics": {
            "test_accuracy": float(results_df.iloc[0]["test_accuracy"]),
            "test_precision": float(results_df.iloc[0]["test_precision"]),
            "test_recall": float(results_df.iloc[0]["test_recall"]),
            "test_f1": float(results_df.iloc[0]["test_f1"]),
            "test_roc_auc": float(results_df.iloc[0]["test_roc_auc"]),
            "cv_f1_mean": float(results_df.iloc[0]["cv_f1_mean"]),
        },
        "default_threshold": CONFIG["decision"].get("default_threshold", 0.5),
        "features": {
            "numeric": num_cols,
            "categorical": cat_cols,
            "sensitive_excluded": ["Gender", "Married"],
        },
        "training_data_statistics": num_stats,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }

    meta_path = get_path("model_meta")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(model_meta, f, indent=2)

    logger.info("Model metadata saved to %s", meta_path)
    return {
        "best_model_name": best_model_name,
        "results": results_df.to_dict(orient="records"),
        "meta": model_meta,
    }


if __name__ == "__main__":
    train_and_track_all_models()
