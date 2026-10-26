# TRD: Loan Approval Prediction with ML Telemetry

## 1. Technical Overview
A Python 3.11 project with a modular `src/` package, an MLflow tracking store, a SQLite telemetry database and a Streamlit front end. All components run locally.

## 2. Architecture
```
                +---------------------+
 data/raw/*.csv |  data_quality.py    |--> reports/data_quality.json
        |       +---------------------+
        v
 +---------------+   +---------------+   +--------------+
 | preprocessing |-->|   features    |-->|   train.py   |--> MLflow (mlruns/)
 +---------------+   +---------------+   |  evaluate.py |--> models/best_model.joblib
                                         +--------------+
                                                |
                    +---------------------------+------------------+
                    v                           v                  v
             +-------------+            +-------------+     +-------------+
             | predict.py  |--logs----->| telemetry   |     | fairness.py |
             | (+ SHAP)    |            | SQLite DB   |     | drift.py    |
             +-------------+            +-------------+     +-------------+
                    ^                           |                  |
                    +------------ Streamlit app (app/) <-----------+
```

## 3. Tech Stack
| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| Data | pandas, numpy |
| ML | scikit-learn, xgboost, imbalanced-learn |
| Explainability | shap |
| Tracking | mlflow (local file store `./mlruns`) |
| Drift | scipy (PSI and KS manual); evidently optional |
| Fairness | fairlearn |
| Telemetry store | SQLite via `sqlite3` |
| App | streamlit, plotly |
| Config | PyYAML |
| Testing | pytest |
| Quality | ruff, black (optional) |

Pin exact versions in `requirements.txt` after the first successful install.

## 4. Repository Layout
```
loan-telemetry/
  config.yaml
  requirements.txt
  README.md
  Makefile
  data/
    raw/train.csv
    processed/
  notebooks/
    01_eda.ipynb
  src/
    __init__.py
    config.py
    data_quality.py
    preprocessing.py
    features.py
    train.py
    evaluate.py
    predict.py
    explain.py
    telemetry.py
    drift.py
    fairness.py
    simulate_drift.py
  app/
    streamlit_app.py
    pages/
      1_Predict.py
      2_Model_Comparison.py
      3_Telemetry.py
      4_Drift.py
      5_Fairness.py
  models/
  reports/
  logs/
  tests/
    test_preprocessing.py
    test_features.py
    test_drift.py
    test_telemetry.py
```

## 5. Data Specification
**Source:** Loan Prediction Problem Dataset (Kaggle). Place the CSV at `data/raw/train.csv`.

| Column | Type | Notes |
|---|---|---|
| Loan_ID | string | Drop before modeling |
| Gender | categorical | Male/Female; **excluded from model inputs**, kept for fairness audit |
| Married | categorical | Yes/No; **excluded from model inputs**, kept for audit |
| Dependents | categorical | 0, 1, 2, 3+ -> convert to int |
| Education | categorical | Graduate / Not Graduate |
| Self_Employed | categorical | Yes/No |
| ApplicantIncome | numeric | Skewed; cap and log |
| CoapplicantIncome | numeric | Skewed |
| LoanAmount | numeric | Thousands; impute median |
| Loan_Amount_Term | numeric | Months; impute median |
| Credit_History | numeric 0/1 | Impute mode |
| Property_Area | categorical | Urban / Semiurban / Rural |
| Loan_Status | target | Y -> 1, N -> 0 |

Split: 80/20 stratified train/test with `random_state=42`. The test set is touched only for final evaluation.

## 6. Module Specifications

### 6.1 `config.yaml`
Holds: seed, paths, test size, CV folds, model hyperparameter grids, threshold default (0.5), drift thresholds, fairness gap threshold (0.10), sensitive attribute list.

### 6.2 `data_quality.py`
`run_checks(df) -> dict` returns: row count, missing % per column, duplicate count, dtype/schema check, numeric range summary, category value counts. Writes `reports/data_quality.json`. Raises a warning (not an error) on failed checks.

### 6.3 `preprocessing.py`
Build a `sklearn.pipeline.Pipeline` with `ColumnTransformer`:
- Numeric: median imputer -> (outlier cap via custom IQR transformer) -> StandardScaler
- Categorical: most-frequent imputer -> OneHotEncoder(handle_unknown="ignore")
- Cleaning of `Dependents` ("3+" -> 3) happens before the pipeline in a pure function.
Everything is fit on training data only inside the pipeline to avoid leakage.

### 6.4 `features.py`
`add_features(df) -> df` adds:
- `Total_Income = ApplicantIncome + CoapplicantIncome`
- `EMI = LoanAmount / Loan_Amount_Term`
- `Balance_Income = Total_Income - EMI * 1000`
- `Loan_to_Income = LoanAmount * 1000 / Total_Income`
- `Log_Total_Income = log1p(Total_Income)`
Guard against division by zero. Selection utilities: correlation matrix, chi-square scores, RandomForest importances; optional PCA experiment logged separately.

### 6.5 `train.py`
- Loads data, runs quality checks, adds features, splits.
- Models: LogisticRegression, DecisionTree, KNN, GaussianNB, RandomForest, XGBClassifier, VotingClassifier (soft), StackingClassifier.
- Tuning: RandomizedSearchCV then GridSearchCV on the best two models, scoring on F1 (configurable).
- Imbalance: run each model with and without `class_weight`/SMOTE (SMOTE only inside an `imblearn.pipeline.Pipeline` so it is applied to training folds only).
- Each model is one MLflow run with params, CV mean/std metrics, test metrics, confusion matrix plot, ROC plot, and the model artifact.
- Saves the best model (by CV F1) to `models/best_model.joblib` plus `models/model_meta.json` (version, metrics, feature list, training data statistics).

### 6.6 `evaluate.py`
`evaluate(model, X, y, threshold) -> dict` returns accuracy, precision, recall, F1, ROC-AUC, confusion matrix. Also provides plot helpers and a threshold sweep (precision/recall vs threshold).

### 6.7 `predict.py`
`predict_one(raw_input: dict, threshold) -> dict`:
1. Validate the input schema.
2. Apply `add_features` and the saved pipeline.
3. Compute probability, decision and latency.
4. Compute top-5 SHAP factors (via `explain.py`).
5. Log to SQLite via `telemetry.log_prediction`.
Returns decision, probability, top factors, latency_ms, prediction_id.

### 6.8 `telemetry.py`
SQLite file `logs/telemetry.db`.

Table `predictions`:
| Column | Type |
|---|---|
| id | INTEGER PK |
| ts | TEXT (ISO UTC) |
| model_version | TEXT |
| inputs_json | TEXT |
| probability | REAL |
| decision | INTEGER |
| threshold | REAL |
| latency_ms | REAL |
| top_factors_json | TEXT |
| true_label | INTEGER NULL (filled later for performance monitoring) |

Functions: `init_db()`, `log_prediction(...)`, `update_true_label(id, label)`, `read_predictions(since=None) -> DataFrame`, `summary_stats()` (count, approval rate, mean/p95 latency).

### 6.9 `drift.py`
- `psi(expected, actual, bins=10)` for numeric and categorical features.
- `ks_test(expected, actual)` using `scipy.stats.ks_2samp` for numeric features.
- `drift_report(train_df, new_df) -> DataFrame` with feature, PSI, KS p-value, status.
- PSI status: < 0.10 = stable, 0.10 to 0.25 = moderate, > 0.25 = significant drift.
- KS flag when p < 0.05.
- Optional: Evidently HTML report, wrapped in try/except so the app works without it.

### 6.10 `simulate_drift.py`
Creates a shifted copy of the test set (income x1.5, credit-history mix changed, loan amount shifted), scores it with the model, computes the drift report and the F1/AUC drop, saves results to `reports/drift_demo.json` and a chart.

### 6.11 `fairness.py`
Using Fairlearn `MetricFrame` per sensitive feature (Gender, Married, Property_Area): selection rate, recall, precision, accuracy. Computes `demographic_parity_difference` and `equal_opportunity_difference` (via TPR difference). Flags any gap above `fairness_gap_threshold`. Writes `reports/fairness.json`.

### 6.12 `explain.py`
Builds a SHAP explainer (TreeExplainer for tree models, otherwise KernelExplainer with a small background sample). Provides `explain_one(...)` and `global_importance(...)`. Must degrade gracefully if SHAP fails (return empty list).

## 7. Streamlit App Specification
| Page | Content |
|---|---|
| Predict | Form for application fields, threshold slider, decision card, probability gauge, top-5 SHAP factors bar chart |
| Model Comparison | Table of all MLflow runs (F1, precision, recall, AUC), bar chart, ROC curves, confusion matrix of best model |
| Telemetry | Predictions count, approval rate over time, latency histogram (mean, p95), recent predictions table, data-quality summary |
| Drift | Run drift on uploaded CSV or simulated drift; PSI table with colored status; before/after metrics |
| Fairness | Per-group metrics table, gap indicators, short written interpretation |

Use `st.cache_resource` for model loading and `st.cache_data` for reports. No network calls.

## 8. Testing Strategy
| Test | Checks |
|---|---|
| `test_preprocessing.py` | No NaNs after pipeline; "3+" converted; unseen category does not crash |
| `test_features.py` | Feature formulas correct; zero income does not produce inf |
| `test_drift.py` | PSI about 0 for identical data; PSI large for shifted data |
| `test_telemetry.py` | Log and read round-trip; `update_true_label` works |
| Smoke test | `predict_one` on a sample row returns valid output and writes one log row |

## 9. Reproducibility and Tooling
- Global seed from config applied to numpy, sklearn, xgboost.
- `Makefile` targets: `setup`, `train`, `app`, `drift-demo`, `fairness`, `test`.
- MLflow UI: `mlflow ui --backend-store-uri ./mlruns`.

## 10. Security and Privacy
- Only the public anonymized dataset is used.
- No secrets in the repo; none are required.
- Sensitive attributes are never model inputs.

## 11. Definition of Done
All acceptance criteria in the PRD pass, tests are green, README documents setup and usage, and screenshots of MLflow, the app pages and the drift demo are saved in `reports/screenshots/`.
