# PRD: Loan Approval Prediction with ML Telemetry

| Field | Value |
|---|---|
| Project name | `loan-telemetry` |
| Type | College ML mini project (team of 3 to 4) |
| Version | 1.0 |
| Duration | About 4 weeks |

## 1. Overview
A machine learning system that predicts whether a loan application should be **Approved (1)** or **Rejected (0)**, wrapped with **ML telemetry**: experiment tracking, data-quality checks, prediction logging, drift monitoring and fairness auditing, all visible in a Streamlit dashboard.

## 2. Problem Statement
Banks screen loan applications manually, which is slow and inconsistent. A model can speed up screening, but a model that is not monitored can silently degrade or behave unfairly. This project builds both the model and the monitoring around it.

## 3. Goals
1. Cover the full ML workflow required by the course: preprocessing, feature engineering, model building, evaluation, and advanced techniques (ensembles, tuning, imbalance handling, explainability).
2. Add a telemetry layer so every training run and every prediction is recorded and can be audited.
3. Deliver an interactive app that demonstrates prediction, explanation and monitoring.
4. Demonstrate ethical ML practice through a fairness audit.

## 4. Non-Goals
- Not a production banking system; no real customer data, authentication or regulatory compliance.
- No deep learning or reinforcement learning (optional stretch only).
- No cloud deployment (optional stretch only).

## 5. Users and Personas
| Persona | Need |
|---|---|
| **Loan officer (demo user)** | Enter an application, get a decision, a probability and the reasons behind it |
| **ML engineer / student team** | Compare experiments, reproduce runs, see how the model behaves over time |
| **Evaluator / professor** | See the rubric items covered clearly, with evidence and ethics discussion |

## 6. User Stories
- As a loan officer, I enter applicant details and get Approved/Rejected with a probability.
- As a loan officer, I see the top factors that pushed the decision (SHAP).
- As an ML engineer, I compare all trained models by F1, precision, recall and AUC in one place.
- As an ML engineer, I see a drift report when new data differs from training data.
- As an evaluator, I see approval rates and recall per demographic group.

## 7. Functional Requirements
| ID | Requirement | Priority |
|---|---|---|
| FR-1 | Load raw CSV and run data-quality checks (missing %, ranges, schema, duplicates) and save a report | Must |
| FR-2 | Preprocess: impute missing values, clean `Dependents`, cap outliers, encode categoricals, scale numerics | Must |
| FR-3 | Engineer features: `Total_Income`, `EMI`, `Balance_Income`, `Loan_to_Income`, `Log_Total_Income` | Must |
| FR-4 | Feature selection using correlation, chi-square and tree importance; optional PCA experiment | Must |
| FR-5 | Train baselines: Logistic Regression, Decision Tree, KNN, Naive Bayes | Must |
| FR-6 | Train advanced models: Random Forest, XGBoost (or GradientBoosting), Voting and Stacking ensembles | Must |
| FR-7 | Hyperparameter tuning with RandomizedSearchCV then GridSearchCV | Must |
| FR-8 | Handle class imbalance (class weights or SMOTE) and compare with and without | Should |
| FR-9 | Evaluate with accuracy, precision, recall, F1, ROC-AUC, confusion matrix using stratified 5-fold CV and a held-out test set | Must |
| FR-10 | Log every run to MLflow: params, metrics, plots, model artifact | Must |
| FR-11 | Log every prediction (timestamp, inputs, probability, decision, latency, model version) to SQLite | Must |
| FR-12 | Drift detection (PSI and KS-test) between training data and incoming data, with a simulated-drift demo | Must |
| FR-13 | Fairness audit per group (gender, married, property area): approval rate, recall, demographic parity and equal opportunity difference | Must |
| FR-14 | SHAP explanations for single predictions and global feature importance | Should |
| FR-15 | Streamlit app with pages: Predict, Model Comparison, Telemetry, Drift, Fairness | Must |
| FR-16 | Threshold selector so users can trade precision against recall | Should |

## 8. Non-Functional Requirements
- **Reproducibility:** fixed random seed, pinned `requirements.txt`, single `config.yaml`.
- **Performance:** single prediction (including logging) under 500 ms on a laptop.
- **Usability:** app runs with one command; clear labels and no ML jargon on the Predict page.
- **Maintainability:** modular `src/` package, type hints, docstrings, unit tests for core functions.
- **Privacy:** only the public anonymized dataset; no personal data stored.

## 9. Success Metrics
| Metric | Target |
|---|---|
| Test accuracy (best model) | >= 80% |
| Test F1 and ROC-AUC | F1 >= 0.85 (approved class), AUC >= 0.75 |
| Runs tracked in MLflow | >= 10 |
| Drift demo | Drift flagged on simulated shift, and F1 drop shown |
| Fairness | Metrics reported for all groups; any gap above 0.10 flagged and discussed |
| Unit tests | Pass with `pytest` |

## 10. Risks and Mitigations
| Risk | Mitigation |
|---|---|
| Small dataset (about 614 rows) gives unstable metrics | Stratified CV, report mean and std, avoid over-tuning |
| Library API changes (Evidently, SHAP) | Implement PSI/KS manually with SciPy; treat Evidently as optional |
| Model learns biased proxies | Exclude sensitive attributes from inputs; audit outcomes anyway |
| Scope creep | Follow phases in the implementation plan; stretch goals only after Phase 8 |

## 11. Milestones
| Week | Milestone |
|---|---|
| 1 | Data checks, EDA, preprocessing, features |
| 2 | Baseline models with MLflow tracking |
| 3 | Ensembles, tuning, SHAP, fairness and drift modules |
| 4 | Streamlit dashboard, tests, report and presentation |

## 12. Acceptance Criteria
1. `make train` (or `python -m src.train`) trains all models and logs them to MLflow.
2. `streamlit run app/streamlit_app.py` launches the app with all five pages working.
3. A prediction made in the app appears in the telemetry page within seconds.
4. The drift demo script produces a drift report and a metric-drop chart.
5. The fairness page shows per-group metrics and flags gaps above 0.10.
6. `pytest` passes.
7. README lets a new person reproduce everything from scratch.

## 13. Ethics Statement
Gender, marital status and similar attributes are not model inputs. Outcomes are still audited across these groups because other features can act as proxies. The tool is decision support only; real lending needs human review, explainability and legal compliance.
