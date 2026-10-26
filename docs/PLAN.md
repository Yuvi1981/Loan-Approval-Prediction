# Stepwise Implementation Plan for Loan Approval Prediction with ML Telemetry

## Guidelines
- Follow `docs/PRD.md` and `docs/TRD.md` exactly.
- Keep code modular under `src/`. Use type hints and docstrings.
- Never use Gender or Married as model inputs.
- Fit all preprocessing on training data only to avoid data leakage.
- Run tests and verifications at each phase.

## Phase Overview
- **Phase 0: Project Setup**: Directory skeleton, requirements, virtual environment, Makefile, starter config.yaml, .gitignore, README.
- **Phase 1: Data and Quality Checks**: `src/config.py`, `src/data_quality.py`, EDA notebook, `reports/data_quality.json`.
- **Phase 2: Preprocessing and Feature Engineering**: `src/preprocessing.py`, `src/features.py`, unit tests in `tests/`.
- **Phase 3: Baseline Models & Evaluation**: `src/evaluate.py`, baseline models in `src/train.py`, MLflow tracking.
- **Phase 4: Advanced Models, Tuning, Imbalance**: RF, XGBoost, Ensembles, Hyperparameter tuning, SMOTE comparison, `models/best_model.joblib`.
- **Phase 5: Prediction, Explainability, Telemetry Core**: `src/explain.py` (SHAP), `src/telemetry.py` (SQLite), `src/predict.py`, traffic generation script, tests.
- **Phase 6: Drift Monitoring**: `src/drift.py` (PSI & KS-test), `src/simulate_drift.py`, `reports/drift_demo.json` and chart.
- **Phase 7: Fairness Audit**: `src/fairness.py` (Fairlearn MetricFrame), `reports/fairness.json`, `reports/fairness_summary.md`.
- **Phase 8: Streamlit Dashboard**: Multi-page app (`app/streamlit_app.py`, 5 pages), testing, interactive verification.
- **Phase 9: Polish, Tests, Documentation**: Comprehensive unit tests, README, linting.
- **Phase 10: Report & Presentation**: Project report outline and 12-slide presentation deck.
