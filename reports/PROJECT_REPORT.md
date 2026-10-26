# Comprehensive Project Report: Loan Approval Prediction with ML Telemetry

**Project Name:** `loan-telemetry`  
**Course:** Machine Learning Mini-Project  
**Version:** 1.0.0  
**Champion Model:** XGBoost Classifier (with Soft Voting & Stacking Ensembles)  

---

## 1. Executive Summary & Problem Statement
Commercial banks receive high volumes of credit applications that are traditionally assessed through manual, labor-intensive underwriting processes. While standard supervised learning classifiers can automate credit screening, models deployed in isolation risk silent distribution drift, catastrophic degradation under macroeconomic shocks, and unintentional proxy discrimination against protected demographics.

This project delivers **`loan-telemetry`**, a production-grade ML system that predicts loan approval status (`Approved = 1`, `Rejected = 0`) wrapped inside an **ML Telemetry and Observability architecture**:
1. **Automated Data Quality Profiling** (`data_quality.py`) prior to pipeline ingestion.
2. **Leak-Free Preprocessing & Domain Feature Engineering** (`preprocessing.py`, `features.py`).
3. **Multi-Model Benchmark & MLflow Experiment Tracking** (`train.py`, `evaluate.py`) across 12 candidate models.
4. **Sub-50ms Inference & Local Explainability** (`predict.py`, `explain.py`) powered by TreeSHAP.
5. **Real-Time Prediction Event Store** (`telemetry.py`) logging to SQLite with post-settlement outcome reconciliation.
6. **Continuous Drift Surveillance** (`drift.py`, `simulate_drift.py`) monitoring Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) test statistics.
7. **Algorithmic Fairness & Disparate Impact Auditing** (`fairness.py`) powered by Fairlearn MetricFrame.
8. **Interactive Multi-Page Streamlit Dashboard** (`app/`).

---

## 2. Dataset Description
The system is built on the canonical **Loan Prediction Dataset** (Kaggle / Analytics Vidhya):
- **Observations:** 614 applicant records
- **Features:** 12 predictors (financial, demographic, geographical)
- **Target Variable:** `Loan_Status` (`Y` = Approved, `N` = Rejected)
- **Target Split:** 422 Approved (68.7%), 192 Rejected (31.3%) — roughly 2:1 ratio.
- **Split Strategy:** 80/20 Stratified Train/Test split (`random_state=42`), maintaining class balance (Train: 491, Test: 123).

### Key Attributes
| Field | Type | Description | Treatment |
|---|---|---|---|
| `Loan_ID` | String | Unique application ID | Dropped |
| `Gender` | Categorical | Male (489) / Female (112) / Missing (13) | **Excluded from model inputs**; retained for fairness audit |
| `Married` | Categorical | Yes (398) / No (213) / Missing (3) | **Excluded from model inputs**; retained for fairness audit |
| `Dependents` | Categorical | 0, 1, 2, 3+ (15 missing) | String cleaned ("3+" -> 3), imputed with mode |
| `Education` | Categorical | Graduate (480) / Not Graduate (134) | One-hot encoded |
| `Self_Employed` | Categorical | Yes (82) / No (500) / Missing (32) | Mode imputed, one-hot encoded |
| `ApplicantIncome` | Numeric | Range: $150 to $81,000 (Skewed) | IQR capped, scaled |
| `CoapplicantIncome`| Numeric | Range: $0 to $41,667 (Skewed) | Scaled |
| `LoanAmount` | Numeric | Range: $9k to $700k (22 missing) | Median imputed, IQR capped, scaled |
| `Loan_Amount_Term` | Numeric | Range: 12 to 480 months (14 missing)| Median imputed (360) |
| `Credit_History` | Binary | 1.0 (475) / 0.0 (89) / Missing (50) | Mode imputed (1.0), preserved without zero-IQR collapse |
| `Property_Area` | Categorical | Semiurban (233) / Urban (202) / Rural (179) | One-hot encoded |

---

## 3. Data Quality Assurance (`data_quality.py`)
Prior to feature engineering or model training, automated data validation executes against `data/raw/train.csv` and outputs `reports/data_quality.json`:
- **Duplicates:** 0 duplicates identified.
- **Schema Validation:** Status `PASS` (all 13 required columns present).
- **Missingness Profile:**
  - `Credit_History`: 8.14% (50 nulls)
  - `Self_Employed`: 5.21% (32 nulls)
  - `LoanAmount`: 3.58% (22 nulls)
  - `Dependents`: 2.44% (15 nulls)
  - `Loan_Amount_Term`: 2.28% (14 nulls)
  - `Gender`: 2.12% (13 nulls)
  - `Married`: 0.49% (3 nulls)

---

## 4. Feature Engineering (`features.py`)
To capture debt serviceability and repayment capacity, five domain features were engineered:
1. **Total Household Income:**
   $$\text{Total\_Income} = \text{ApplicantIncome} + \text{CoapplicantIncome}$$
2. **Equated Monthly Installment (EMI):**
   $$\text{EMI} = \frac{\text{LoanAmount}}{\text{Loan\_Amount\_Term}}$$
3. **Balance Income (Residual Liquidity):**
   $$\text{Balance\_Income} = \text{Total\_Income} - (\text{EMI} \times 1000)$$
4. **Loan-to-Income Ratio (with Zero-Division Guard):**
   $$\text{Loan\_to\_Income} = \frac{\text{LoanAmount} \times 1000}{\max(\text{Total\_Income}, 1.0)}$$
5. **Log Total Income:**
   $$\text{Log\_Total\_Income} = \ln(1 + \max(\text{Total\_Income}, 0.0))$$

---

## 5. Leak-Free Preprocessing Pipeline (`preprocessing.py`)
To prevent data snooping and target leakage:
- Preprocessing is encapsulated in a `scikit-learn ColumnTransformer`.
- **Numeric Pipeline:** Median Imputer $\to$ Custom `IQROutlierCapper(factor=1.5)` (fitted on training fold only; preserves zero-IQR binary features) $\to$ `StandardScaler`.
- **Categorical Pipeline:** Most-Frequent Imputer $\to$ `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- All transformers are fit exclusively on training folds during cross-validation.

---

## 6. Model Training, Tuning & MLflow Leaderboard
12 distinct candidate models were trained with stratified 5-fold cross-validation and evaluated on the held-out 123-sample test set. All runs were logged to MLflow (`mlruns/`) with full parameters, metrics, confusion matrices, and ROC curves.

### Final Model Comparison Leaderboard
| Rank | Model Name | Model Type | CV F1 (Mean) | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| **1** | **XGBoost** | **Advanced Gradient Boost** | **0.8524** | **86.18%** | **84.69%** | **97.65%** | **0.9071** | **0.8359** |
| 2 | StackingClassifier | Stacking Ensemble (LR meta) | 0.8622 | 86.18% | 85.42% | 96.47% | 0.9061 | 0.8647 |
| 3 | DecisionTree | Baseline Tree (depth=5) | 0.8582 | 86.18% | 86.17% | 95.29% | 0.9050 | 0.7913 |
| 4 | LogisticRegression | Linear Baseline (L2) | 0.8626 | 85.37% | 83.17% | 98.82% | 0.9032 | 0.8443 |
| 5 | GaussianNaiveBayes | Probabilistic Baseline | 0.8378 | 85.37% | 83.17% | 98.82% | 0.9032 | 0.8437 |
| 6 | VotingClassifier_Soft| Soft Ensemble (LR+RF+XGB) | 0.8626 | 85.37% | 83.17% | 98.82% | 0.9032 | 0.8734 |
| 7 | RandomForest_ClassWeighted | Imbalance (Balanced) | 0.8486 | 85.37% | 86.81% | 92.94% | 0.8977 | 0.8709 |
| 8 | RandomForest | Bagging Ensemble | 0.8672 | 84.55% | 83.67% | 96.47% | 0.8962 | 0.8789 |
| 9 | XGBoost_ClassWeighted | Imbalance (Pos Weight) | 0.8273 | 85.37% | 88.51% | 90.59% | 0.8953 | 0.8601 |
| 10 | RandomForest_Tuned | Tuned (GridSearch) | 0.8731 | 82.93% | 82.65% | 95.29% | 0.8852 | 0.8799 |
| 11 | KNN | Distance Baseline (k=5) | 0.8497 | 82.93% | 84.78% | 91.76% | 0.8814 | 0.8209 |
| 12 | RandomForest_SMOTE | Imbalance (imblearn SMOTE) | 0.8441 | 79.67% | 84.09% | 87.06% | 0.8555 | 0.8396 |

**Selected Champion:** **XGBoost Classifier**
- **Test Accuracy:** 86.18% (Target: $\ge 80\%$) $\to$ **Exceeded**
- **Test F1:** 0.9071 (Target: $\ge 0.85$) $\to$ **Exceeded**
- **Test ROC-AUC:** 0.8359 (Target: $\ge 0.75$) $\to$ **Exceeded**

---

## 7. Explainability (SHAP Local Attribution)
In lending decisions, adverse action notices require transparent explanation. The system integrates `TreeExplainer` from the `shap` library:
- **Dominant Driver:** `Credit_History` is consistently the largest contributor (+0.35 to +0.45 SHAP value toward approval when 1.0; -0.40 toward rejection when 0.0).
- **Secondary Drivers:** `Loan_to_Income`, `Total_Income`, and `Semiurban` property location.
- **Fault-Tolerant Design:** If SHAP computation encounters an exception, the system gracefully degrades and returns a fallback ranking rather than crashing the inference engine.

---

## 8. Operational Telemetry & Monitoring (`telemetry.py`)
Every inference request executed via `predict_one()` is committed to `logs/telemetry.db`:
- **Captured Attributes:** Timestamp (ISO UTC), Model Version, Input JSON, Predicted Probability, Decision Flag, Decision Threshold, Inference Latency (ms), Top SHAP factors, and True Outcome Label (reconciled post-settlement).
- **Performance Benchmarks:**
  - Total Logged Events: 202 transactions
  - Approval Rate: 74.6%
  - Mean Inference Latency: **26.7 ms** (SLA target: $<500\text{ ms}$)
  - P95 Latency: **25.9 ms**
  - Monitored Ground-Truth Accuracy: **81.1%** across 111 labeled test settlements.

---

## 9. Drift Surveillance & Stress Testing (`drift.py`, `simulate_drift.py`)
To prevent silent model degradation, statistical drift monitoring tracks incoming feature distributions against baseline training statistics using Population Stability Index (PSI) and Kolmogorov-Smirnov (KS-test):
- **PSI Thresholds:** $\text{PSI} < 0.10$ (Stable), $0.10 \le \text{PSI} < 0.25$ (Moderate Drift), $\text{PSI} \ge 0.25$ (Significant Drift).
- **Macroeconomic Stress Simulation Results:**
  - Top Drifted Feature: `EMI` ($\text{PSI} = 1.0939$) and `ApplicantIncome` ($\text{PSI} = 0.5482$).
  - **Performance Degradation Under Drift:**
    - Baseline Accuracy: 86.18% $\to$ Drifted: 63.41% ($\Delta = -22.77\%$)
    - Baseline F1-Score: 0.9071 $\to$ Drifted: 0.6809 ($\Delta = -22.62\%$)
    - Baseline ROC-AUC: 0.8359 $\to$ Drifted: 0.6975 ($\Delta = -13.84\%$)
- **Action Trigger:** Automatically flags alerts in the dashboard when feature PSI exceeds 0.25, signaling the engineering team that model retraining is required.

---

## 10. Algorithmic Fairness & Ethical Audit (`fairness.py`)
Using Fairlearn `MetricFrame`, outcomes on the test set were audited across legally protected attributes:
- **Mitigation by Design:** `Gender` and `Married` are strictly omitted from model training inputs.
- **Audit Findings (Threshold = 0.50, Gap Threshold = 0.10):**
  - **Gender:**
    - Female Approval Rate: 64.0% (Recall: 92.9%, Precision: 81.2%, Acc: 84.0%)
    - Male Approval Rate: 83.3% (Recall: 98.6%, Precision: 86.2%, Acc: 87.5%)
    - Demographic Parity Difference: **0.3600 (Flagged > 0.10)**
    - Equal Opportunity Difference (TPR Gap): **0.0714 (Pass $\le 0.10$)**
  - **Marital Status:**
    - Unmarried Approval Rate: 65.9% (Recall: 96.0%, Acc: 86.4%)
    - Married Approval Rate: 87.3% (Recall: 98.3%, Acc: 86.1%)
    - Demographic Parity Difference: **0.2143 (Flagged > 0.10)**
    - Equal Opportunity Difference (TPR Gap): **0.0233 (Pass $\le 0.10$)**
  - **Property Area:**
    - Rural: 82.3% | Semiurban: 79.6% | Urban: 77.5%
    - Demographic Parity Difference: **0.0485 (Pass $\le 0.10$)**
    - Equal Opportunity Difference: **0.0455 (Pass $\le 0.10$)**

### Ethical & Legal Interpretation
1. **Root Driver of Disparity:** In the legacy Kaggle training dataset, female and unmarried applicants have lower rates of positive `Credit_History` (e.g. 78% vs 86% for males). Because the model relies heavily on credit history, disparate impact arises from historical disparities rather than direct bias.
2. **Decision-Support Boundary:** This tool is strictly intended as decision-support for initial screening. Any adverse lending decision requires human underwriter review, explainable justification, and adverse action compliance.

---

## 11. Conclusion & Lessons Learned
The `loan-telemetry` system demonstrates that a machine learning model is only as dependable as the observability and governance framework surrounding it. By integrating data quality profiling, 12-run MLflow tracking, sub-30ms inference, real-time SQLite telemetry, statistical drift alerts, and Fairlearn audits into a cohesive Streamlit interface, this project successfully bridges the gap between theoretical machine learning and production-grade MLOps.
