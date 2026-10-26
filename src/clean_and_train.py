"""
End-to-End Data Cleaning, Feature Engineering, Model Training, and Prediction Output Script.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.explain import LoanExplainer
from src.features import add_features
from src.preprocessing import build_preprocessor, clean_dependents, extract_features_and_target


def clean_and_prepare_dataset(raw_csv_path: str = "data/raw/train.csv") -> pd.DataFrame:
    """Load raw dataset, clean columns, impute missing values, cap outliers, and add engineered features.

    Returns:
        pd.DataFrame: Fully cleaned and enriched dataset.
    """
    print("=" * 70)
    print("STEP 1: LOADING RAW DATASET")
    print("=" * 70)
    df_raw = pd.read_csv(raw_csv_path)
    print(f"Loaded raw dataset from: {raw_csv_path}")
    print(f"Raw Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns\n")

    print("Missing values in raw dataset:")
    missing_raw = df_raw.isna().sum()
    print(missing_raw[missing_raw > 0].to_string())
    print("-" * 70)

    print("\n" + "=" * 70)
    print("STEP 2: CLEANING & IMPUTATION")
    print("=" * 70)

    df_cleaned = df_raw.copy()

    # 1. Clean Dependents ('3+' -> '3')
    df_cleaned = clean_dependents(df_cleaned)
    mode_dependents = df_cleaned["Dependents"].mode()[0]
    df_cleaned["Dependents"] = df_cleaned["Dependents"].fillna(mode_dependents)
    print(f"Cleaned 'Dependents': Replaced '3+' with 3, imputed {missing_raw.get('Dependents', 0)} missing with mode '{mode_dependents}'.")

    # 2. Impute Categoricals with Mode
    mode_gender = df_cleaned["Gender"].mode()[0]
    df_cleaned["Gender"] = df_cleaned["Gender"].fillna(mode_gender)

    mode_married = df_cleaned["Married"].mode()[0]
    df_cleaned["Married"] = df_cleaned["Married"].fillna(mode_married)

    mode_self_emp = df_cleaned["Self_Employed"].mode()[0]
    df_cleaned["Self_Employed"] = df_cleaned["Self_Employed"].fillna(mode_self_emp)

    mode_credit = df_cleaned["Credit_History"].mode()[0]
    df_cleaned["Credit_History"] = df_cleaned["Credit_History"].fillna(mode_credit)
    print(f"Imputed categorical missing values: Gender -> '{mode_gender}', Married -> '{mode_married}', Self_Employed -> '{mode_self_emp}', Credit_History -> {mode_credit}.")

    # 3. Impute Continuous Numerics with Median
    med_loan_amt = df_cleaned["LoanAmount"].median()
    df_cleaned["LoanAmount"] = df_cleaned["LoanAmount"].fillna(med_loan_amt)

    med_loan_term = df_cleaned["Loan_Amount_Term"].median()
    df_cleaned["Loan_Amount_Term"] = df_cleaned["Loan_Amount_Term"].fillna(med_loan_term)
    print(f"Imputed numeric missing values: LoanAmount -> {med_loan_amt}k, Loan_Amount_Term -> {med_loan_term} months.")

    # 4. Outlier Capping (IQR Method on skewed financial columns)
    cols_to_cap = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
    print("\nApplying IQR Outlier Capping on skewed numeric variables:")
    for col in cols_to_cap:
        q25 = df_cleaned[col].quantile(0.25)
        q75 = df_cleaned[col].quantile(0.75)
        iqr = q75 - q25
        lower = max(0.0, q25 - 1.5 * iqr)
        upper = q75 + 1.5 * iqr
        outliers_count = ((df_cleaned[col] < lower) | (df_cleaned[col] > upper)).sum()
        df_cleaned[col] = df_cleaned[col].clip(lower=lower, upper=upper)
        print(f" - {col:18s}: [Q1={q25:7.1f}, Q3={q75:7.1f}, IQR={iqr:7.1f}] -> Bounds [{lower:7.1f}, {upper:7.1f}] (Capped {outliers_count} outliers)")

    # 5. Feature Engineering
    print("\n" + "=" * 70)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 70)
    df_cleaned = add_features(df_cleaned)
    print("Created 5 domain features:")
    print(" 1. Total_Income      = ApplicantIncome + CoapplicantIncome")
    print(" 2. EMI               = LoanAmount / Loan_Amount_Term")
    print(" 3. Balance_Income    = Total_Income - (EMI * 1000)")
    print(" 4. Loan_to_Income    = (LoanAmount * 1000) / Total_Income (zero-division guarded)")
    print(" 5. Log_Total_Income  = log1p(Total_Income)")

    # Verify no missing values remain
    remaining_missing = df_cleaned.isna().sum().sum()
    print(f"\nMissing values check after cleaning: {remaining_missing} null values remaining.")
    assert remaining_missing == 0, "Error: missing values still detected!"

    # Save cleaned dataset
    out_path = Path("data/processed/cleaned_loan_data.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(out_path, index=False)
    print(f"Saved fully cleaned dataset to: {out_path} ({len(df_cleaned)} rows, {df_cleaned.shape[1]} columns)")
    print("=" * 70 + "\n")

    return df_cleaned


def train_and_output_results(df_cleaned: pd.DataFrame):
    """Train machine learning models on the cleaned dataset, evaluate metrics, and display sample predictions."""
    print("=" * 70)
    print("STEP 4: MODEL TRAINING & EVALUATION (STRATIFIED 80/20 SPLIT)")
    print("=" * 70)

    X, y, sensitive_df = extract_features_and_target(df_cleaned)
    
    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"Training Samples: {len(X_train)} ({y_train.sum()} Approved, {len(y_train) - y_train.sum()} Rejected)")
    print(f"Testing Samples : {len(X_test)} ({y_test.sum()} Approved, {len(y_test) - y_test.sum()} Rejected)")

    num_cols = [
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
        "Credit_History", "Total_Income", "EMI", "Balance_Income", "Loan_to_Income", "Log_Total_Income"
    ]
    cat_cols = ["Dependents", "Education", "Self_Employed", "Property_Area"]

    preprocessor = build_preprocessor(num_cols, cat_cols)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=42),
    }

    results = []
    trained_pipelines = {}

    for name, clf in models.items():
        from sklearn.pipeline import Pipeline
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe

        probs = pipe.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.50).astype(int)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs)

        results.append({
            "Model": name,
            "Accuracy": f"{acc * 100:.2f}%",
            "Precision": f"{prec * 100:.2f}%",
            "Recall": f"{rec * 100:.2f}%",
            "F1-Score": f"{f1:.4f}",
            "ROC-AUC": f"{auc:.4f}",
        })

    results_df = pd.DataFrame(results)
    print("\nMODEL PERFORMANCE LEADERBOARD:")
    print(results_df.to_string(index=False))
    print("-" * 70)

    # Champion Model Evaluation
    best_model_name = "XGBoost"
    best_pipe = trained_pipelines[best_model_name]
    best_probs = best_pipe.predict_proba(X_test)[:, 1]
    best_preds = (best_probs >= 0.50).astype(int)

    print(f"\nDETAILED CLASSIFICATION REPORT FOR CHAMPION MODEL ({best_model_name}):")
    print(classification_report(y_test, best_preds, target_names=["Rejected (0)", "Approved (1)"]))

    cm = confusion_matrix(y_test, best_preds)
    print("CONFUSION MATRIX:")
    print(f"True Negatives  (Correct Rejections): {cm[0, 0]}")
    print(f"False Positives (Incorrect Approvals): {cm[0, 1]}")
    print(f"False Negatives (Incorrect Rejections): {cm[1, 0]}")
    print(f"True Positives  (Correct Approvals) : {cm[1, 1]}")
    print("=" * 70)

    # Save champion model to disk
    joblib.dump(best_pipe, "models/best_model.joblib")
    print("Champion model saved to models/best_model.joblib\n")

    # STEP 5: PREDICTIONS ON DIVERSE REALISTIC APPLICANTS
    print("=" * 70)
    print("STEP 5: PREDICTION OUTPUTS ON SAMPLE APPLICANTS")
    print("=" * 70)

    explainer = LoanExplainer(best_pipe, background_data=X_train.head(50))

    sample_applicants = [
        {
            "Applicant": "Applicant A (High Income, Perfect Credit, Low Debt)",
            "ApplicantIncome": 7500, "CoapplicantIncome": 3000, "LoanAmount": 130,
            "Loan_Amount_Term": 360, "Credit_History": 1.0, "Dependents": "0",
            "Education": "Graduate", "Self_Employed": "No", "Property_Area": "Semiurban"
        },
        {
            "Applicant": "Applicant B (Good Income, But History of Defaults)",
            "ApplicantIncome": 6000, "CoapplicantIncome": 1500, "LoanAmount": 150,
            "Loan_Amount_Term": 360, "Credit_History": 0.0, "Dependents": "1",
            "Education": "Graduate", "Self_Employed": "No", "Property_Area": "Urban"
        },
        {
            "Applicant": "Applicant C (Low Income, Requesting Large Loan)",
            "ApplicantIncome": 1800, "CoapplicantIncome": 0, "LoanAmount": 220,
            "Loan_Amount_Term": 360, "Credit_History": 1.0, "Dependents": "2",
            "Education": "Not Graduate", "Self_Employed": "Yes", "Property_Area": "Rural"
        },
        {
            "Applicant": "Applicant D (Moderate Income, Coapplicant Support, Good Credit)",
            "ApplicantIncome": 4000, "CoapplicantIncome": 2500, "LoanAmount": 120,
            "Loan_Amount_Term": 360, "Credit_History": 1.0, "Dependents": "1",
            "Education": "Graduate", "Self_Employed": "No", "Property_Area": "Urban"
        },
        {
            "Applicant": "Applicant E (Self-Employed, High Debt Burden, Default History)",
            "ApplicantIncome": 3200, "CoapplicantIncome": 0, "LoanAmount": 180,
            "Loan_Amount_Term": 180, "Credit_History": 0.0, "Dependents": "3+",
            "Education": "Not Graduate", "Self_Employed": "Yes", "Property_Area": "Rural"
        }
    ]

    for app in sample_applicants:
        app_name = app.pop("Applicant")
        raw_df = pd.DataFrame([app])
        feat_df = add_features(raw_df)
        clean_df = clean_dependents(feat_df)
        
        prob = float(best_pipe.predict_proba(clean_df)[0, 1])
        decision = "APPROVED (1)" if prob >= 0.50 else "REJECTED (0)"
        status_tag = "[APPROVED]" if prob >= 0.50 else "[REJECTED]"

        # Get SHAP explanation factors
        factors = explainer.explain_one(clean_df, top_k=3)

        print(f"\n{status_tag} {app_name}")
        print(f"   Inputs     : Income=${app['ApplicantIncome']+app['CoapplicantIncome']} | Loan=${app['LoanAmount']}k | CreditHistory={app['Credit_History']} | Term={app['Loan_Amount_Term']}m")
        print(f"   Decision   : {decision} | Probability of Approval: {prob:.1%}")
        print("   Key Drivers:")
        for f in factors:
            arrow = "(+) Push to Approve" if f["impact"] > 0 else "(-) Push to Reject"
            print(f"     * {f['feature']:16s}: impact = {f['impact']:+.3f} [{arrow}]")
        print("-" * 70)


if __name__ == "__main__":
    cleaned_df = clean_and_prepare_dataset("data/raw/train.csv")
    train_and_output_results(cleaned_df)
