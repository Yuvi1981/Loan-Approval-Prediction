"""
LOANIQ.ML - Predict Page
Real-time loan risk evaluation with clean white cards, teal buttons,
and local SHAP feature attributions matching FINbank.IO design tokens.
"""

import plotly.graph_objects as go
import streamlit as st

from app.components import top_header
from src.predict import load_model_and_meta, predict_one


def render_predict():
    """Render the Predict page with loan screening and SHAP explainability."""
    top_header(greeting="Loan Application Assessment", date_str="Live ML Screening Engine")

    _, meta = load_model_and_meta()
    default_thresh = float(meta.get("default_threshold", 0.50))
    model_name = meta.get("model_name", "XGBoost")
    version = meta.get("version", "1.0.0")

    # Profile Presets
    st.markdown('<div class="card-title-muted" style="margin-bottom: 6px;">Quick Presets:</div>', unsafe_allow_html=True)
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        if st.button("★ Prime Applicant", key="preset_prime", use_container_width=True):
            st.session_state.app_inc = 8500
            st.session_state.coapp_inc = 3200
            st.session_state.loan_amt = 130
            st.session_state.credit = 1.0
            st.session_state.area = "Semiurban"
            st.session_state.edu = "Graduate"
            st.session_state.dep = "0"
            st.session_state.self_emp = "No"
            st.session_state.gender = "Female"
            st.session_state.married = "Yes"
    with p_col2:
        if st.button("✕ High Risk Default", key="preset_default", use_container_width=True):
            st.session_state.app_inc = 6200
            st.session_state.coapp_inc = 1500
            st.session_state.loan_amt = 160
            st.session_state.credit = 0.0
            st.session_state.area = "Urban"
            st.session_state.edu = "Graduate"
            st.session_state.dep = "1"
            st.session_state.self_emp = "No"
            st.session_state.gender = "Male"
            st.session_state.married = "No"
    with p_col3:
        if st.button("Δ Overleveraged Rural", key="preset_rural", use_container_width=True):
            st.session_state.app_inc = 1800
            st.session_state.coapp_inc = 0
            st.session_state.loan_amt = 240
            st.session_state.credit = 1.0
            st.session_state.area = "Rural"
            st.session_state.edu = "Not Graduate"
            st.session_state.dep = "2"
            st.session_state.self_emp = "Yes"
            st.session_state.gender = "Male"
            st.session_state.married = "Yes"
    with p_col4:
        if st.button("∞ Dual Income", key="preset_dual", use_container_width=True):
            st.session_state.app_inc = 4500
            st.session_state.coapp_inc = 3500
            st.session_state.loan_amt = 125
            st.session_state.credit = 1.0
            st.session_state.area = "Urban"
            st.session_state.edu = "Graduate"
            st.session_state.dep = "1"
            st.session_state.self_emp = "No"
            st.session_state.gender = "Female"
            st.session_state.married = "Yes"

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    form_col, result_col = st.columns([1.35, 1])

    # Left: Loan Application Form
    with form_col:
        st.markdown(
            f"""
            <div class="fin-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div style="font-size: 1.15rem; font-weight: 800; color: #1A1D1F;">Applicant Financial Profile</div>
                    <span class="pill-status success">{model_name} v{version}</span>
                </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("loan_screening_form"):
            r1_1, r1_2 = st.columns(2)
            with r1_1:
                app_inc = st.number_input(
                    "Applicant Monthly Income ($)",
                    min_value=0,
                    value=st.session_state.get("app_inc", 5000),
                    step=250,
                )
            with r1_2:
                coapp_inc = st.number_input(
                    "Coapplicant Monthly Income ($)",
                    min_value=0,
                    value=st.session_state.get("coapp_inc", 1500),
                    step=250,
                )

            r2_1, r2_2, r2_3 = st.columns(3)
            with r2_1:
                loan_amt = st.number_input(
                    "Loan Amount ($k)",
                    min_value=10,
                    value=st.session_state.get("loan_amt", 140),
                    step=5,
                )
            with r2_2:
                loan_term = st.selectbox("Term (Months)", options=[360, 240, 180, 120], index=0)
            with r2_3:
                credit_hist = st.selectbox(
                    "Credit Guidelines Met?",
                    options=[1.0, 0.0],
                    format_func=lambda x: "Yes (1.0 — Good)" if x == 1.0 else "No (0.0 — Default)",
                    index=0 if st.session_state.get("credit", 1.0) == 1.0 else 1,
                )

            r3_1, r3_2, r3_3 = st.columns(3)
            with r3_1:
                property_area = st.selectbox(
                    "Property Area",
                    options=["Semiurban", "Urban", "Rural"],
                    index=["Semiurban", "Urban", "Rural"].index(st.session_state.get("area", "Semiurban")),
                )
            with r3_2:
                education = st.selectbox(
                    "Education",
                    options=["Graduate", "Not Graduate"],
                    index=["Graduate", "Not Graduate"].index(st.session_state.get("edu", "Graduate")),
                )
            with r3_3:
                dependents = st.selectbox(
                    "Dependents",
                    options=["0", "1", "2", "3+"],
                    index=["0", "1", "2", "3+"].index(st.session_state.get("dep", "0")),
                )

            r4_1, r4_2, r4_3 = st.columns(3)
            with r4_1:
                self_employed = st.selectbox(
                    "Self Employed",
                    options=["No", "Yes"],
                    index=["No", "Yes"].index(st.session_state.get("self_emp", "No")),
                )
            with r4_2:
                gender = st.selectbox("Gender (Audit Only)", options=["Male", "Female"], index=0)
            with r4_3:
                married = st.selectbox("Marital Status", options=["Yes", "No"], index=0)

            threshold = st.slider("Decision Cutoff Threshold", 0.10, 0.90, default_thresh, 0.05)

            submitted = st.form_submit_button("⚡ Check Eligibility via XGBoost", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Right: Results Card with SHAP Factors
    with result_col:
        st.markdown(
            """
            <div class="fin-card">
                <div style="font-size: 1.15rem; font-weight: 800; color: #1A1D1F; margin-bottom: 12px;">Evaluation Outcome</div>
            """,
            unsafe_allow_html=True,
        )

        if submitted:
            payload = {
                "ApplicantIncome": float(app_inc),
                "CoapplicantIncome": float(coapp_inc),
                "LoanAmount": float(loan_amt),
                "Loan_Amount_Term": float(loan_term),
                "Credit_History": float(credit_hist),
                "Education": education,
                "Dependents": dependents,
                "Self_Employed": self_employed,
                "Property_Area": property_area,
                "Gender": gender,
                "Married": married,
            }

            res = predict_one(payload, threshold=threshold, log_to_telemetry=True)
            is_approved = (res["decision"] == 1)
            prob = res["probability"]

            badge_color = "approved" if is_approved else "rejected"
            badge_text = "APPROVED" if is_approved else "REJECTED"
            msg = "Credit history and leverage satisfy requirements." if is_approved else "Failed to meet repayment risk threshold."

            st.markdown(
                f"""
                <div style="text-align: center; padding: 12px 0;">
                    <span class="pill-status {badge_color}" style="font-size: 1.1rem; padding: 6px 20px;">
                        {badge_text}
                    </span>
                    <div style="font-size: 0.85rem; color: #6F767E; margin-top: 8px;">{msg}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Circular gauge for approval probability
            fig_p = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%", "font": {"family": "Plus Jakarta Sans", "size": 32, "color": "#1A1D1F"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#9A9FA5"},
                        "bar": {"color": "#2A9D8F" if is_approved else "#EF4444", "thickness": 0.3},
                        "bgcolor": "#E5E7EB",
                        "borderwidth": 0,
                    },
                )
            )
            fig_p.update_layout(height=160, margin=dict(l=20, r=20, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar": False})

            # Meta pills
            st.markdown(
                f"""
                <div style="display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; margin-bottom: 14px;">
                    <span class="pill-status info">⏱️ {res['latency_ms']:.1f} ms</span>
                    <span class="pill-status success">Threshold: {threshold:.2f}</span>
                    <span class="pill-status info">#ID: {res.get('prediction_id', 'N/A')}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # SHAP Feature Attributions (Teal = push approve, Red = push reject)
            factors = res.get("top_factors", [])
            if factors:
                st.markdown("<div style='font-size: 0.88rem; font-weight: 700; margin-bottom: 6px;'>Key Driving Factors (SHAP Values)</div>", unsafe_allow_html=True)
                for f in factors:
                    feat = f["feature"].replace("Property_Area_", "Area: ").replace("Education_", "Edu: ")
                    imp = f["impact"]
                    is_pos = (imp > 0)
                    clr = "#2A9D8F" if is_pos else "#EF4444"
                    txt = "Pushes Approve" if is_pos else "Pushes Reject"

                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; font-size: 0.78rem; margin-top: 6px;">
                            <span style="font-weight: 600; color: #1A1D1F;">{feat}</span>
                            <span style="color: {clr}; font-weight: 700;">{'+' if is_pos else ''}{imp:.3f} ({txt})</span>
                        </div>
                        <div style="width: 100%; height: 6px; background: #E5E7EB; border-radius: 3px; overflow: hidden; margin-top: 2px;">
                            <div style="width: {min(100, int(abs(imp)*180))}%; height: 100%; background: {clr}; border-radius: 3px;"></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Fill out the applicant financial details and click **Check Eligibility** to generate live inference and SHAP attributions.")

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    from app.components import sidebar
    from app.styles import CUSTOM_CSS
    st.set_page_config(layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    sidebar(active_page="Predict")
    render_predict()
