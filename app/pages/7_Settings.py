"""
LOANIQ.ML - Settings Page
System configuration, operational thresholds, telemetry traffic generation,
and environment parameters matching FINbank.IO design tokens.
"""

from pathlib import Path

import streamlit as st

from app.components import sidebar, top_header
from src.predict import load_model_and_meta
from src.telemetry import summary_stats


def render_settings():
    """Render the Settings and System Configuration page."""
    top_header(greeting="System Settings & Telemetry Controls", date_str="Operational Configuration")

    # Load Model Metadata
    try:
        _, meta = load_model_and_meta()
        model_name = meta.get("model_name", "XGBoost")
        version = meta.get("version", "v1.0.0")
        default_thresh = float(meta.get("default_threshold", 0.50))
    except Exception:
        model_name = "XGBoost"
        version = "v1.0.0"
        default_thresh = 0.50

    if "global_threshold" not in st.session_state:
        st.session_state.global_threshold = default_thresh

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        # Card 1: Decision Threshold & Policy
        st.markdown(
            """
            <div class="fin-card">
                <div style="font-size: 1.1rem; font-weight: 700; color: #1A1D1F; margin-bottom: 4px;">Inference Decision Policy</div>
                <div style="font-size: 0.8rem; color: #8A8A8E; margin-bottom: 16px;">Configure real-time approval cutoff threshold and risk tolerance.</div>
            """,
            unsafe_allow_html=True,
        )

        new_thresh = st.slider(
            "Global Decision Cutoff Threshold",
            min_value=0.20,
            max_value=0.80,
            value=float(st.session_state.global_threshold),
            step=0.01,
            help="Applications with probability >= threshold receive an Approved decision.",
        )
        if new_thresh != st.session_state.global_threshold:
            st.session_state.global_threshold = new_thresh
            st.toast(f"Threshold updated to {new_thresh:.2f}")

        st.markdown(
            f"""
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px 16px; margin-top: 14px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.85rem; color: #64748B;">Active Policy:</span>
                    <span class="pill-status info">Cutoff = {st.session_state.global_threshold:.2f}</span>
                </div>
                <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 6px;">
                    Lower thresholds increase recall for borrowers, while higher thresholds prioritize lower default exposure.
                </div>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Card 2: Synthetic Telemetry Generator
        st.markdown(
            """
            <div class="fin-card">
                <div style="font-size: 1.1rem; font-weight: 700; color: #1A1D1F; margin-bottom: 4px;">Telemetry Traffic Simulator</div>
                <div style="font-size: 0.8rem; color: #8A8A8E; margin-bottom: 16px;">Stream realistic synthetic loan applications directly into the SQLite telemetry log.</div>
            """,
            unsafe_allow_html=True,
        )

        sim_count = st.selectbox("Batch Traffic Volume", options=[10, 25, 50, 100], index=1)
        if st.button(f"⚡ Generate {sim_count} Real-Time Applications", use_container_width=True):
            with st.spinner(f"Simulating {sim_count} application evaluations..."):
                try:
                    from scripts.generate_traffic import main as gen_traffic
                    gen_traffic(n=sim_count)
                    st.success(f"Successfully evaluated and logged {sim_count} applications to telemetry.db!")
                    st.rerun()
                except Exception:
                    # Fallback simulation
                    import random

                    from src.telemetry import log_prediction
                    for i in range(sim_count):
                        p = random.uniform(0.1, 0.95)
                        dec = 1 if p >= st.session_state.global_threshold else 0
                        log_prediction(
                            inputs={"ApplicantIncome": random.randint(2000, 10000), "LoanAmount": random.randint(80, 400), "Credit_History": random.choice([0.0, 1.0])},
                            probability=p,
                            decision=dec,
                            threshold=st.session_state.global_threshold,
                            latency_ms=random.uniform(18.0, 45.0),
                            top_factors={"Credit_History": 0.45, "ApplicantIncome": 0.12},
                        )
                    st.success(f"Logged {sim_count} simulated applications to telemetry DB.")
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        # Card 3: Storage & Deployment Environment
        stats = summary_stats()
        db_path = Path("logs/telemetry.db")
        db_size_kb = db_path.stat().st_size / 1024 if db_path.exists() else 0.0

        st.markdown(
            f"""
            <div class="fin-card">
                <div style="font-size: 1.1rem; font-weight: 700; color: #1A1D1F; margin-bottom: 4px;">System Architecture Registry</div>
                <div style="font-size: 0.8rem; color: #8A8A8E; margin-bottom: 16px;">Connected MLflow artifacts and local SQLite telemetry engines.</div>
                
                <div style="display: flex; flex-direction: column; gap: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #F1F3F5; padding-bottom: 8px;">
                        <span style="font-size: 0.82rem; color: #6F767E;">Champion Model</span>
                        <span style="font-size: 0.82rem; font-weight: 700; color: #1A1D1F;">{model_name} ({version})</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #F1F3F5; padding-bottom: 8px;">
                        <span style="font-size: 0.82rem; color: #6F767E;">Telemetry Database</span>
                        <span style="font-size: 0.82rem; font-weight: 700; color: #1A1D1F;">SQLite (logs/telemetry.db)</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #F1F3F5; padding-bottom: 8px;">
                        <span style="font-size: 0.82rem; color: #6F767E;">DB File Size</span>
                        <span style="font-size: 0.82rem; font-weight: 700; color: #1A1D1F;">{db_size_kb:.1f} KB</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #F1F3F5; padding-bottom: 8px;">
                        <span style="font-size: 0.82rem; color: #6F767E;">Total Logged Records</span>
                        <span class="pill-status success">{stats.get('total_predictions', 0)} calls</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #F1F3F5; padding-bottom: 8px;">
                        <span style="font-size: 0.82rem; color: #6F767E;">MLflow Tracking URI</span>
                        <span style="font-size: 0.78rem; font-family: monospace; color: #2A9D8F;">file:./mlruns</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 0.82rem; color: #6F767E;">FastAPI Gateway</span>
                        <span class="pill-status info">http://127.0.0.1:8000</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Card 4: About & Version
        st.markdown(
            """
            <div class="fin-card">
                <div style="font-size: 1.1rem; font-weight: 700; color: #1A1D1F; margin-bottom: 4px;">About LOANIQ.ML</div>
                <div style="font-size: 0.8rem; color: #8A8A8E; margin-bottom: 12px;">Full-Stack Machine Learning Governance &amp; Telemetry</div>
                <div style="font-size: 0.8rem; color: #6F767E; line-height: 1.5;">
                    Built with Streamlit, Plotly, Scikit-Learn, XGBoost, and Fairlearn.
                    Redesigned to adhere to the FINbank.IO design tokens (soft shadows, 16px radius, dark sidebar, teal accents).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    from app.styles import CUSTOM_CSS
    st.set_page_config(layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    sidebar(active_page="Settings")
    render_settings()
