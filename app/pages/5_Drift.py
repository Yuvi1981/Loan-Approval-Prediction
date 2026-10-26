"""
LOANIQ.ML - Data Drift Surveillance Page
Monitors PSI and KS-test shifts between baseline training data and live applications
using FINbank.IO design tokens (white cards, teal accent, status banners, Plotly charts).
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components import sidebar, top_header


def render_drift():
    """Render the Drift Surveillance page."""
    top_header(greeting="Data Drift Surveillance", date_str="Population Stability & Feature Distribution Shifts")

    drift_report_path = Path("reports/drift_demo.json")
    if drift_report_path.exists():
        with open(drift_report_path, "r", encoding="utf-8") as f:
            drift_data = json.load(f)
    else:
        # Fallback demo data
        drift_data = {
            "drift_summary": {
                "features_evaluated": 11,
                "features_with_drift": 3,
                "max_psi": 0.482,
                "top_drifted_feature": "ApplicantIncome",
            },
            "performance_impact": {
                "baseline": {"accuracy": 0.8618, "f1": 0.9071, "roc_auc": 0.8359, "precision": 0.8469, "recall": 0.9765},
                "drifted": {"accuracy": 0.7398, "f1": 0.8095, "roc_auc": 0.6982, "precision": 0.7727, "recall": 0.8500},
                "deltas": {"accuracy_drop": -0.122, "f1_drop": -0.0976, "roc_auc_drop": -0.1377},
            },
            "feature_drift_details": [
                {"feature": "ApplicantIncome", "psi": 0.4821, "drift_detected": True, "ks_statistic": 0.284, "ks_p_value": 0.0001},
                {"feature": "LoanAmount", "psi": 0.3124, "drift_detected": True, "ks_statistic": 0.215, "ks_p_value": 0.0012},
                {"feature": "Credit_History", "psi": 0.2745, "drift_detected": True, "ks_statistic": None, "ks_p_value": None},
                {"feature": "CoapplicantIncome", "psi": 0.0812, "drift_detected": False, "ks_statistic": 0.065, "ks_p_value": 0.3421},
                {"feature": "Property_Area", "psi": 0.0431, "drift_detected": False, "ks_statistic": None, "ks_p_value": None},
                {"feature": "Loan_Amount_Term", "psi": 0.0315, "drift_detected": False, "ks_statistic": 0.041, "ks_p_value": 0.6120},
            ],
        }

    summary = drift_data.get("drift_summary", {})
    perf = drift_data.get("performance_impact", {})
    features_drift = drift_data.get("feature_drift_details", [])

    # Macro Status Banner
    has_alert = summary.get("features_with_drift", 0) > 0
    if has_alert:
        st.markdown(
            f"""
            <div style="background-color: #FEF3C7; border: 1px solid #FCD34D; border-radius: 12px; padding: 14px 18px; display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="font-size: 1.4rem;">⚠️</div>
                    <div>
                        <div style="font-size: 0.92rem; font-weight: 700; color: #92400E;">Statistical Drift Alert Detected</div>
                        <div style="font-size: 0.8rem; color: #B45309;">
                            {summary.get('features_with_drift', 0)} of {summary.get('features_evaluated', 11)} features exhibit high distribution shift (PSI &gt; 0.25). Top shift driver: <b>{summary.get('top_drifted_feature', 'N/A')}</b>.
                        </div>
                    </div>
                </div>
                <span class="pill-status warning">Needs Review</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background-color: #EBF8F4; border: 1px solid #A7F3D0; border-radius: 12px; padding: 14px 18px; display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="font-size: 1.4rem;">✅</div>
                    <div>
                        <div style="font-size: 0.92rem; font-weight: 700; color: #065F46;">Distribution Stability Confirmed</div>
                        <div style="font-size: 0.8rem; color: #047857;">All feature PSI scores remain well below the 0.10 threshold. Model reliability is stable.</div>
                    </div>
                </div>
                <span class="pill-status success">Healthy</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Row 1: KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div class="fin-card" style="text-align: center;">
                <div class="card-title-muted">Features Evaluated</div>
                <div class="card-val-big">{summary.get('features_evaluated', 11)}</div>
                <div style="font-size:0.75rem; color:#8A8A8E; margin-top:4px;">Continuous &amp; Categorical</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        drift_cnt = summary.get("features_with_drift", 0)
        color_val = "#EF4444" if drift_cnt > 0 else "#2A9D8F"
        st.markdown(
            f"""
            <div class="fin-card" style="text-align: center;">
                <div class="card-title-muted">Drifted Features</div>
                <div class="card-val-big" style="color: {color_val};">{drift_cnt}</div>
                <div style="font-size:0.75rem; color:#8A8A8E; margin-top:4px;">PSI &gt; 0.25 Critical</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        max_psi = summary.get("max_psi", 0.0)
        st.markdown(
            f"""
            <div class="fin-card" style="text-align: center;">
                <div class="card-title-muted">Max Feature PSI</div>
                <div class="card-val-big">{max_psi:.3f}</div>
                <div style="font-size:0.75rem; color:#EF4444; margin-top:4px;">Threshold: 0.25</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="fin-card" style="text-align: center;">
                <div class="card-title-muted">Top Drift Source</div>
                <div class="card-val-big" style="font-size: 1.35rem; margin-top: 4px;">{summary.get('top_drifted_feature', 'None')}</div>
                <div style="font-size:0.75rem; color:#8A8A8E; margin-top:4px;">Wage Inflation &amp; Rate Shock</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Row 2: Charts
    c_left, c_right = st.columns([1.2, 1])

    with c_left:
        st.markdown(
            """
            <div class="fin-card">
                <div style="font-size: 1.05rem; font-weight: 700; color: #1A1D1F; margin-bottom: 4px;">Population Stability Index (PSI) by Feature</div>
                <div style="font-size: 0.8rem; color: #8A8A8E; margin-bottom: 12px;">Green &lt; 0.10 (Stable) | Yellow 0.10–0.25 (Moderate) | Red &gt; 0.25 (Action Required)</div>
            """,
            unsafe_allow_html=True,
        )

        df_psi = pd.DataFrame(features_drift)
        if not df_psi.empty and "feature" in df_psi.columns and "psi" in df_psi.columns:
            df_psi = df_psi.sort_values(by="psi", ascending=True)
            colors = []
            for v in df_psi["psi"]:
                if v >= 0.25:
                    colors.append("#EF4444")
                elif v >= 0.10:
                    colors.append("#F59E0B")
                else:
                    colors.append("#2A9D8F")

            fig_psi = go.Figure()
            fig_psi.add_trace(
                go.Bar(
                    y=df_psi["feature"],
                    x=df_psi["psi"],
                    orientation="h",
                    marker_color=colors,
                    width=0.45,
                )
            )
            # Add threshold reference lines
            fig_psi.add_vline(x=0.10, line_dash="dash", line_color="#F59E0B", annotation_text="Warn (0.10)", annotation_position="top right")
            fig_psi.add_vline(x=0.25, line_dash="dash", line_color="#EF4444", annotation_text="Alert (0.25)", annotation_position="top right")
            fig_psi.update_layout(
                height=260,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="#F1F3F5", range=[0, max(0.6, df_psi["psi"].max() + 0.1)]),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_psi, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_right:
        st.markdown(
            """
            <div class="fin-card">
                <div style="font-size: 1.05rem; font-weight: 700; color: #1A1D1F; margin-bottom: 4px;">Macro Model Metric Degradation Under Shock</div>
                <div style="font-size: 0.8rem; color: #8A8A8E; margin-bottom: 12px;">Baseline Reference vs Live Shifted Stream</div>
            """,
            unsafe_allow_html=True,
        )

        categories = ["Accuracy", "F1-Score", "ROC-AUC", "Recall"]
        base_vals = [
            perf.get("baseline", {}).get("accuracy", 0.86),
            perf.get("baseline", {}).get("f1", 0.91),
            perf.get("baseline", {}).get("roc_auc", 0.84),
            perf.get("baseline", {}).get("recall", 0.98),
        ]
        drift_vals = [
            perf.get("drifted", {}).get("accuracy", 0.74),
            perf.get("drifted", {}).get("f1", 0.81),
            perf.get("drifted", {}).get("roc_auc", 0.70),
            perf.get("drifted", {}).get("recall", 0.85),
        ]

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(name="Baseline Reference", x=categories, y=base_vals, marker_color="#2A9D8F", width=0.32))
        fig_comp.add_trace(go.Bar(name="Under Drift Shock", x=categories, y=drift_vals, marker_color="#E5E7EB", width=0.32))
        fig_comp.update_layout(
            barmode="group",
            height=260,
            margin=dict(l=20, r=20, t=10, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
            yaxis=dict(range=[0, 1.1], gridcolor="#F1F3F5"),
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 3: Feature Distribution Details
    st.markdown(
        """
        <div class="fin-card">
            <div style="font-size: 1.05rem; font-weight: 700; color: #1A1D1F; margin-bottom: 12px;">Detailed Drift Audit Registry</div>
        """,
        unsafe_allow_html=True,
    )
    if features_drift:
        df_display = pd.DataFrame(features_drift)
        df_display["Status"] = df_display["drift_detected"].apply(
            lambda x: "🚨 DRIFT DETECTED" if x else "✅ STABLE"
        )
        st.dataframe(
            df_display[["feature", "psi", "ks_statistic", "ks_p_value", "Status"]].rename(
                columns={
                    "feature": "Feature",
                    "psi": "PSI Score",
                    "ks_statistic": "KS Statistic",
                    "ks_p_value": "KS p-value",
                }
            ),
            use_container_width=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    from app.styles import CUSTOM_CSS
    st.set_page_config(layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    sidebar(active_page="Drift")
    render_drift()
