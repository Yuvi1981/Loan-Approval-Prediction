"""
LOANIQ.ML - Experiments Page
MLflow experiment runs table styled as cards + candidate metric comparison bar chart.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components import status_pill, top_header

LEADERBOARD_DATA = [
    {"Model": "XGBoost", "Type": "Gradient Boosting", "CV F1": 0.8524, "Accuracy": 0.8618, "Precision": 0.8469, "Recall": 0.9765, "F1-Score": 0.9071, "ROC-AUC": 0.8359, "IsChamp": True},
    {"Model": "StackingClassifier", "Type": "Stacking Ensemble", "CV F1": 0.8622, "Accuracy": 0.8618, "Precision": 0.8542, "Recall": 0.9647, "F1-Score": 0.9061, "ROC-AUC": 0.8647, "IsChamp": False},
    {"Model": "DecisionTree", "Type": "Baseline Tree", "CV F1": 0.8582, "Accuracy": 0.8618, "Precision": 0.8617, "Recall": 0.9529, "F1-Score": 0.9050, "ROC-AUC": 0.7913, "IsChamp": False},
    {"Model": "LogisticRegression", "Type": "Linear Baseline", "CV F1": 0.8626, "Accuracy": 0.8537, "Precision": 0.8317, "Recall": 0.9882, "F1-Score": 0.9032, "ROC-AUC": 0.8443, "IsChamp": False},
    {"Model": "GaussianNaiveBayes", "Type": "Probabilistic Baseline", "CV F1": 0.8378, "Accuracy": 0.8537, "Precision": 0.8317, "Recall": 0.9882, "F1-Score": 0.9032, "ROC-AUC": 0.8437, "IsChamp": False},
    {"Model": "VotingClassifier_Soft", "Type": "Soft Ensemble", "CV F1": 0.8626, "Accuracy": 0.8537, "Precision": 0.8317, "Recall": 0.9882, "F1-Score": 0.9032, "ROC-AUC": 0.8734, "IsChamp": False},
    {"Model": "RandomForest_ClassWeighted", "Type": "Imbalance (Balanced)", "CV F1": 0.8486, "Accuracy": 0.8537, "Precision": 0.8681, "Recall": 0.9294, "F1-Score": 0.8977, "ROC-AUC": 0.8709, "IsChamp": False},
    {"Model": "RandomForest", "Type": "Bagging Ensemble", "CV F1": 0.8672, "Accuracy": 0.8455, "Precision": 0.8367, "Recall": 0.9647, "F1-Score": 0.8962, "ROC-AUC": 0.8789, "IsChamp": False},
    {"Model": "XGBoost_ClassWeighted", "Type": "Imbalance (Pos Weight)", "CV F1": 0.8273, "Accuracy": 0.8537, "Precision": 0.8851, "Recall": 0.9059, "F1-Score": 0.8953, "ROC-AUC": 0.8601, "IsChamp": False},
    {"Model": "RandomForest_Tuned", "Type": "Tuned GridSearchCV", "CV F1": 0.8731, "Accuracy": 0.8293, "Precision": 0.8265, "Recall": 0.9529, "F1-Score": 0.8852, "ROC-AUC": 0.8799, "IsChamp": False},
    {"Model": "KNN", "Type": "Distance Baseline", "CV F1": 0.8497, "Accuracy": 0.8293, "Precision": 0.8478, "Recall": 0.9176, "F1-Score": 0.8814, "ROC-AUC": 0.8209, "IsChamp": False},
    {"Model": "RandomForest_SMOTE", "Type": "Imbalance (SMOTE)", "CV F1": 0.8441, "Accuracy": 0.7967, "Precision": 0.8409, "Recall": 0.8706, "F1-Score": 0.8555, "ROC-AUC": 0.8396, "IsChamp": False},
]


def render_experiments():
    """Render the Experiments & Model Benchmark Leaderboard page."""
    top_header(greeting="MLflow Model Experiments", date_str="12 Tracked Benchmark Runs")

    st.markdown(
        """
        <div class="fin-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div style="font-size: 1.15rem; font-weight: 800; color: #1A1D1F;">Candidate Models Comparison: Test F1-Score &amp; ROC-AUC</div>
                <span class="pill-status success">Champion: XGBoost (86.2% Acc, 0.907 F1)</span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    models_names = [d["Model"] for d in LEADERBOARD_DATA]
    f1_vals = [d["F1-Score"] for d in LEADERBOARD_DATA]
    auc_vals = [d["ROC-AUC"] for d in LEADERBOARD_DATA]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=models_names, y=f1_vals, name="Test F1-Score", marker_color="#2A9D8F"))
    fig.add_trace(go.Bar(x=models_names, y=auc_vals, name="Test ROC-AUC", marker_color="#9CA3AF"))
    fig.update_layout(
        barmode="group",
        height=260,
        margin=dict(l=20, r=20, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
        yaxis=dict(range=[0.7, 1.0], gridcolor="#F1F3F5"),
        xaxis=dict(tickangle=-30),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # Leaderboard Cards Table
    st.markdown(
        """
        <div class="fin-card">
            <div style="font-size: 1.15rem; font-weight: 800; color: #1A1D1F; margin-bottom: 14px;">Evaluated Models Performance Breakdown</div>
        """,
        unsafe_allow_html=True,
    )

    df_display = pd.DataFrame(LEADERBOARD_DATA)
    for idx, r in df_display.iterrows():
        is_champ = r["IsChamp"]
        champ_pill = status_pill("CHAMPION", "approved") if is_champ else ""
        border_accent = "border-left: 4px solid #2A9D8F;" if is_champ else "border-left: 4px solid #ECEFF2;"
        bg_color = "#F8FDFB" if is_champ else "#FFFFFF"

        st.markdown(
            f"""
            <div style="background: {bg_color}; border: 1px solid #ECEFF2; {border_accent} border-radius: 10px; padding: 12px 18px; margin-bottom: 8px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <div style="min-width: 200px;">
                    <div style="font-weight: 800; font-size: 0.95rem; color: #1A1D1F;">#{idx+1} {r['Model']} {champ_pill}</div>
                    <div style="font-size: 0.78rem; color: #8A8A8E;">{r['Type']}</div>
                </div>
                <div style="display: flex; gap: 24px; font-size: 0.85rem;">
                    <div><span style="color:#8A8A8E;">CV F1:</span> <b>{r['CV F1']:.4f}</b></div>
                    <div><span style="color:#8A8A8E;">Accuracy:</span> <b>{r['Accuracy']:.1%}</b></div>
                    <div><span style="color:#8A8A8E;">Precision:</span> <b>{r['Precision']:.1%}</b></div>
                    <div><span style="color:#8A8A8E;">Recall:</span> <b>{r['Recall']:.1%}</b></div>
                    <div><span style="color:#8A8A8E;">F1-Score:</span> <b style="color:#2A9D8F;">{r['F1-Score']:.4f}</b></div>
                    <div><span style="color:#8A8A8E;">ROC-AUC:</span> <b>{r['ROC-AUC']:.4f}</b></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    from app.components import sidebar
    from app.styles import CUSTOM_CSS
    st.set_page_config(layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    sidebar(active_page="Experiments")
    render_experiments()
