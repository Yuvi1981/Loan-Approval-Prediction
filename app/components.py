"""
LOANIQ.ML - Reusable UI Components
Provides styled modular components matching the FINbank.IO dashboard:
- sidebar()
- top_header()
- metric_card()
- list_row()
- status_pill()
- gauge_chart()
- grouped_bar_chart()
- model_credit_card()
- telemetry_tile()
"""

import base64
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import plotly.graph_objects as go
import streamlit as st


def get_avatar_base64() -> str:
    """Read avatar image and convert to base64 for inline rendering."""
    possible_paths = [
        Path("frontend/tanzir_avatar.jpg"),
        Path(__file__).resolve().parent.parent / "frontend" / "tanzir_avatar.jpg",
    ]
    for p in possible_paths:
        if p.exists():
            try:
                with open(p, "rb") as f:
                    return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"
            except Exception:
                pass
    return "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=100&auto=format&fit=crop&q=80"


def sidebar(active_page: str = "Overview") -> str:
    """Render the sleek near-black sidebar matching the reference UI."""
    avatar_src = get_avatar_base64()

    with st.sidebar:
        # Wordmark
        st.markdown(
            """
            <div class="brand-wordmark">
                LOANIQ<span>.ML</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Nav Buttons Definition
        nav_items = [
            ("Overview", "⊞  Overview"),
            ("Predict", "⚡  Predict"),
            ("Experiments", "📊  Experiments"),
            ("Data Quality", "🔍  Data Quality"),
            ("Drift", "🌊  Drift"),
            ("Fairness", "⚖️  Fairness"),
            ("Settings", "⚙️  Settings"),
        ]

        for page_key, label in nav_items:
            is_active = (active_page == page_key)
            btn_key = f"nav_btn_{page_key}"
            
            # Use Streamlit button; CSS styles the active one with teal pill
            if is_active:
                st.markdown(
                    f"""
                    <style>
                    div.row-widget.stButton button[key="{btn_key}"] {{
                        background-color: #2A9D8F !important;
                        color: #FFFFFF !important;
                        box-shadow: 0 4px 14px rgba(42, 157, 143, 0.3) !important;
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

            if st.button(label, key=btn_key, use_container_width=True):
                st.session_state.current_page = page_key
                st.rerun()

        # Spacer to push profile to bottom
        st.markdown("<div style='height: 120px;'></div>", unsafe_allow_html=True)

        # Logout button
        if st.button("➔  Logout", key="btn_logout", use_container_width=True):
            st.toast("Signed out successfully.")
            st.session_state.current_page = "Overview"
            st.rerun()

        # User Profile Block
        st.markdown(
            f"""
            <div class="sidebar-profile">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <img src="{avatar_src}" class="profile-avatar" alt="Avatar">
                    <div>
                        <div class="profile-name">Tanzir Rahman</div>
                        <div class="profile-role">View profile</div>
                    </div>
                </div>
                <div style="color: #8A8A8E; cursor: pointer; padding: 4px; font-size: 1.1rem;">⋮</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return st.session_state.get("current_page", "Overview")


def top_header(greeting: str = "Hello Tanzir", date_str: Optional[str] = None) -> None:
    """Render top bar matching the reference: greeting + date, notification bell, search pill."""
    if date_str is None:
        date_str = datetime.now().strftime("%b %d, %Y")

    st.markdown(
        f"""
        <div class="top-header-bar">
            <div class="header-greeting">
                <span>{greeting}</span>
                <span class="header-date">&gt;&gt; {date_str}</span>
            </div>
            <div class="header-actions">
                <div class="bell-badge" title="Notifications">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#1A1D1F" stroke-width="2">
                        <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/>
                        <path d="M13.73 21a2 2 0 0 1-3.46 0"/>
                    </svg>
                    <div class="bell-dot"></div>
                </div>
                <div class="search-box-pill">
                    <input type="text" placeholder="Search here">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#9A9FA5" stroke-width="2">
                        <circle cx="11" cy="11" r="8"/>
                        <line x1="21" y1="21" x2="16.65" y2="16.65"/>
                    </svg>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(
    title: str,
    value: str,
    subtitle: Optional[str] = None,
    delta: Optional[str] = None,
    is_positive: bool = True,
    extra_html: Optional[str] = None,
) -> None:
    """Render a standard white card with title, bold value, and optional delta/subtitle."""
    delta_html = ""
    if delta:
        arrow = "↑" if is_positive else "↓"
        cls = "pos" if is_positive else "neg"
        delta_html = f'<div class="tt-delta {cls}" style="margin-top: 4px;">{delta} {arrow}</div>'
    sub_html = f'<div style="font-size:0.75rem; color:#8A8A8E; margin-top:4px;">{subtitle}</div>' if subtitle else ""
    extra = extra_html if extra_html else ""

    st.markdown(
        f"""
        <div class="fin-card">
            <div class="card-title-muted">{title}</div>
            <div class="card-val-big">{value}</div>
            {delta_html}
            {sub_html}
            {extra}
        </div>
        """,
        unsafe_allow_html=True,
    )


def list_row(
    title: str,
    subtitle: str,
    amount_or_value: str,
    status_text: str,
    status_type: str = "approved",
    icon_svg: Optional[str] = None,
    probability_str: Optional[str] = None,
) -> None:
    """Render a single prediction or transaction list row with square icon, pill, and amounts."""
    if not icon_svg:
        icon_svg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#6F767E" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>'

    prob_html = f'<div style="font-size: 0.75rem; color: #8A8A8E; text-align: right;">{probability_str}</div>' if probability_str else ""
    pill_html = status_pill(status_text, status_type)

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #F4F5F7;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div class="icon-tile-square">
                    {icon_svg}
                </div>
                <div>
                    <div style="font-size: 0.88rem; font-weight: 700; color: #1A1D1F;">{title}</div>
                    <div style="font-size: 0.75rem; color: #8A8A8E;">{subtitle}</div>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 16px;">
                <div style="text-align: right;">
                    <div style="font-size: 0.92rem; font-weight: 700; color: #1A1D1F;">{amount_or_value}</div>
                    {prob_html}
                </div>
                <div>
                    {pill_html}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_pill(text: str, status: str = "success") -> str:
    """Generate HTML markup for a rounded status pill."""
    return f'<span class="pill-status {status}">{text}</span>'


def gauge_chart(value: float, min_val: float = 0.0, max_val: float = 1.0, title: str = "Target vs Achievement", target_val: float = 0.90) -> go.Figure:
    """Generate a clean Plotly semicircle gauge chart in teal #2A9D8F."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"valueformat": ".2f", "font": {"family": "Plus Jakarta Sans", "size": 24, "color": "#1A1D1F"}},
            gauge={
                "axis": {"range": [min_val, max_val], "tickwidth": 1, "tickcolor": "#9A9FA5", "tickfont": {"size": 10, "color": "#8A8A8E"}},
                "bar": {"color": "#2A9D8F", "thickness": 0.28},
                "bgcolor": "#E5E7EB",
                "borderwidth": 0,
                "threshold": {
                    "line": {"color": "#1A1D1F", "width": 3},
                    "thickness": 0.8,
                    "value": target_val,
                },
            },
        )
    )
    fig.update_layout(
        height=130,
        margin=dict(l=15, r=15, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans"),
    )
    return fig


def grouped_bar_chart(
    days: List[str],
    this_week: List[float],
    last_week: List[float],
    y_format: str = "{val}",
) -> go.Figure:
    """Generate the Plotly dual grouped bar chart for Weekly Comparison."""
    fig = go.Figure()
    
    # Last week (Light Gray)
    fig.add_trace(
        go.Bar(
            x=days,
            y=last_week,
            name="Last week",
            marker_color="#E5E7EB",
            width=0.32,
        )
    )

    # This week (Teal)
    fig.add_trace(
        go.Bar(
            x=days,
            y=this_week,
            name="This week",
            marker_color="#2A9D8F",
            width=0.32,
        )
    )

    fig.update_layout(
        barmode="group",
        height=180,
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#6F767E"),
        ),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=11, color="#8A8A8E", family="Inter"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#F1F3F5",
            tickfont=dict(size=10, color="#8A8A8E", family="Inter"),
        ),
    )
    return fig


def telemetry_tile(
    icon_svg: str,
    label: str,
    value: str,
    delta_text: str,
    is_positive: bool,
    target_page: str,
) -> None:
    """Render a single breakdown tile in the 3x2 grid."""
    arrow = "↑" if "↑" in delta_text or "+" in delta_text or is_positive else "↓"
    cls = "pos" if is_positive else "neg"

    st.markdown(
        f"""
        <div class="telemetry-tile">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div class="icon-tile-square">
                    {icon_svg}
                </div>
                <div>
                    <div class="tt-label">{label}</div>
                    <div class="tt-value">{value}</div>
                    <div class="tt-delta {cls}">{delta_text} {arrow}</div>
                </div>
            </div>
            <div style="color: #9A9FA5; font-size: 1.1rem; cursor: pointer;" title="View details">→</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
