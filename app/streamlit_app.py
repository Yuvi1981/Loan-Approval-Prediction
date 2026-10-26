"""
LOANIQ.ML - Streamlit Application Entrypoint
Redesigned dashboard matching FINbank.IO UI tokens with modular routing across:
- 1_Overview
- 2_Predict
- 3_Experiments
- 4_Data_Quality
- 5_Drift
- 6_Fairness
- 7_Settings
"""

import importlib
import sys
from pathlib import Path

# Add project root to sys.path so 'src' and 'app' are always importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components import sidebar
from app.styles import CUSTOM_CSS

# Configure Streamlit page
st.set_page_config(
    page_title="LOANIQ.ML - Loan Approval & ML Telemetry",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Custom Design Tokens (FINbank.IO Design System)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize Session State
if "current_page" not in st.session_state:
    st.session_state.current_page = "Overview"

# Render Sidebar (LOANIQ.ML near-black brand sidebar)
active_page = sidebar(active_page=st.session_state.current_page)

# Page Registry Mapping
PAGE_MAPPING = {
    "Overview": ("app.pages.1_Overview", "render_overview"),
    "Predict": ("app.pages.2_Predict", "render_predict"),
    "Experiments": ("app.pages.3_Experiments", "render_experiments"),
    "Data Quality": ("app.pages.4_Data_Quality", "render_data_quality"),
    "Drift": ("app.pages.5_Drift", "render_drift"),
    "Fairness": ("app.pages.6_Fairness", "render_fairness"),
    "Settings": ("app.pages.7_Settings", "render_settings"),
}

# Dispatch Page
target_module_name, target_func_name = PAGE_MAPPING.get(active_page, ("app.pages.1_Overview", "render_overview"))

try:
    page_module = importlib.import_module(target_module_name)
    render_func = getattr(page_module, target_func_name)
    render_func()
except Exception as e:
    st.error(f"Error loading page '{active_page}': {e}")
    # Fallback to Overview
    fallback_mod = importlib.import_module("app.pages.1_Overview")
    fallback_mod.render_overview()
