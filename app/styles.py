"""
LOANIQ.ML - Streamlit Design System & Custom CSS
Pixel-perfect matching of the FINbank.IO dashboard:
- Near-black sidebar (#1A1A1A), ~230px
- Active pill (#2A9D8F)
- Very light gray canvas (#F4F5F7)
- White cards with soft shadows, 14-16px radius, no hard borders
- Hidden Streamlit headers/footers
"""

CUSTOM_CSS = """
<style>
/* Import Google Fonts: Plus Jakarta Sans & Inter */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

/* Hide default Streamlit chrome */
#MainMenu { visibility: hidden !important; }
header { visibility: hidden !important; }
footer { visibility: hidden !important; }
[data-testid="stSidebarNav"] { display: none !important; }
div[data-testid="stToolbar"] { display: none !important; }

/* Global App Styling */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp {
    background-color: #F4F5F7 !important;
}

/* Container Padding */
.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1400px !important;
}

/* ==========================================================================
   Near-Black Sidebar (#1A1A1A)
   ========================================================================== */
[data-testid="stSidebar"] {
    background-color: #1A1A1A !important;
    border-right: 1px solid #282828 !important;
    width: 240px !important;
    min-width: 240px !important;
}

[data-testid="stSidebar"] > div:first-child {
    background-color: #1A1A1A !important;
    padding-top: 1.5rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

/* Wordmark LOANIQ.ML */
.brand-wordmark {
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    color: #FFFFFF !important;
    margin-bottom: 2rem;
    padding-left: 0.5rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.brand-wordmark span {
    color: #2A9D8F;
}

/* Sidebar Nav Buttons */
div[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    background-color: transparent !important;
    color: #8A8A8E !important;
    border: none !important;
    border-radius: 10px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 0.65rem 1rem !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
}

div[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #262626 !important;
    color: #FFFFFF !important;
}

/* Active Nav Button Styling */
div[data-testid="stSidebar"] .stButton > button.active-nav-btn {
    background-color: #2A9D8F !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 14px rgba(42, 157, 143, 0.3) !important;
}

/* Sidebar Profile Block */
.sidebar-profile {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 0.5rem;
    margin-top: 1.5rem;
    border-top: 1px solid #282828;
}

.profile-avatar {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    object-fit: cover;
}

.profile-name {
    font-size: 0.88rem;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.2;
}

.profile-role {
    font-size: 0.75rem;
    color: #8A8A8E;
}

/* ==========================================================================
   Top Header Bar
   ========================================================================== */
.top-header-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
}

.header-greeting {
    font-size: 1.6rem;
    font-weight: 800;
    color: #1A1D1F;
    letter-spacing: -0.02em;
    display: flex;
    align-items: baseline;
    gap: 12px;
}

.header-date {
    font-size: 0.9rem;
    font-weight: 500;
    color: #8A8A8E;
}

.header-actions {
    display: flex;
    align-items: center;
    gap: 16px;
}

.bell-badge {
    position: relative;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #1A1D1F;
}

.bell-dot {
    position: absolute;
    top: 6px;
    right: 8px;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: #2A9D8F;
}

.search-box-pill {
    background: #FFFFFF;
    border: 1px solid #ECEFF2;
    border-radius: 9999px;
    padding: 8px 16px 8px 16px;
    display: flex;
    align-items: center;
    gap: 8px;
    width: 240px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}

.search-box-pill input {
    border: none;
    outline: none;
    font-size: 0.85rem;
    color: #1A1D1F;
    width: 100%;
}

.search-box-pill input::placeholder {
    color: #9A9FA5;
}

/* ==========================================================================
   Card Styling: White, 14-16px Radius, Soft Shadow (0 2px 10px rgba(0,0,0,0.05))
   ========================================================================== */
.fin-card {
    background-color: #FFFFFF !important;
    border-radius: 16px !important;
    padding: 20px !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04) !important;
    border: 1px solid #ECEFF2 !important;
    margin-bottom: 1.25rem !important;
    height: 100% !important;
}

.card-title-muted {
    font-size: 0.95rem;
    font-weight: 600;
    color: #8A8A8E;
    margin-bottom: 0.5rem;
}

.card-val-big {
    font-size: 1.85rem;
    font-weight: 800;
    color: #1A1D1F;
    letter-spacing: -0.02em;
    line-height: 1.1;
}

/* Teal Gradient Model Card (Credit Card equivalent) */
.model-credit-card {
    background: linear-gradient(135deg, #247B6D 0%, #1A5F54 100%);
    border-radius: 14px;
    padding: 16px 18px;
    color: #FFFFFF;
    margin-top: 14px;
    margin-bottom: 12px;
    box-shadow: 0 6px 18px rgba(36, 123, 109, 0.25);
    position: relative;
    overflow: hidden;
}

.model-credit-card::before {
    content: '';
    position: absolute;
    top: -40px;
    right: -40px;
    width: 130px;
    height: 130px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 70%);
}

.mcc-type-lbl {
    font-size: 0.7rem;
    color: rgba(255,255,255,0.75);
    font-weight: 500;
}

.mcc-type-val {
    font-size: 0.95rem;
    font-weight: 700;
    color: #FFFFFF;
}

.mcc-meta-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    margin-top: 18px;
}

.mcc-version {
    font-size: 0.88rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    font-family: monospace;
}

.mcc-f1-val {
    font-size: 1.15rem;
    font-weight: 800;
}

/* Icon Tiles (Light Gray Squares beside rows) */
.icon-tile-square {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    background-color: #F4F5F7;
    border: 1px solid #ECEFF2;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #6F767E;
    flex-shrink: 0;
}

/* Status Pills */
.pill-status {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
}

.pill-status.approved, .pill-status.success {
    background-color: #EBF8F4;
    color: #2A9D8F;
}

.pill-status.rejected, .pill-status.danger {
    background-color: #FEE2E2;
    color: #EF4444;
}

.pill-status.warning {
    background-color: #FEF3C7;
    color: #D97706;
}

.pill-status.info {
    background-color: #EFF6FF;
    color: #2563EB;
}

/* Telemetry Tile (3x2 breakdown card) */
.telemetry-tile {
    background-color: #FFFFFF;
    border: 1px solid #ECEFF2;
    border-radius: 12px;
    padding: 14px 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: all 0.15s ease;
}

.telemetry-tile:hover {
    box-shadow: 0 4px 14px rgba(0,0,0,0.04);
    border-color: #D1D5DB;
}

.tt-label {
    font-size: 0.75rem;
    color: #8A8A8E;
    font-weight: 600;
}

.tt-value {
    font-size: 1.1rem;
    font-weight: 800;
    color: #1A1D1F;
    line-height: 1.2;
}

.tt-delta {
    font-size: 0.72rem;
    font-weight: 600;
}

.tt-delta.pos {
    color: #10B981;
}

.tt-delta.neg {
    color: #EF4444;
}

/* Check eligibility button */
.btn-teal-action {
    background-color: #2A9D8F !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(42, 157, 143, 0.25) !important;
}

.btn-teal-action:hover {
    background-color: #228766 !important;
}
</style>
"""
