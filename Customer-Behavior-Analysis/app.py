"""
Smart Customer Behavior Analysis & Churn Prediction System
Main entry point – app.py
"""

import streamlit as st
import os
import sys

# ── Path setup ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Smart Customer Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide Streamlit's auto-generated multi-page sidebar nav
st.markdown("""
<style>
[data-testid="stSidebarNav"] { display: none !important; }
</style>""", unsafe_allow_html=True)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── Dark background ── */
.stApp {
    background: linear-gradient(135deg, #0A0E1A 0%, #0D1B2A 40%, #111827 100%);
    color: #E0E6F0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #0A0E1A 100%) !important;
    border-right: 1px solid rgba(30,136,229,0.2) !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
}

/* ── Sidebar text ── */
section[data-testid="stSidebar"] * {
    color: #B0BEC5 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    color: #CFD8DC !important;
    font-size: 0.92rem;
    padding: 4px 0;
}

/* ── Headers ── */
h1 { color: #E8F0FE !important; font-weight: 800; }
h2 { color: #BBDEFB !important; font-weight: 700; }
h3 { color: #90CAF9 !important; font-weight: 600; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg,#1E3A5F,#0D1B2A) !important;
    border: 1px solid rgba(30,136,229,0.25) !important;
    border-radius: 14px !important;
    padding: 18px !important;
}
[data-testid="metric-container"] label {
    color: #90A4AE !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: .6px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #E8F0FE !important;
    font-size: 1.9rem !important;
    font-weight: 700;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1565C0, #1E88E5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 1.6rem !important;
    transition: all 0.25s ease;
    box-shadow: 0 3px 12px rgba(30,136,229,0.35);
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(30,136,229,0.5) !important;
}

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: rgba(13,27,42,0.8) !important;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
[data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #90A4AE !important;
    font-weight: 500;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: linear-gradient(135deg,#1565C0,#1E88E5) !important;
    color: white !important;
}

/* ── Selectbox / inputs ── */
.stSelectbox > div > div, .stNumberInput > div > div > input {
    background: rgba(13,27,42,0.9) !important;
    border: 1px solid rgba(30,136,229,0.3) !important;
    border-radius: 8px !important;
    color: #E0E6F0 !important;
}

/* ── Success / info / warning ── */
.stAlert { border-radius: 10px !important; }
.element-container .stAlert[data-baseweb="notification"] {
    background: rgba(13,27,42,0.8) !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #1E88E5, #43A047) !important;
    border-radius: 6px;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(30,136,229,0.08) !important;
    border-radius: 8px !important;
    color: #90CAF9 !important;
    font-weight: 600;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(30,136,229,0.4) !important;
    border-radius: 14px !important;
    background: rgba(13,27,42,0.6) !important;
}

/* ── Divider ── */
hr { border-color: rgba(30,136,229,0.2) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0D1B2A; }
::-webkit-scrollbar-thumb { background: #1E88E5; border-radius: 3px; }

/* ── Custom card ── */
.insight-card {
    background: linear-gradient(135deg, rgba(30,58,95,0.8), rgba(13,27,42,0.9));
    border: 1px solid rgba(30,136,229,0.2);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    transition: all 0.3s ease;
}
.insight-card:hover {
    border-color: rgba(30,136,229,0.5);
    box-shadow: 0 4px 20px rgba(30,136,229,0.2);
    transform: translateY(-2px);
}
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: .4px;
}
.badge-blue { background: rgba(30,136,229,0.2); color: #90CAF9; border: 1px solid rgba(30,136,229,0.4); }
.badge-green { background: rgba(67,160,71,0.2); color: #A5D6A7; border: 1px solid rgba(67,160,71,0.4); }
.badge-red { background: rgba(229,57,53,0.2); color: #EF9A9A; border: 1px solid rgba(229,57,53,0.4); }
.badge-orange { background: rgba(251,140,0,0.2); color: #FFCC80; border: 1px solid rgba(251,140,0,0.4); }
</style>
""", unsafe_allow_html=True)


# ── Session State Defaults ────────────────────────────────────────────────────
def init_session():
    defaults = {
        "raw_df": None,
        "clean_df": None,
        "segmented_df": None,
        "churn_results": None,
        "best_model_name": None,
        "best_model": None,
        "best_scaler": None,
        "feature_names": None,
        "current_page": "🏠 Home",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo / branding
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px;">
        <div style="font-size:2.8rem;">📊</div>
        <div style="font-size:1.0rem; font-weight:700; color:#E8F0FE; letter-spacing:.5px;">
            Smart Analytics
        </div>
        <div style="font-size:0.72rem; color:#546E7A; margin-top:2px;">
            Customer Intelligence Platform
        </div>
    </div>
    <hr style="border-color:rgba(30,136,229,0.2); margin-bottom:12px;">
    """, unsafe_allow_html=True)

    pages = [
        "🏠 Home",
        "📂 Upload Dataset",
        "🧹 Data Cleaning",
        "📊 Exploratory Data Analysis",
        "👥 Customer Segmentation",
        "🤖 Churn Prediction",
        "📈 Dashboard",
        "💡 Business Insights",
        "📄 Download Report",
    ]
    page = st.radio("Navigation", pages, index=pages.index(st.session_state["current_page"]))
    st.session_state["current_page"] = page

    st.markdown("<hr style='border-color:rgba(30,136,229,0.2);'>", unsafe_allow_html=True)

    # Dataset status indicator
    if st.session_state.raw_df is not None:
        df_shape = st.session_state.raw_df.shape
        st.markdown(f"""
        <div style="background:rgba(67,160,71,0.12);border:1px solid rgba(67,160,71,0.3);
             border-radius:10px;padding:12px;text-align:center;">
            <div style="color:#A5D6A7;font-size:0.78rem;font-weight:600;">✅ DATASET LOADED</div>
            <div style="color:#E0E6F0;font-size:0.85rem;margin-top:4px;">
                {df_shape[0]:,} rows × {df_shape[1]} cols
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(251,140,0,0.1);border:1px solid rgba(251,140,0,0.3);
             border-radius:10px;padding:12px;text-align:center;">
            <div style="color:#FFCC80;font-size:0.78rem;font-weight:600;">⚠️ NO DATASET</div>
            <div style="color:#90A4AE;font-size:0.78rem;margin-top:4px;">
                Upload data to begin
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="position:fixed;bottom:20px;font-size:0.68rem;color:#37474F;text-align:center;">
        Smart Analytics Platform v1.0<br>Built with Python & Streamlit
    </div>""", unsafe_allow_html=True)


# ── Page Router ───────────────────────────────────────────────────────────────
if page == "🏠 Home":
    from pages.Dashboard_Home import show
    show()
elif page == "📂 Upload Dataset":
    from pages.Data_Upload import show
    show()
elif page == "🧹 Data Cleaning":
    from pages.Data_Cleaning import show
    show()
elif page == "📊 Exploratory Data Analysis":
    from pages.EDA import show
    show()
elif page == "👥 Customer Segmentation":
    from pages.Segmentation import show
    show()
elif page == "🤖 Churn Prediction":
    from pages.Churn_Prediction import show
    show()
elif page == "📈 Dashboard":
    from pages.Dashboard import show
    show()
elif page == "💡 Business Insights":
    from pages.Business_Insights import show
    show()
elif page == "📄 Download Report":
    from pages.Report import show
    show()
