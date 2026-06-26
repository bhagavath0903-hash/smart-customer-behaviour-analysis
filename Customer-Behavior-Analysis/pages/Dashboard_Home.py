"""
Home Page – Dashboard_Home.py
"""

import streamlit as st


def show():
    # ── Hero Section ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        background: linear-gradient(135deg,rgba(21,101,192,0.25),rgba(30,136,229,0.12));
        border:1px solid rgba(30,136,229,0.25);
        border-radius:20px; padding:48px 40px 36px; text-align:center;
        margin-bottom:32px; position:relative; overflow:hidden;">
        <div style="
            position:absolute;top:0;left:0;right:0;bottom:0;
            background:radial-gradient(ellipse at 50% 0%,rgba(30,136,229,0.15),transparent 70%);
            pointer-events:none;">
        </div>
        <div style="font-size:3.5rem; margin-bottom:12px;">📊</div>
        <h1 style="
            font-size:2.6rem; font-weight:800; margin:0;
            background:linear-gradient(135deg,#E8F0FE,#90CAF9,#1E88E5);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            background-clip:text;">
            Smart Customer Behavior Analysis
        </h1>
        <h2 style="
            font-size:1.3rem; color:#90A4AE; font-weight:400;
            margin:8px 0 0; -webkit-text-fill-color:#90A4AE;">
            & Churn Prediction System
        </h2>
        <p style="
            color:#78909C; font-size:1.0rem; max-width:640px;
            margin:16px auto 0; line-height:1.7;">
            An end-to-end AI-powered customer intelligence platform for data-driven
            business decisions — segmentation, churn prediction, and actionable insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Objectives ────────────────────────────────────────────────────────────
    st.markdown("## 🎯 Platform Objectives")
    cols = st.columns(3)
    objectives = [
        ("🔍", "Understand Behavior", "Deep-dive into purchase patterns, frequency, and spending habits across customer segments."),
        ("👥", "Segment Customers", "Group customers using KMeans clustering to reveal Premium, Regular, Budget, and At-Risk profiles."),
        ("🤖", "Predict Churn", "Train ML models to identify customers likely to churn before it happens."),
        ("📈", "Interactive Dashboard", "Professional BI-style dashboard with live KPI cards, charts, and filters."),
        ("💡", "Business Insights", "Auto-generate executive-level recommendations and retention strategies."),
        ("📄", "Export Reports", "Download PDF/Excel reports with charts, segments, predictions, and insights."),
    ]
    for i, (icon, title, desc) in enumerate(objectives):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="insight-card" style="text-align:center; min-height:170px;">
                <div style="font-size:2rem;">{icon}</div>
                <div style="font-size:1.0rem; font-weight:700; color:#90CAF9; margin:8px 0 6px;">{title}</div>
                <div style="font-size:0.83rem; color:#78909C; line-height:1.6;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    # ── Workflow ───────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## 🔄 Analysis Workflow")
    steps = [
        ("1", "Upload Data", "CSV/Excel"),
        ("2", "Clean Data", "Handle NaN"),
        ("3", "EDA", "Visualize"),
        ("4", "Segment", "KMeans"),
        ("5", "Predict", "ML Models"),
        ("6", "Insights", "Report"),
    ]
    cols2 = st.columns(len(steps))
    for i, (num, title, sub) in enumerate(steps):
        with cols2[i]:
            arrow = "→" if i < len(steps) - 1 else "✅"
            st.markdown(f"""
            <div style="text-align:center;">
                <div style="
                    width:54px; height:54px; margin:0 auto 10px;
                    background:linear-gradient(135deg,#1565C0,#1E88E5);
                    border-radius:50%; display:flex; align-items:center;
                    justify-content:center; font-size:1.1rem; font-weight:700;
                    color:white; box-shadow:0 4px 14px rgba(30,136,229,0.4);">
                    {num}
                </div>
                <div style="font-size:0.88rem; font-weight:600; color:#90CAF9;">{title}</div>
                <div style="font-size:0.74rem; color:#546E7A; margin-top:3px;">{sub}</div>
            </div>""", unsafe_allow_html=True)

    # ── Tech Stack ─────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## ⚙️ Technology Stack")
    techs = [
        ("🐍", "Python 3.10+", "#FFD600"),
        ("🌊", "Streamlit", "#FF4B4B"),
        ("🐼", "Pandas", "#150458"),
        ("🔢", "NumPy", "#4DABCF"),
        ("📉", "Matplotlib", "#11557C"),
        ("🎨", "Seaborn", "#4C72B0"),
        ("📡", "Plotly", "#3D9BE9"),
        ("🤖", "Scikit-Learn", "#F7931E"),
    ]
    t_cols = st.columns(8)
    for i, (icon, name, color) in enumerate(techs):
        with t_cols[i]:
            st.markdown(f"""
            <div style="
                text-align:center; padding:14px 8px;
                background:rgba(13,27,42,0.8);
                border:1px solid rgba(30,136,229,0.2);
                border-radius:12px;">
                <div style="font-size:1.6rem;">{icon}</div>
                <div style="font-size:0.72rem; color:#90A4AE; margin-top:6px; font-weight:500;">{name}</div>
            </div>""", unsafe_allow_html=True)

    # ── Quick Start ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="
        background:linear-gradient(135deg,rgba(67,160,71,0.15),rgba(0,172,193,0.1));
        border:1px solid rgba(67,160,71,0.25); border-radius:16px; padding:28px 32px;">
        <h3 style="color:#A5D6A7; margin:0 0 12px;">🚀 Quick Start Guide</h3>
        <ol style="color:#B0BEC5; font-size:0.9rem; line-height:2.0; margin:0; padding-left:20px;">
            <li>Navigate to <strong style="color:#90CAF9">📂 Upload Dataset</strong> and upload your customer CSV file
                or use the provided <code>sample_customer_data.csv</code></li>
            <li>Run <strong style="color:#90CAF9">🧹 Data Cleaning</strong> to handle missing values and encode features</li>
            <li>Explore patterns in <strong style="color:#90CAF9">📊 Exploratory Data Analysis</strong></li>
            <li>Segment customers in <strong style="color:#90CAF9">👥 Customer Segmentation</strong></li>
            <li>Train ML models in <strong style="color:#90CAF9">🤖 Churn Prediction</strong></li>
            <li>View the live <strong style="color:#90CAF9">📈 Dashboard</strong> for a complete overview</li>
            <li>Download your <strong style="color:#90CAF9">📄 Business Report</strong></li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_b:
        if st.button("🚀  Start Analysis  →", use_container_width=True):
            st.session_state["current_page"] = "📂 Upload Dataset"
            st.rerun()
