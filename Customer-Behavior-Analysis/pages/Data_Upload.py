"""
Data Upload Page – Data_Upload.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.preprocessing import load_data, get_dataset_info


def show():
    st.markdown("## 📂 Upload Customer Dataset")
    st.markdown(
        "<p style='color:#78909C;'>Upload your CSV or Excel file to begin the analysis pipeline.</p>",
        unsafe_allow_html=True,
    )

    # ── Upload section ────────────────────────────────────────────────────────
    col_up, col_sample = st.columns([2, 1])
    with col_up:
        uploaded_file = st.file_uploader(
            "Drop your CSV / Excel file here",
            type=["csv", "xlsx", "xls"],
            help="Supported formats: CSV, XLSX, XLS",
        )

    with col_sample:
        st.markdown("<br>", unsafe_allow_html=True)
        sample_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "sample_customer_data.csv"
        )
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Sample CSV",
                    data=f,
                    file_name="sample_customer_data.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            if st.button("📋 Load Sample Dataset", use_container_width=True):
                df = pd.read_csv(sample_path)
                st.session_state["raw_df"] = df
                st.session_state["clean_df"] = None
                st.success("✅ Sample dataset loaded successfully!")
                st.rerun()
        else:
            st.info("Generate sample data first via `python generate_data.py`")

    # ── Handle uploaded file ──────────────────────────────────────────────────
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.session_state["raw_df"] = df
            st.session_state["clean_df"] = None
            st.success(f"✅ **{uploaded_file.name}** loaded successfully! ({len(df):,} rows)")
        except ValueError as e:
            st.error(f"❌ {e}")
            return

    df = st.session_state.get("raw_df")
    if df is None:
        st.markdown("""
        <div style="text-align:center; padding:60px 20px;">
            <div style="font-size:3rem;">📁</div>
            <h3 style="color:#546E7A;">No dataset loaded</h3>
            <p style="color:#37474F;">Upload a file above or load the sample dataset to begin.</p>
        </div>""", unsafe_allow_html=True)
        return

    # ── Dataset overview metrics ──────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 📋 Dataset Overview")
    info = get_dataset_info(df)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Rows", f"{info['rows']:,}", help="Number of records")
    m2.metric("Total Columns", info["columns"])
    m3.metric("Duplicate Rows", info["duplicates"], delta=f"-{info['duplicates']} to remove" if info["duplicates"] > 0 else None)
    m4.metric("Missing Values", info["missing_total"], delta=f"{info['missing_total']} cells" if info["missing_total"] > 0 else "None ✅")
    m5.metric("Memory Usage", f"{info['memory_mb']} MB")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["🗂️ Data Preview", "📐 Data Types", "📊 Statistics", "⚠️ Missing Values"])

    with tab1:
        n_rows = st.slider("Rows to preview", min_value=5, max_value=min(100, len(df)), value=10)
        st.dataframe(df.head(n_rows), use_container_width=True, height=360)

    with tab2:
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": [str(t) for t in df.dtypes],
            "Non-Null Count": df.notnull().sum().values,
            "Null Count": df.isnull().sum().values,
            "Null %": (df.isnull().sum().values / len(df) * 100).round(2),
            "Unique Values": [df[c].nunique() for c in df.columns],
        })
        st.dataframe(dtype_df, use_container_width=True, height=420)

    with tab3:
        st.dataframe(df.describe(include="all").T.round(3), use_container_width=True, height=420)

    with tab4:
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            st.success("🎉 No missing values found in the dataset!")
        else:
            miss_df = pd.DataFrame({
                "Column": missing.index,
                "Missing Count": missing.values,
                "Missing %": (missing.values / len(df) * 100).round(2),
            })
            st.dataframe(miss_df, use_container_width=True)
            import plotly.express as px
            fig = px.bar(
                miss_df, x="Column", y="Missing %",
                color="Missing %", color_continuous_scale="Reds",
                template="plotly_dark", text="Missing %",
            )
            fig.update_layout(
                paper_bgcolor="rgba(20,26,46,0.95)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#B0BEC5"),
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Column search ─────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 🔍 Search & Filter")
    col_search, col_val = st.columns(2)
    with col_search:
        search_col = st.selectbox("Select column", ["– select –"] + df.columns.tolist())
    with col_val:
        if search_col != "– select –":
            uniq = df[search_col].dropna().unique()[:200]
            val = st.selectbox("Filter value", ["– all –"] + sorted([str(x) for x in uniq]))
            if val != "– all –":
                filtered = df[df[search_col].astype(str) == val]
                st.info(f"Showing {len(filtered):,} matching rows")
                st.dataframe(filtered, use_container_width=True, height=280)

    # ── Navigate ──────────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    _, c, _ = st.columns([2, 1, 2])
    with c:
        if st.button("➡️  Proceed to Data Cleaning", use_container_width=True):
            st.session_state["current_page"] = "🧹 Data Cleaning"
            st.rerun()
