"""
Data Cleaning Page – Data_Cleaning.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.preprocessing import (
    remove_duplicates, handle_missing_values,
    label_encode_columns, one_hot_encode_columns,
    scale_features, auto_clean_pipeline,
)


def _guard():
    if st.session_state.get("raw_df") is None:
        st.warning("⚠️ Please upload a dataset first.")
        if st.button("Go to Upload"):
            st.session_state["current_page"] = "📂 Upload Dataset"
            st.rerun()
        return False
    return True


def show():
    st.markdown("## 🧹 Data Cleaning & Preprocessing")
    if not _guard():
        return

    df_raw = st.session_state["raw_df"].copy()

    # ── Sidebar options ───────────────────────────────────────────────────────
    st.markdown("### ⚙️ Cleaning Configuration")
    col1, col2, col3 = st.columns(3)

    with col1:
        remove_dup = st.checkbox("Remove Duplicate Rows", value=True)
        impute_missing = st.checkbox("Handle Missing Values", value=True)
        impute_strategy = st.selectbox(
            "Imputation Strategy", ["median", "mean", "most_frequent"],
            disabled=not impute_missing,
        )

    with col2:
        do_label_enc = st.checkbox("Label Encode Categorical", value=False)
        cat_cols_all = df_raw.select_dtypes(include="object").columns.tolist()
        le_cols = st.multiselect(
            "Columns to Label Encode", cat_cols_all,
            disabled=not do_label_enc,
        )

        do_ohe = st.checkbox("One-Hot Encode Categorical", value=False)
        ohe_cols = st.multiselect(
            "Columns to OHE", cat_cols_all,
            disabled=not do_ohe,
        )

    with col3:
        do_scale = st.checkbox("Scale Numeric Features", value=False)
        scale_method = st.selectbox("Scaling Method", ["standard", "minmax"], disabled=not do_scale)
        num_cols_all = df_raw.select_dtypes(include=np.number).columns.tolist()
        scale_cols = st.multiselect(
            "Columns to Scale", num_cols_all,
            disabled=not do_scale,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🚀  Apply Cleaning Pipeline", use_container_width=False)

    if run_btn:
        df_clean = df_raw.copy()
        steps_log = []

        with st.spinner("Applying cleaning steps…"):
            if remove_dup:
                before = len(df_clean)
                df_clean = remove_duplicates(df_clean)
                removed = before - len(df_clean)
                steps_log.append(f"✅ Removed **{removed}** duplicate rows")

            if impute_missing:
                df_clean = handle_missing_values(df_clean, strategy=impute_strategy)
                steps_log.append(f"✅ Missing values imputed using **{impute_strategy}** strategy")

            if do_label_enc and le_cols:
                df_clean, _ = label_encode_columns(df_clean, le_cols)
                steps_log.append(f"✅ Label encoded: {', '.join(le_cols)}")

            if do_ohe and ohe_cols:
                df_clean = one_hot_encode_columns(df_clean, ohe_cols)
                steps_log.append(f"✅ One-Hot encoded: {', '.join(ohe_cols)}")

            if do_scale and scale_cols:
                df_clean, _ = scale_features(df_clean, scale_cols, method=scale_method)
                steps_log.append(f"✅ Scaled using **{scale_method}**: {', '.join(scale_cols)}")

        st.session_state["clean_df"] = df_clean

        for msg in steps_log:
            st.success(msg)
        st.balloons()

    # ── Before / After comparison ─────────────────────────────────────────────
    df_clean = st.session_state.get("clean_df")
    if df_clean is not None:
        st.markdown("---")
        st.markdown("### 📊 Before vs. After")
        col_b, col_a = st.columns(2)
        with col_b:
            st.markdown("#### 🔴 Raw Dataset")
            m1, m2 = st.columns(2)
            m1.metric("Rows", f"{len(df_raw):,}")
            m2.metric("Columns", df_raw.shape[1])
            m3, m4 = st.columns(2)
            m3.metric("Missing", int(df_raw.isnull().sum().sum()))
            m4.metric("Duplicates", int(df_raw.duplicated().sum()))

        with col_a:
            st.markdown("#### 🟢 Cleaned Dataset")
            m1, m2 = st.columns(2)
            m1.metric("Rows", f"{len(df_clean):,}")
            m2.metric("Columns", df_clean.shape[1])
            m3, m4 = st.columns(2)
            m3.metric("Missing", int(df_clean.isnull().sum().sum()))
            m4.metric("Duplicates", int(df_clean.duplicated().sum()))

        # ── Preview & download ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🗂️ Cleaned Dataset Preview")
        n = st.slider("Rows", 5, min(100, len(df_clean)), 10)
        st.dataframe(df_clean.head(n), use_container_width=True)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_bytes = df_clean.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Cleaned CSV",
                data=csv_bytes,
                file_name="cleaned_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_dl2:
            buf = io.BytesIO()
            df_clean.to_excel(buf, index=False)
            st.download_button(
                "⬇️ Download Cleaned Excel",
                data=buf.getvalue(),
                file_name="cleaned_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        # ── Data types after cleaning ──────────────────────────────────────────
        with st.expander("🔎 Column Data Types After Cleaning"):
            dtype_df = pd.DataFrame({
                "Column": df_clean.columns,
                "Type": [str(t) for t in df_clean.dtypes],
                "Non-Null": df_clean.notnull().sum().values,
                "Null": df_clean.isnull().sum().values,
            })
            st.dataframe(dtype_df, use_container_width=True)

        st.markdown("---")
        _, c, _ = st.columns([2, 1, 2])
        with c:
            if st.button("➡️  Go to EDA", use_container_width=True):
                st.session_state["current_page"] = "📊 Exploratory Data Analysis"
                st.rerun()
    else:
        st.markdown("""
        <div style="text-align:center; padding:40px;">
            <div style="font-size:2.5rem;">🧹</div>
            <p style="color:#546E7A;">Configure options above and click <strong>Apply Cleaning Pipeline</strong>.</p>
        </div>""", unsafe_allow_html=True)
