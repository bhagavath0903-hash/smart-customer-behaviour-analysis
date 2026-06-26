"""
Exploratory Data Analysis Page – EDA.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.visualization import (
    plot_histogram, plot_box, plot_pie, plot_bar,
    plot_scatter, plot_correlation_heatmap, plot_missing_heatmap,
)

THEME = "plotly_dark"
BG = "rgba(20,26,46,0.95)"
COLORS = [
    "#1E88E5", "#43A047", "#FB8C00", "#E53935",
    "#8E24AA", "#00ACC1", "#F4511E", "#6D4C41",
]


def _get_df() -> pd.DataFrame:
    df = st.session_state.get("clean_df")
    if df is None:
        df = st.session_state.get("raw_df")
    return df


def _guard():
    if _get_df() is None:
        st.warning("⚠️ Please upload a dataset first.")
        if st.button("Go to Upload"):
            st.session_state["current_page"] = "📂 Upload Dataset"
            st.rerun()
        return False
    return True


def _layout(title: str, h: int = 400) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=15, color="#E0E6F0"), x=0.03),
        paper_bgcolor=BG,
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#B0BEC5", family="Inter, Arial"),
        height=h,
        margin=dict(l=40, r=20, t=50, b=40),
    )


def show():
    st.markdown("## 📊 Exploratory Data Analysis")
    if not _guard():
        return

    df = _get_df()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    st.markdown(f"""
    <div style="background:rgba(30,136,229,0.1);border:1px solid rgba(30,136,229,0.25);
         border-radius:10px;padding:14px 20px;margin-bottom:16px;">
        📌 Dataset: <strong>{len(df):,}</strong> rows ×
        <strong>{len(df.columns)}</strong> columns &nbsp;|&nbsp;
        Numeric: <strong>{len(num_cols)}</strong> &nbsp;|&nbsp;
        Categorical: <strong>{len(cat_cols)}</strong>
    </div>""", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📋 Summary", "📈 Distributions", "📦 Boxplots",
        "🍩 Categorical", "🔗 Correlations", "🔵 Scatter Plots",
    ])

    # ────── TAB 1: Summary ────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("### 📋 Statistical Summary")
        st.dataframe(df.describe(include="all").T.round(3), use_container_width=True)

        st.markdown("### ⚠️ Missing Values")
        st.plotly_chart(plot_missing_heatmap(df), use_container_width=True)

    # ────── TAB 2: Distributions ─────────────────────────────────────────────
    with tabs[1]:
        st.markdown("### 📈 Numeric Distributions")
        if not num_cols:
            st.info("No numeric columns found.")
        else:
            sel = st.selectbox("Select Column", num_cols, key="hist_col")
            col_h, col_v = st.columns([3, 1])
            with col_h:
                st.plotly_chart(plot_histogram(df, sel), use_container_width=True)
            with col_v:
                desc = df[sel].describe()
                st.metric("Mean", f"{desc['mean']:.2f}")
                st.metric("Median", f"{df[sel].median():.2f}")
                st.metric("Std Dev", f"{desc['std']:.2f}")
                st.metric("Min / Max", f"{desc['min']:.2f} / {desc['max']:.2f}")
                skew = df[sel].skew()
                st.metric("Skewness", f"{skew:.3f}")

            # Auto-generate histograms for all numeric
            st.markdown("---")
            st.markdown("#### All Numeric Columns")
            cols_per_row = 3
            rows = [num_cols[i:i+cols_per_row] for i in range(0, len(num_cols), cols_per_row)]
            for row in rows:
                cols_ui = st.columns(cols_per_row)
                for j, c in enumerate(row):
                    with cols_ui[j]:
                        fig = px.histogram(
                            df, x=c, nbins=30,
                            color_discrete_sequence=[COLORS[j % len(COLORS)]],
                            template=THEME,
                        )
                        fig.update_layout(**_layout(c, 280))
                        st.plotly_chart(fig, use_container_width=True)

    # ────── TAB 3: Boxplots ───────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("### 📦 Boxplots")
        col1, col2 = st.columns(2)
        with col1:
            box_col = st.selectbox("Numeric Column", num_cols, key="box_num")
        with col2:
            group = st.selectbox("Group By (optional)", ["– none –"] + cat_cols, key="box_cat")

        g = None if group == "– none –" else group
        st.plotly_chart(plot_box(df, box_col, group_col=g), use_container_width=True)

        # All boxplots grid
        if num_cols:
            st.markdown("#### All Numeric Boxplots")
            cols_ui = st.columns(min(4, len(num_cols)))
            for i, nc in enumerate(num_cols[:12]):
                with cols_ui[i % 4]:
                    fig = px.box(
                        df, y=nc, template=THEME,
                        color_discrete_sequence=[COLORS[i % len(COLORS)]],
                    )
                    fig.update_layout(**_layout(nc, 270))
                    st.plotly_chart(fig, use_container_width=True)

    # ────── TAB 4: Categorical ────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("### 🍩 Categorical Analysis")
        if not cat_cols:
            st.info("No categorical columns found.")
        else:
            sel_cat = st.selectbox("Select Categorical Column", cat_cols, key="cat_col")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_bar(df, sel_cat), use_container_width=True)
            with c2:
                st.plotly_chart(plot_pie(df, sel_cat), use_container_width=True)

            # Value counts table
            vc = df[sel_cat].value_counts().reset_index()
            vc.columns = [sel_cat, "Count"]
            vc["Percentage"] = (vc["Count"] / len(df) * 100).round(2)
            st.dataframe(vc, use_container_width=True)

            st.markdown("---")
            st.markdown("#### All Categorical Columns")
            rows_c = [cat_cols[i:i+3] for i in range(0, len(cat_cols), 3)]
            for row in rows_c:
                cc = st.columns(3)
                for j, c in enumerate(row):
                    with cc[j]:
                        fig = plot_pie(df, c)
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)

    # ────── TAB 5: Correlations ───────────────────────────────────────────────
    with tabs[4]:
        st.markdown("### 🔗 Correlation Analysis")
        if len(num_cols) < 2:
            st.info("Need at least 2 numeric columns.")
        else:
            st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)

            # Top correlations table
            corr_mat = df[num_cols].corr().abs()
            upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
            top_corr = (
                upper.stack().reset_index()
                .rename(columns={"level_0": "Feature A", "level_1": "Feature B", 0: "|Correlation|"})
                .sort_values("|Correlation|", ascending=False)
                .head(20)
            )
            st.markdown("#### Top 20 Correlated Pairs")
            st.dataframe(top_corr.reset_index(drop=True).round(4), use_container_width=True)

    # ────── TAB 6: Scatter Plots ──────────────────────────────────────────────
    with tabs[5]:
        st.markdown("### 🔵 Scatter Plot Explorer")
        if len(num_cols) < 2:
            st.info("Need at least 2 numeric columns.")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("X Axis", num_cols, key="sc_x")
            with col2:
                y_col = st.selectbox("Y Axis", num_cols, index=1 if len(num_cols) > 1 else 0, key="sc_y")
            with col3:
                color_col = st.selectbox("Color By", ["– none –"] + cat_cols, key="sc_c")

            cc = None if color_col == "– none –" else color_col
            st.plotly_chart(plot_scatter(df, x_col, y_col, color=cc), use_container_width=True)

            # Scatter matrix (Pair Plot) – limited columns
            if len(num_cols) >= 3:
                st.markdown("---")
                st.markdown("#### Pair Plot (Scatter Matrix)")
                pair_cols = num_cols[:min(5, len(num_cols))]
                color_pair = cat_cols[0] if cat_cols else None
                fig = px.scatter_matrix(
                    df, dimensions=pair_cols,
                    color=color_pair,
                    color_discrete_sequence=COLORS,
                    template=THEME,
                )
                fig.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.6))
                fig.update_layout(
                    paper_bgcolor=BG,
                    font=dict(color="#B0BEC5"),
                    height=600,
                )
                st.plotly_chart(fig, use_container_width=True)
