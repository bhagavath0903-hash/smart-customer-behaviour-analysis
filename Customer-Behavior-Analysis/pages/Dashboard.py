"""
Analytics Dashboard Page – Dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.visualization import (
    kpi_card_html, PRIMARY_PALETTE, _base_layout,
)

THEME = "plotly_dark"
BG = "rgba(20,26,46,0.95)"


def _get_df():
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


def _detect_col(df, keywords):
    """Return first column matching any keyword (case-insensitive)."""
    for col in df.columns:
        for kw in keywords:
            if kw in col.lower():
                return col
    return None


def show():
    st.markdown("## 📈 Analytics Dashboard")
    if not _guard():
        return

    df = _get_df().copy()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # ── Detect key columns ─────────────────────────────────────────────────────
    churn_col = _detect_col(df, ["churn"])
    income_col = _detect_col(df, ["income", "salary"])
    spend_col = _detect_col(df, ["spend", "amount", "charge", "revenue"])
    age_col = _detect_col(df, ["age"])
    gender_col = _detect_col(df, ["gender", "sex"])
    category_col = _detect_col(df, ["category", "product", "segment", "type"])
    city_col = _detect_col(df, ["city", "region", "location", "state"])
    score_col = _detect_col(df, ["score", "satisfaction"])
    tenure_col = _detect_col(df, ["tenure", "months", "duration"])

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    st.markdown("### 🎯 Key Performance Indicators")

    total = len(df)
    churn_count = 0
    active_count = total
    churn_rate = 0.0

    if churn_col:
        churn_vals = df[churn_col].astype(str).str.lower()
        churn_count = int((churn_vals.isin(["yes", "1", "true"])).sum())
        active_count = total - churn_count
        churn_rate = round(churn_count / total * 100, 1) if total > 0 else 0

    avg_income = f"₹{df[income_col].mean():,.0f}" if income_col else "N/A"
    avg_spend = f"₹{df[spend_col].mean():,.0f}" if spend_col else "N/A"
    retention_rate = round((1 - churn_count / total) * 100, 1) if total > 0 else 100

    kpi_data = [
        ("Total Customers", f"{total:,}", "👥", "#1E3A5F"),
        ("Active Customers", f"{active_count:,}", "✅", "#1B4332"),
        ("Churn Customers", f"{churn_count:,}", "⚠️", "#4A1942"),
        ("Retention Rate", f"{retention_rate}%", "🔁", "#1B3A4B"),
        ("Avg. Income", avg_income, "💰", "#2D3748"),
        ("Avg. Spending", avg_spend, "🛒", "#3D2B1F"),
    ]

    kpi_cols = st.columns(6)
    for i, (label, val, icon, bg) in enumerate(kpi_data):
        with kpi_cols[i]:
            st.markdown(kpi_card_html(label, val, icon, bg=bg), unsafe_allow_html=True)

    # ── Churn indicator bar ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:rgba(13,27,42,0.8);border:1px solid rgba(30,136,229,0.2);
         border-radius:12px;padding:16px 20px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="color:#A5D6A7;font-size:0.85rem;">✅ Active: {retention_rate}%</span>
            <span style="color:#EF9A9A;font-size:0.85rem;">⚠️ Churn: {churn_rate}%</span>
        </div>
        <div style="background:#0D1B2A;border-radius:6px;height:12px;overflow:hidden;">
            <div style="
                width:{retention_rate}%;height:100%;
                background:linear-gradient(90deg,#43A047,#1E88E5);
                border-radius:6px;transition:width 1s ease;">
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Row 1: Income + Spending distributions ────────────────────────────────
    st.markdown("### 📊 Distributions")
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        if income_col:
            fig = px.histogram(df, x=income_col, nbins=40, template=THEME,
                               color_discrete_sequence=[PRIMARY_PALETTE[0]])
            fig.update_layout(**_base_layout(f"Income Distribution"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No income column detected.")

    with r1c2:
        if spend_col:
            fig = px.histogram(df, x=spend_col, nbins=40, template=THEME,
                               color_discrete_sequence=[PRIMARY_PALETTE[2]])
            fig.update_layout(**_base_layout(f"Spending Distribution"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No spending column detected.")

    # ── Row 2: Age + Gender ───────────────────────────────────────────────────
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        if age_col:
            fig = px.histogram(df, x=age_col, nbins=30, template=THEME,
                               color_discrete_sequence=[PRIMARY_PALETTE[1]],
                               marginal="violin")
            fig.update_layout(**_base_layout("Age Distribution"))
            st.plotly_chart(fig, use_container_width=True)

    with r2c2:
        if gender_col:
            counts = df[gender_col].value_counts()
            fig = go.Figure(go.Pie(
                labels=counts.index.tolist(), values=counts.values.tolist(),
                marker=dict(colors=PRIMARY_PALETTE), hole=0.45,
            ))
            fig.update_layout(**_base_layout("Gender Distribution", 420))
            st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Category breakdown ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🛍️ Customer Segments & Category Analysis")
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        # Segmented customers
        if st.session_state.get("segmented_df") is not None:
            seg_df = st.session_state["segmented_df"]
            from utils.segmentation import assign_cluster_label
            cluster_counts = seg_df["Cluster"].value_counts().sort_index()
            labels = [assign_cluster_label(i) for i in cluster_counts.index]
            fig = go.Figure(go.Bar(
                x=labels, y=cluster_counts.values,
                marker=dict(color=PRIMARY_PALETTE[:len(labels)]),
                text=cluster_counts.values,
                textposition="outside",
            ))
            fig.update_layout(**_base_layout("Customer Segments", 420))
            st.plotly_chart(fig, use_container_width=True)
        elif cat_cols:
            first_cat = cat_cols[0]
            counts = df[first_cat].value_counts().head(10)
            fig = px.bar(x=counts.index, y=counts.values,
                         color=counts.values, color_continuous_scale="Blues",
                         template=THEME)
            fig.update_layout(**_base_layout(f"Top {first_cat} Values", 420))
            st.plotly_chart(fig, use_container_width=True)

    with r3c2:
        if category_col:
            counts_c = df[category_col].value_counts().head(10)
            fig = go.Figure(go.Bar(
                x=counts_c.values, y=counts_c.index,
                orientation="h",
                marker=dict(color=PRIMARY_PALETTE[:len(counts_c)]),
                text=counts_c.values, textposition="auto",
            ))
            fig.update_layout(**_base_layout("Top Purchasing Categories", 420))
            st.plotly_chart(fig, use_container_width=True)
        elif len(num_cols) >= 2:
            fig = px.scatter(df, x=num_cols[0], y=num_cols[1],
                             color=cat_cols[0] if cat_cols else None,
                             color_discrete_sequence=PRIMARY_PALETTE,
                             template=THEME, opacity=0.65)
            fig.update_layout(**_base_layout(f"{num_cols[0]} vs {num_cols[1]}", 420))
            st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: Churn Analysis ─────────────────────────────────────────────────
    if churn_col:
        st.markdown("---")
        st.markdown("### 🔴 Churn Analysis")
        r4c1, r4c2, r4c3 = st.columns(3)

        with r4c1:
            churn_counts = df[churn_col].astype(str).value_counts()
            fig = go.Figure(go.Pie(
                labels=churn_counts.index, values=churn_counts.values,
                marker=dict(colors=["#43A047", "#E53935"]),
                hole=0.5, textfont=dict(size=13),
            ))
            fig.update_layout(**_base_layout("Churn vs Retention", 340))
            st.plotly_chart(fig, use_container_width=True)

        with r4c2:
            if gender_col:
                churn_gender = df.groupby([gender_col, churn_col]).size().reset_index(name="Count")
                fig = px.bar(churn_gender, x=gender_col, y="Count", color=churn_col,
                             barmode="group", template=THEME,
                             color_discrete_map={"Yes": "#E53935", "No": "#43A047"})
                fig.update_layout(**_base_layout("Churn by Gender", 340))
                st.plotly_chart(fig, use_container_width=True)

        with r4c3:
            if age_col:
                df_temp = df.copy()
                df_temp["AgeGroup"] = pd.cut(
                    df_temp[age_col].fillna(df_temp[age_col].median()),
                    bins=[0, 25, 35, 45, 55, 65, 100],
                    labels=["<25", "25-35", "35-45", "45-55", "55-65", "65+"]
                )
                churn_age = df_temp.groupby(["AgeGroup", churn_col]).size().reset_index(name="Count")
                fig = px.bar(churn_age, x="AgeGroup", y="Count", color=churn_col,
                             barmode="stack", template=THEME,
                             color_discrete_map={"Yes": "#E53935", "No": "#43A047"})
                fig.update_layout(**_base_layout("Churn by Age Group", 340))
                st.plotly_chart(fig, use_container_width=True)

    # ── Row 5: Income vs Spending heatmap / scatter ───────────────────────────
    st.markdown("---")
    if income_col and spend_col and churn_col:
        st.markdown("### 🔥 Income vs Spending – Churn Heatmap")
        df_temp2 = df.copy()
        df_temp2["IncomeGroup"] = pd.cut(
            df_temp2[income_col].fillna(df_temp2[income_col].median()),
            bins=6
        ).astype(str)
        df_temp2["SpendGroup"] = pd.cut(
            df_temp2[spend_col].fillna(df_temp2[spend_col].median()),
            bins=6
        ).astype(str)
        heat_data = df_temp2.groupby(["IncomeGroup", "SpendGroup"])[churn_col].apply(
            lambda x: (x.astype(str).str.lower().isin(["yes", "1"])).mean() * 100
        ).reset_index(name="ChurnRate%")
        pivot = heat_data.pivot(index="IncomeGroup", columns="SpendGroup", values="ChurnRate%")
        fig = px.imshow(
            pivot, color_continuous_scale="RdYlGn_r",
            template=THEME, text_auto=".1f",
            labels=dict(color="Churn Rate (%)"),
        )
        fig.update_layout(**_base_layout("Churn Rate by Income & Spending Band", 440))
        st.plotly_chart(fig, use_container_width=True)

    # ── ML Prediction Summary ────────────────────────────────────────────────
    if st.session_state.get("churn_results"):
        st.markdown("---")
        st.markdown("### 🤖 ML Model Performance Summary")
        results = st.session_state["churn_results"]
        best = st.session_state["best_model_name"]

        rows = []
        for name, res in results.items():
            rows.append({
                "Model": f"{'🏆 ' if name == best else ''}{name}",
                "Accuracy": res["accuracy"],
                "Precision": res["precision"],
                "Recall": res["recall"],
                "F1 Score": res["f1"],
                "AUC": res["auc_score"],
            })
        ml_df = pd.DataFrame(rows)
        fig = go.Figure()
        for metric in ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]:
            fig.add_trace(go.Bar(
                name=metric, x=ml_df["Model"], y=ml_df[metric],
                text=ml_df[metric].round(3), textposition="outside",
            ))
        fig.update_layout(
            barmode="group",
            **_base_layout("Model Performance Comparison", 420),
            colorway=PRIMARY_PALETTE,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Correlation matrix ────────────────────────────────────────────────────
    st.markdown("---")
    if len(num_cols) >= 3:
        st.markdown("### 🔗 Correlation Heatmap")
        from utils.visualization import plot_correlation_heatmap
        st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)
