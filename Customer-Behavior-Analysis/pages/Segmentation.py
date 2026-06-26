"""
Customer Segmentation Page – Segmentation.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.segmentation import (
    select_segmentation_features, run_kmeans, pca_reduce,
    cluster_summary, get_cluster_profiles, elbow_inertias,
    assign_cluster_label,
)
from utils.visualization import (
    plot_cluster_scatter, plot_cluster_bar, PRIMARY_PALETTE, _base_layout,
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


def show():
    st.markdown("## 👥 Customer Segmentation")
    if not _guard():
        return

    df = _get_df().copy()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    auto_features = select_segmentation_features(df)

    # ── Configuration ─────────────────────────────────────────────────────────
    st.markdown("### ⚙️ Clustering Configuration")
    col1, col2 = st.columns([2, 1])
    with col1:
        features = st.multiselect(
            "Select features for clustering",
            num_cols,
            default=auto_features,
            help="Choose numeric columns that best describe customer behavior.",
        )
    with col2:
        n_clusters = st.slider("Number of Clusters (K)", 2, 8, 4)
        run_btn = st.button("🚀  Run KMeans Clustering", use_container_width=True)

    if not features:
        st.info("Please select at least 2 features to run clustering.")
        return

    # ── Elbow Curve ───────────────────────────────────────────────────────────
    with st.expander("📉 Elbow Method (find optimal K)"):
        with st.spinner("Computing inertias…"):
            inertias = elbow_inertias(df, features, max_k=10)
        ks = list(range(1, 11))
        fig_elbow = px.line(
            x=ks, y=inertias,
            markers=True,
            labels={"x": "Number of Clusters (K)", "y": "Inertia"},
            template=THEME,
            color_discrete_sequence=[PRIMARY_PALETTE[0]],
        )
        fig_elbow.update_traces(line=dict(width=2.5), marker=dict(size=8))
        fig_elbow.update_layout(
            paper_bgcolor=BG, plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#B0BEC5"), height=350,
            title=dict(text="Elbow Curve", font=dict(color="#E0E6F0"), x=0.03),
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    # ── Run clustering ────────────────────────────────────────────────────────
    if run_btn or st.session_state.get("segmented_df") is not None:
        if run_btn or st.session_state.get("segmented_df") is None:
            with st.spinner("Running KMeans…"):
                df_seg, km_model, seg_scaler = run_kmeans(df, features, n_clusters=n_clusters)
                df_seg = pca_reduce(df_seg, features, scaler=seg_scaler)
                df_seg["ClusterLabel"] = df_seg["Cluster"].apply(assign_cluster_label)
                st.session_state["segmented_df"] = df_seg
                st.success(f"✅ Clustered {len(df_seg):,} customers into **{n_clusters}** segments!")
        else:
            df_seg = st.session_state["segmented_df"]

        st.markdown("---")

        # ── Cluster profiles ──────────────────────────────────────────────────
        st.markdown("### 🏷️ Cluster Profiles")
        valid_feats = [f for f in features if f in df_seg.columns]
        profiles = get_cluster_profiles(df_seg, valid_feats)
        profiles_disp = profiles.copy()
        profiles_disp.index = profiles_disp["Label"]
        profiles_disp = profiles_disp.drop(columns=["Label"])
        st.dataframe(profiles_disp.round(2), use_container_width=True)

        # ── KPI row ───────────────────────────────────────────────────────────
        st.markdown("### 📊 Cluster Distribution")
        kpi_cols = st.columns(min(n_clusters, 6))
        for i in range(n_clusters):
            label = assign_cluster_label(i)
            size = int((df_seg["Cluster"] == i).sum())
            pct = round(size / len(df_seg) * 100, 1)
            with kpi_cols[i % len(kpi_cols)]:
                st.markdown(f"""
                <div style="
                    background:linear-gradient(135deg,{PRIMARY_PALETTE[i % len(PRIMARY_PALETTE)]}22,#0D1B2A);
                    border:1px solid {PRIMARY_PALETTE[i % len(PRIMARY_PALETTE)]}55;
                    border-radius:12px; padding:16px; text-align:center;">
                    <div style="font-size:1.5rem;">{label.split()[0]}</div>
                    <div style="color:#E8F0FE;font-size:1.3rem;font-weight:700;">{size:,}</div>
                    <div style="color:#90A4AE;font-size:0.8rem;">{label[2:]}</div>
                    <div style="color:{PRIMARY_PALETTE[i % len(PRIMARY_PALETTE)]};font-size:0.9rem;font-weight:600;">{pct}%</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Visualizations ────────────────────────────────────────────────────
        tab1, tab2, tab3 = st.tabs(["🗺️ 2D PCA View", "📊 Cluster Metrics", "🔍 Detailed Profiles"])

        with tab1:
            st.plotly_chart(
                plot_cluster_scatter(df_seg, "PC1", "PC2", cluster_col="Cluster"),
                use_container_width=True,
            )

        with tab2:
            if valid_feats:
                cols2 = st.columns(min(2, len(valid_feats)))
                for i, feat in enumerate(valid_feats[:6]):
                    with cols2[i % 2]:
                        cluster_mean = df_seg.groupby("Cluster")[feat].mean().reset_index()
                        cluster_mean["Label"] = cluster_mean["Cluster"].apply(assign_cluster_label)
                        fig = px.bar(
                            cluster_mean, x="Label", y=feat,
                            color="Cluster",
                            color_discrete_sequence=PRIMARY_PALETTE,
                            template=THEME, text=feat,
                        )
                        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                        fig.update_layout(
                            paper_bgcolor=BG, plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#B0BEC5"), height=340,
                            title=dict(text=f"Avg {feat} by Cluster", font=dict(color="#E0E6F0")),
                            showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True)

            # Pie of cluster sizes
            sizes = df_seg["Cluster"].value_counts().sort_index()
            labels = [assign_cluster_label(i) for i in sizes.index]
            fig_pie = go.Figure(go.Pie(
                labels=labels, values=sizes.values,
                marker=dict(colors=PRIMARY_PALETTE),
                hole=0.45,
                textfont=dict(size=12),
            ))
            fig_pie.update_layout(
                title=dict(text="Cluster Size Distribution", font=dict(color="#E0E6F0"), x=0.03),
                paper_bgcolor=BG, font=dict(color="#B0BEC5"), height=360,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with tab3:
            sel_clust = st.selectbox("Select Cluster", list(range(n_clusters)),
                                     format_func=assign_cluster_label)
            cluster_df = df_seg[df_seg["Cluster"] == sel_clust]
            st.markdown(f"#### {assign_cluster_label(sel_clust)} – {len(cluster_df):,} customers")
            st.dataframe(cluster_df.drop(columns=["PC1", "PC2", "Cluster", "ClusterLabel"],
                                         errors="ignore").head(50),
                         use_container_width=True)

        # ── Download segmented data ───────────────────────────────────────────
        st.markdown("---")
        csv_seg = df_seg.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Segmented Dataset",
            data=csv_seg,
            file_name="segmented_customers.csv",
            mime="text/csv",
        )
