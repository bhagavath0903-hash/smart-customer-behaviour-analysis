"""
Visualization Utilities
Reusable Plotly and Seaborn chart functions for the analytics platform.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64

# ── Global color palette ─────────────────────────────────────────────────────
PRIMARY_PALETTE = [
    "#1E88E5", "#43A047", "#FB8C00", "#E53935", "#8E24AA",
    "#00ACC1", "#F4511E", "#6D4C41", "#1DE9B6", "#D81B60",
]

CLUSTER_COLORS = {
    0: "#1E88E5",
    1: "#43A047",
    2: "#FB8C00",
    3: "#E53935",
    4: "#8E24AA",
    5: "#00ACC1",
}

PLOTLY_THEME = "plotly_dark"


def _base_layout(title: str = "", height: int = 420) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=16, color="#E0E6F0"), x=0.03),
        paper_bgcolor="rgba(20,26,46,0.95)",
        plot_bgcolor="rgba(20,26,46,0.0)",
        font=dict(color="#B0BEC5", family="Inter, Arial"),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#B0BEC5")),
    )


# ── Distribution Charts ───────────────────────────────────────────────────────

def plot_histogram(df: pd.DataFrame, column: str, title: str = None) -> go.Figure:
    """Histogram with KDE overlay."""
    fig = px.histogram(
        df, x=column, nbins=40,
        color_discrete_sequence=[PRIMARY_PALETTE[0]],
        marginal="violin",
        title=title or f"Distribution of {column}",
        template=PLOTLY_THEME,
    )
    fig.update_layout(**_base_layout(title or f"Distribution of {column}"))
    return fig


def plot_box(df: pd.DataFrame, column: str, group_col: str = None) -> go.Figure:
    """Box plot – optional group split."""
    if group_col and group_col in df.columns:
        fig = px.box(
            df, x=group_col, y=column,
            color=group_col,
            color_discrete_sequence=PRIMARY_PALETTE,
            template=PLOTLY_THEME,
        )
    else:
        fig = px.box(
            df, y=column,
            color_discrete_sequence=[PRIMARY_PALETTE[0]],
            template=PLOTLY_THEME,
        )
    fig.update_layout(**_base_layout(f"Boxplot: {column}"))
    return fig


def plot_pie(df: pd.DataFrame, column: str, title: str = None) -> go.Figure:
    """Pie chart for categorical distribution."""
    counts = df[column].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        marker=dict(colors=PRIMARY_PALETTE),
        hole=0.4,
        textfont=dict(size=13),
    ))
    fig.update_layout(**_base_layout(title or f"{column} Distribution"))
    return fig


def plot_bar(df: pd.DataFrame, x: str, y: str = None, title: str = "",
             color: str = None, orientation: str = "v") -> go.Figure:
    """Vertical / horizontal bar chart."""
    if y is None:
        counts = df[x].value_counts().reset_index()
        counts.columns = [x, "Count"]
        fig = px.bar(
            counts, x=x, y="Count",
            color=x,
            color_discrete_sequence=PRIMARY_PALETTE,
            orientation=orientation,
            template=PLOTLY_THEME,
        )
    else:
        fig = px.bar(
            df, x=x, y=y,
            color=color or x,
            color_discrete_sequence=PRIMARY_PALETTE,
            orientation=orientation,
            template=PLOTLY_THEME,
        )
    fig.update_layout(**_base_layout(title))
    return fig


def plot_scatter(df: pd.DataFrame, x: str, y: str, color: str = None,
                 size: str = None, title: str = "") -> go.Figure:
    """Interactive scatter plot."""
    fig = px.scatter(
        df, x=x, y=y,
        color=color,
        size=size,
        color_discrete_sequence=PRIMARY_PALETTE,
        template=PLOTLY_THEME,
    )
    fig.update_layout(**_base_layout(title or f"{x} vs {y}"))
    return fig


# ── Heatmaps ─────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Plotly correlation heatmap for all numeric columns."""
    num_df = df.select_dtypes(include=np.number)
    corr = num_df.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="Blues",
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        colorbar=dict(title="Corr"),
    ))
    fig.update_layout(**_base_layout("Correlation Heatmap", height=520))
    return fig


def plot_missing_heatmap(df: pd.DataFrame) -> go.Figure:
    """Bar chart showing missing value counts per column."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        fig = go.Figure()
        fig.add_annotation(text="✅ No missing values found!", showarrow=False,
                           font=dict(size=18, color="#43A047"))
        fig.update_layout(**_base_layout("Missing Values"))
        return fig
    fig = px.bar(
        x=missing.index.tolist(), y=missing.values.tolist(),
        labels={"x": "Column", "y": "Missing Count"},
        color=missing.values.tolist(),
        color_continuous_scale="Reds",
        template=PLOTLY_THEME,
    )
    fig.update_layout(**_base_layout("Missing Values Per Column"))
    return fig


# ── Segmentation ─────────────────────────────────────────────────────────────

def plot_cluster_scatter(df: pd.DataFrame, x: str, y: str, cluster_col: str = "Cluster") -> go.Figure:
    """2-D scatter coloured by cluster."""
    df = df.copy()
    df[cluster_col] = df[cluster_col].astype(str)
    fig = px.scatter(
        df, x=x, y=y, color=cluster_col,
        color_discrete_sequence=PRIMARY_PALETTE,
        template=PLOTLY_THEME,
        symbol=cluster_col,
        size_max=12,
    )
    fig.update_traces(marker=dict(size=8, opacity=0.85))
    fig.update_layout(**_base_layout(f"Customer Segments: {x} vs {y}"))
    return fig


def plot_cluster_bar(cluster_summary: pd.DataFrame, metric: str, title: str = "") -> go.Figure:
    """Bar chart per cluster for a given metric."""
    fig = px.bar(
        cluster_summary.reset_index(), x="Cluster", y=metric,
        color="Cluster",
        color_discrete_sequence=PRIMARY_PALETTE,
        template=PLOTLY_THEME,
        text=metric,
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(**_base_layout(title or f"Average {metric} by Cluster"))
    return fig


# ── ML / Evaluation ──────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, labels: list = None) -> go.Figure:
    """Annotated confusion matrix."""
    labels = labels or ["Stay", "Churn"]
    fig = px.imshow(
        cm, text_auto=True,
        x=labels, y=labels,
        color_continuous_scale="Blues",
        template=PLOTLY_THEME,
        aspect="auto",
    )
    fig.update_layout(**_base_layout("Confusion Matrix", height=380))
    fig.update_xaxes(title="Predicted")
    fig.update_yaxes(title="Actual")
    return fig


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_score: float) -> go.Figure:
    """ROC curve with AUC annotation."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"AUC = {auc_score:.3f}",
        line=dict(color=PRIMARY_PALETTE[0], width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random",
        line=dict(color="gray", dash="dash"),
    ))
    fig.update_layout(**_base_layout("ROC Curve", height=400))
    fig.update_xaxes(title="False Positive Rate")
    fig.update_yaxes(title="True Positive Rate")
    return fig


def plot_feature_importance(features: list, importances: list) -> go.Figure:
    """Horizontal bar for feature importances."""
    pairs = sorted(zip(importances, features))
    fig = go.Figure(go.Bar(
        x=[p[0] for p in pairs],
        y=[p[1] for p in pairs],
        orientation="h",
        marker=dict(color=[p[0] for p in pairs], colorscale="Blues"),
    ))
    fig.update_layout(**_base_layout("Feature Importance", height=max(350, len(features) * 28)))
    fig.update_xaxes(title="Importance")
    return fig


# ── Dashboard KPI cards (HTML string) ────────────────────────────────────────

def kpi_card_html(label: str, value, icon: str = "📊", delta: str = None,
                  bg: str = "#1E3A5F") -> str:
    delta_html = f"<div style='font-size:0.78rem;color:#B0BEC5;margin-top:4px'>{delta}</div>" if delta else ""
    return f"""
    <div style="
        background: linear-gradient(135deg,{bg},#0D1B2A);
        border-radius:14px; padding:20px 24px;
        border:1px solid rgba(255,255,255,0.08);
        box-shadow:0 4px 18px rgba(0,0,0,0.35);
        text-align:center; min-height:110px;
        display:flex; flex-direction:column; justify-content:center;">
        <div style='font-size:2rem;'>{icon}</div>
        <div style='font-size:1.7rem; font-weight:700; color:#E8F0FE; margin:4px 0;'>{value}</div>
        <div style='font-size:0.82rem; color:#90A4AE; letter-spacing:.5px; text-transform:uppercase;'>{label}</div>
        {delta_html}
    </div>"""


def line_chart(df: pd.DataFrame, x: str, y: str, title: str = "") -> go.Figure:
    """Time-series or sequence line chart."""
    fig = px.line(
        df, x=x, y=y,
        color_discrete_sequence=[PRIMARY_PALETTE[0]],
        template=PLOTLY_THEME,
    )
    fig.update_traces(line=dict(width=2.5))
    fig.update_layout(**_base_layout(title))
    return fig


def area_chart(df: pd.DataFrame, x: str, y: str, title: str = "") -> go.Figure:
    """Filled area chart."""
    fig = px.area(
        df, x=x, y=y,
        color_discrete_sequence=[PRIMARY_PALETTE[0]],
        template=PLOTLY_THEME,
    )
    fig.update_layout(**_base_layout(title))
    return fig


def funnel_chart(stages: list, values: list, title: str = "") -> go.Figure:
    """Marketing / retention funnel."""
    fig = go.Figure(go.Funnel(
        y=stages, x=values,
        marker=dict(color=PRIMARY_PALETTE[:len(stages)]),
        textinfo="value+percent initial",
    ))
    fig.update_layout(**_base_layout(title))
    return fig


def matplotlib_to_base64(fig_mpl) -> str:
    """Convert a matplotlib figure to a base64 PNG string for embedding."""
    buf = io.BytesIO()
    fig_mpl.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                    facecolor="#141A2E")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig_mpl)
    return f"data:image/png;base64,{encoded}"
