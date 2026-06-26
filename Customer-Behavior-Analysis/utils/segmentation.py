"""
Segmentation Utilities
KMeans clustering, PCA reduction, and cluster labelling.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


CLUSTER_LABELS = {
    0: "💎 Premium Customers",
    1: "🟢 Regular Customers",
    2: "💰 Budget Customers",
    3: "⚠️ At Risk Customers",
    4: "🔵 New Customers",
    5: "🏆 Loyal Customers",
}


def select_segmentation_features(df: pd.DataFrame) -> list:
    """
    Auto-select the best numeric columns for clustering.
    Prefers columns containing keywords: spend, income, age, frequency, score, amount.
    """
    keywords = ["spend", "income", "age", "frequency", "score", "amount",
                "purchase", "value", "tenure", "balance", "salary"]
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    preferred = [c for c in num_cols if any(k in c.lower() for k in keywords)]
    return preferred if len(preferred) >= 2 else num_cols[:min(6, len(num_cols))]


def run_kmeans(df: pd.DataFrame, features: list, n_clusters: int = 4,
               random_state: int = 42) -> tuple:
    """
    Fit KMeans and return (df_with_cluster, model, scaler).
    """
    df = df.copy()
    X = df[features].copy().fillna(df[features].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=12)
    labels = km.fit_predict(X_scaled)
    df["Cluster"] = labels
    return df, km, scaler


def pca_reduce(df: pd.DataFrame, features: list, scaler=None) -> pd.DataFrame:
    """
    PCA 2-D reduction for cluster visualization.
    Returns df with PC1 and PC2 columns added.
    """
    X = df[features].copy().fillna(df[features].median())
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)
    df = df.copy()
    df["PC1"] = components[:, 0]
    df["PC2"] = components[:, 1]
    return df


def cluster_summary(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Compute mean of each feature grouped by cluster."""
    grp = df.groupby("Cluster")[features].mean().round(2)
    grp.index.name = "Cluster"
    return grp


def assign_cluster_label(cluster_id: int) -> str:
    """Map a cluster integer id to a human-readable label."""
    return CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}")


def elbow_inertias(df: pd.DataFrame, features: list, max_k: int = 10) -> list:
    """Return inertia values for k = 1..max_k for elbow plotting."""
    X = df[features].copy().fillna(df[features].median())
    X_scaled = StandardScaler().fit_transform(X)
    inertias = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=8)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    return inertias


def get_cluster_profiles(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    For each cluster return a profile row: mean values + size + percentage.
    """
    summary = cluster_summary(df, features)
    counts = df["Cluster"].value_counts().sort_index()
    summary["Size"] = counts
    summary["Percentage"] = (counts / len(df) * 100).round(1)
    summary["Label"] = [assign_cluster_label(i) for i in summary.index]
    return summary
