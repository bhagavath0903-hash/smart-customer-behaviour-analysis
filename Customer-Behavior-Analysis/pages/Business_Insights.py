"""
Business Insights Page – Business_Insights.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.visualization import PRIMARY_PALETTE, _base_layout
from utils.segmentation import assign_cluster_label

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
    for col in df.columns:
        for kw in keywords:
            if kw in col.lower():
                return col
    return None


def _generate_insights(df: pd.DataFrame) -> dict:
    """Auto-generate business insights from the dataset."""
    insights = {}
    n = len(df)

    churn_col = _detect_col(df, ["churn"])
    income_col = _detect_col(df, ["income", "salary"])
    spend_col = _detect_col(df, ["spend", "amount", "charge"])
    age_col = _detect_col(df, ["age"])
    gender_col = _detect_col(df, ["gender", "sex"])
    tenure_col = _detect_col(df, ["tenure", "months"])
    score_col = _detect_col(df, ["satisfaction", "score"])
    category_col = _detect_col(df, ["category", "product"])
    support_col = _detect_col(df, ["support", "complaint", "call"])

    # Churn stats
    if churn_col:
        churn_mask = df[churn_col].astype(str).str.lower().isin(["yes", "1", "true"])
        churn_rate = round(churn_mask.mean() * 100, 2)
        insights["churn_rate"] = churn_rate
        insights["churn_count"] = int(churn_mask.sum())
        insights["active_count"] = n - int(churn_mask.sum())

        # High risk: churned + low tenure/satisfaction
        risk_mask = churn_mask.copy()
        if tenure_col:
            low_tenure = df[tenure_col] < df[tenure_col].quantile(0.25)
            risk_mask = churn_mask & low_tenure
        insights["high_risk_count"] = int(risk_mask.sum())
    else:
        insights["churn_rate"] = "N/A"
        insights["churn_count"] = "N/A"
        insights["active_count"] = n
        insights["high_risk_count"] = "N/A"

    # Revenue
    if income_col:
        insights["avg_income"] = round(df[income_col].mean(), 2)
        insights["top_income"] = round(df[income_col].quantile(0.9), 2)

    if spend_col:
        insights["avg_spend"] = round(df[spend_col].mean(), 2)
        insights["total_revenue"] = round(df[spend_col].sum(), 2)

    # Top category
    if category_col:
        top_cat = df[category_col].value_counts().idxmax()
        insights["top_category"] = top_cat
        insights["top_category_pct"] = round(
            df[category_col].value_counts().max() / n * 100, 1
        )

    # Age insights
    if age_col:
        insights["avg_age"] = round(df[age_col].mean(), 1)
        age_group_counts = pd.cut(
            df[age_col].fillna(df[age_col].median()),
            bins=[0, 25, 35, 45, 55, 100],
            labels=["<25", "25-35", "35-45", "45-55", "55+"]
        ).value_counts()
        insights["dominant_age_group"] = str(age_group_counts.idxmax())

    # Support calls
    if support_col:
        insights["avg_support_calls"] = round(df[support_col].mean(), 2)

    # Satisfaction
    if score_col:
        insights["avg_satisfaction"] = round(df[score_col].mean(), 2)

    insights["total_customers"] = n
    insights["total_columns"] = len(df.columns)

    return insights


def show():
    st.markdown("## 💡 Business Insights & Recommendations")
    if not _guard():
        return

    df = _get_df().copy()
    insights = _generate_insights(df)

    # ── Executive Summary Banner ──────────────────────────────────────────────
    churn_rate = insights.get("churn_rate", "N/A")
    churn_badge_color = "#E53935" if isinstance(churn_rate, float) and churn_rate > 20 else "#FB8C00" if isinstance(churn_rate, float) and churn_rate > 10 else "#43A047"

    st.markdown(f"""
    <div style="
        background:linear-gradient(135deg,rgba(21,101,192,0.2),rgba(13,27,42,0.95));
        border:1px solid rgba(30,136,229,0.3); border-radius:18px;
        padding:28px 32px; margin-bottom:24px;">
        <h3 style="color:#90CAF9; margin:0 0 16px;">📑 Executive Summary</h3>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;">
            <div style="background:rgba(13,27,42,0.6);border-radius:10px;padding:14px;text-align:center;">
                <div style="color:#78909C;font-size:0.75rem;text-transform:uppercase;">Total Customers</div>
                <div style="color:#E8F0FE;font-size:1.6rem;font-weight:700;">{insights['total_customers']:,}</div>
            </div>
            <div style="background:rgba(13,27,42,0.6);border-radius:10px;padding:14px;text-align:center;">
                <div style="color:#78909C;font-size:0.75rem;text-transform:uppercase;">Active</div>
                <div style="color:#A5D6A7;font-size:1.6rem;font-weight:700;">{insights['active_count']:,}</div>
            </div>
            <div style="background:rgba(13,27,42,0.6);border-radius:10px;padding:14px;text-align:center;">
                <div style="color:#78909C;font-size:0.75rem;text-transform:uppercase;">Churned</div>
                <div style="color:#EF9A9A;font-size:1.6rem;font-weight:700;">{insights['churn_count']}</div>
            </div>
            <div style="background:rgba(13,27,42,0.6);border-radius:10px;padding:14px;text-align:center;">
                <div style="color:#78909C;font-size:0.75rem;text-transform:uppercase;">Churn Rate</div>
                <div style="color:{churn_badge_color};font-size:1.6rem;font-weight:700;">{churn_rate}%</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Insight Cards ─────────────────────────────────────────────────────────
    ins_col1, ins_col2 = st.columns(2)

    with ins_col1:
        # Key Findings
        st.markdown("""
        <div class="insight-card">
            <h4 style="color:#90CAF9; margin:0 0 12px;">🔍 Key Findings</h4>
        """, unsafe_allow_html=True)

        findings = []
        if isinstance(insights.get("churn_rate"), float):
            if insights["churn_rate"] > 25:
                findings.append(("🔴", "Critical", f"High churn rate of {insights['churn_rate']}% — immediate intervention required."))
            elif insights["churn_rate"] > 15:
                findings.append(("🟠", "Warning", f"Elevated churn rate of {insights['churn_rate']}% — proactive retention needed."))
            else:
                findings.append(("🟢", "Healthy", f"Churn rate of {insights['churn_rate']}% is within acceptable bounds."))

        if insights.get("avg_satisfaction"):
            s = insights["avg_satisfaction"]
            if s < 3:
                findings.append(("🔴", "Low", f"Average satisfaction score is {s}/5 — customer experience needs improvement."))
            elif s < 4:
                findings.append(("🟠", "Moderate", f"Average satisfaction score is {s}/5 — room for improvement."))
            else:
                findings.append(("🟢", "High", f"Average satisfaction score is {s}/5 — customers are happy."))

        if insights.get("dominant_age_group"):
            findings.append(("📊", "Demographics", f"Dominant age group: {insights['dominant_age_group']} years."))

        if insights.get("top_category"):
            findings.append(("🛍️", "Category", f"Top category: {insights['top_category']} ({insights.get('top_category_pct', '')}% of purchases)."))

        for icon, badge_label, text in findings:
            badge_color = {"🔴": "badge-red", "🟠": "badge-orange", "🟢": "badge-green"}.get(icon, "badge-blue")
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:10px;">
                <span style="font-size:1.2rem;">{icon}</span>
                <div>
                    <span class="badge {badge_color}">{badge_label}</span>
                    <div style="color:#B0BEC5;font-size:0.85rem;margin-top:4px;">{text}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Revenue Opportunities
        st.markdown("""
        <div class="insight-card">
            <h4 style="color:#90CAF9; margin:0 0 12px;">💰 Revenue Opportunities</h4>
        """, unsafe_allow_html=True)
        rev_items = []
        if insights.get("avg_income") and insights.get("avg_spend"):
            ratio = insights["avg_spend"] / insights["avg_income"] * 100
            rev_items.append(f"Customers spend an average of <strong>{ratio:.1f}%</strong> of income.")
        if insights.get("total_revenue"):
            rev_items.append(f"Total dataset revenue: <strong>₹{insights['total_revenue']:,.0f}</strong>")
        if insights.get("top_income"):
            rev_items.append(f"Top 10% income customers earn <strong>₹{insights['top_income']:,.0f}+</strong> — high upsell potential.")
        if not rev_items:
            rev_items = ["Upload income/spending data to unlock revenue insights."]
        for item in rev_items:
            st.markdown(f"<p style='color:#B0BEC5;font-size:0.88rem;margin:6px 0;'>💎 {item}</p>",
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with ins_col2:
        # Retention Strategies
        st.markdown("""
        <div class="insight-card">
            <h4 style="color:#90CAF9; margin:0 0 12px;">🔁 Retention Strategies</h4>
        """, unsafe_allow_html=True)
        strategies = [
            ("📧", "Personalized Email Campaigns", "Target high-risk segments with personalized re-engagement emails offering exclusive discounts."),
            ("🎁", "Loyalty Rewards Program", "Introduce tiered loyalty points to reward consistent spenders and incentivize long-term retention."),
            ("📞", "Proactive Support Outreach", "Identify customers with 3+ support calls and offer dedicated account managers."),
            ("📱", "Mobile App Notifications", "Push personalized offers and product recommendations based on purchase history."),
            ("🤝", "Annual Contract Incentives", "Offer 15-20% discount for customers willing to upgrade to annual contracts."),
        ]
        for icon, title, desc in strategies:
            st.markdown(f"""
            <div style="display:flex;gap:10px;margin-bottom:10px;">
                <span style="font-size:1.1rem;">{icon}</span>
                <div>
                    <div style="color:#90CAF9;font-size:0.88rem;font-weight:600;">{title}</div>
                    <div style="color:#78909C;font-size:0.8rem;margin-top:2px;">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Marketing Recommendations
        st.markdown("""
        <div class="insight-card">
            <h4 style="color:#90CAF9; margin:0 0 12px;">📣 Marketing Recommendations</h4>
        """, unsafe_allow_html=True)
        recs = [
            "🎯 Launch targeted ads for the dominant age demographic.",
            "📦 Bundle top-selling categories with complementary products.",
            "🌐 Invest in digital marketing channels to reach budget-conscious segments.",
            "💡 A/B test pricing tiers to optimize conversion rates.",
            "📊 Use churn scores to prioritize outreach in customer success teams.",
            "🏆 Showcase loyalty benefits prominently in all customer touchpoints.",
        ]
        for r in recs:
            st.markdown(f"<p style='color:#B0BEC5;font-size:0.85rem;margin:5px 0;'>{r}</p>",
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Customer Segment Insights ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 👥 Segment-Level Insights")
    seg_df = st.session_state.get("segmented_df")
    if seg_df is not None:
        n_clusters = seg_df["Cluster"].nunique()
        seg_cols = st.columns(min(n_clusters, 4))
        for i in range(n_clusters):
            label = assign_cluster_label(i)
            cluster_data = seg_df[seg_df["Cluster"] == i]
            size = len(cluster_data)
            pct = round(size / len(seg_df) * 100, 1)
            num_cols_seg = cluster_data.select_dtypes(include=np.number).columns
            num_cols_seg = [c for c in num_cols_seg if c not in ["Cluster", "PC1", "PC2"]]

            with seg_cols[i % 4]:
                icon = label.split()[0]
                metrics_html = ""
                for nc in num_cols_seg[:3]:
                    val = cluster_data[nc].mean()
                    metrics_html += f"""
                    <div style="display:flex;justify-content:space-between;margin:4px 0;">
                        <span style="color:#78909C;font-size:0.75rem;">{nc}</span>
                        <span style="color:#E8F0FE;font-size:0.75rem;font-weight:600;">{val:.1f}</span>
                    </div>"""

                st.markdown(f"""
                <div style="
                    background:linear-gradient(135deg,{PRIMARY_PALETTE[i % len(PRIMARY_PALETTE)]}18,rgba(13,27,42,0.9));
                    border:1px solid {PRIMARY_PALETTE[i % len(PRIMARY_PALETTE)]}44;
                    border-radius:14px; padding:18px; margin-bottom:8px;">
                    <div style="font-size:1.8rem;text-align:center;">{icon}</div>
                    <div style="color:#E8F0FE;font-weight:700;text-align:center;margin:8px 0 4px;font-size:0.9rem;">
                        {label[2:]}
                    </div>
                    <div style="color:{PRIMARY_PALETTE[i % len(PRIMARY_PALETTE)]};text-align:center;font-size:0.85rem;">
                        {size:,} customers ({pct}%)
                    </div>
                    <hr style="border-color:rgba(255,255,255,0.08);margin:10px 0;">
                    {metrics_html}
                </div>""", unsafe_allow_html=True)
    else:
        st.info("Run Customer Segmentation first to see segment-level insights.")

    # ── Growth Suggestions ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🚀 Customer Growth Suggestions")
    growth_items = [
        ("🌱", "New Customer Acquisition", "high",
         "Increase digital ad spend by 20% focusing on demographics aged 25-35 — highest LTV potential."),
        ("🔄", "Win-Back Campaigns", "medium",
         "Re-target churned customers within 90 days with a compelling offer (e.g., 30% off next purchase)."),
        ("⬆️", "Upselling to Premium", "high",
         "Identify 'Regular' segment customers near Premium threshold and offer upgrade incentives."),
        ("📊", "Data-Driven Pricing", "medium",
         "Implement dynamic pricing models using spending score data to maximize revenue per customer."),
        ("🌐", "Geographic Expansion", "low",
         "Analyze underperforming cities and launch localized campaigns with geo-specific offers."),
        ("🤖", "AI-Powered Personalization", "high",
         "Deploy recommendation engines using purchase history to increase average order value by 15-25%."),
    ]

    priority_colors = {"high": ("#E53935", "🔴 HIGH"), "medium": ("#FB8C00", "🟠 MEDIUM"), "low": ("#43A047", "🟢 LOW")}
    for icon, title, priority, desc in growth_items:
        color, p_label = priority_colors[priority]
        st.markdown(f"""
        <div class="insight-card" style="display:flex;gap:16px;align-items:flex-start;">
            <div style="font-size:1.8rem;min-width:40px;text-align:center;">{icon}</div>
            <div style="flex:1;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <span style="color:#90CAF9;font-weight:700;font-size:0.95rem;">{title}</span>
                    <span style="color:{color};font-size:0.72rem;font-weight:600;
                        background:{color}22;border:1px solid {color}55;
                        border-radius:12px;padding:2px 10px;">{p_label}</span>
                </div>
                <div style="color:#78909C;font-size:0.85rem;line-height:1.5;">{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)
