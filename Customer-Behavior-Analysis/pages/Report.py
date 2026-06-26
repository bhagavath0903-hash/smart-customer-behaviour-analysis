"""
Report Download Page – Report.py
Generates a professional PDF report using fpdf2.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False


def _get_df():
    df = st.session_state.get("clean_df")
    if df is None:
        df = st.session_state.get("raw_df")
    return df


class CustomerReport(FPDF):
    def header(self):
        self.set_fill_color(13, 27, 42)
        self.rect(0, 0, 210, 20, "F")
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(144, 202, 249)
        self.cell(0, 14, "Smart Customer Analytics Platform", ln=True, align="C")
        self.set_text_color(150, 150, 150)
        self.set_font("Helvetica", "", 7)
        self.cell(0, 5, f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}", ln=True, align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, f"Page {self.page_no()} | Smart Customer Analytics Platform",
                  align="C")

    def section_title(self, title: str, color=(30, 136, 229)):
        self.set_fill_color(*color)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 12)
        self.rect(10, self.get_y(), 190, 8, "F")
        self.set_xy(12, self.get_y() + 1)
        self.cell(186, 6, title, ln=True)
        self.set_text_color(30, 30, 30)
        self.ln(3)

    def kv_row(self, key: str, value: str):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(80, 80, 80)
        self.cell(70, 6, key + ":", ln=False)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 30, 30)
        self.cell(120, 6, str(value), ln=True)

    def bullet(self, text: str):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(50, 50, 50)
        self.cell(8, 6, chr(149), ln=False)
        self.multi_cell(182, 5, text)
        self.ln(1)


def generate_pdf(df: pd.DataFrame) -> bytes:
    pdf = CustomerReport(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(10, 25, 10)

    # ── Cover ─────────────────────────────────────────────────────────────────
    pdf.set_fill_color(13, 27, 42)
    pdf.rect(0, 0, 210, 297, "F")

    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(232, 240, 254)
    pdf.set_y(80)
    pdf.cell(0, 20, "Smart Customer", ln=True, align="C")
    pdf.cell(0, 20, "Analytics Report", ln=True, align="C")

    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(144, 164, 174)
    pdf.cell(0, 12, "Behavior Analysis & Churn Prediction", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(144, 202, 249)
    pdf.cell(0, 8, datetime.now().strftime("%d %B %Y"), ln=True, align="C")

    # ── Page 2: Dataset Summary ───────────────────────────────────────────────
    pdf.add_page()
    pdf.set_fill_color(255, 255, 255)
    pdf.rect(0, 0, 210, 297, "F")

    pdf.section_title("1. Dataset Summary")
    pdf.kv_row("Total Records", f"{len(df):,}")
    pdf.kv_row("Total Features", str(len(df.columns)))
    pdf.kv_row("Missing Values", str(int(df.isnull().sum().sum())))
    pdf.kv_row("Duplicate Rows", str(int(df.duplicated().sum())))
    pdf.kv_row("Numeric Columns", str(len(df.select_dtypes(include=np.number).columns)))
    pdf.kv_row("Categorical Columns", str(len(df.select_dtypes(include="object").columns)))
    pdf.ln(5)

    pdf.section_title("2. Column Overview")
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(200, 220, 255)
    pdf.cell(60, 6, "Column", border=1, fill=True)
    pdf.cell(35, 6, "Type", border=1, fill=True)
    pdf.cell(30, 6, "Non-Null", border=1, fill=True)
    pdf.cell(30, 6, "Null", border=1, fill=True)
    pdf.cell(35, 6, "Unique", border=1, fill=True, ln=True)

    pdf.set_font("Helvetica", "", 8)
    pdf.set_fill_color(240, 240, 240)
    for i, col in enumerate(df.columns):
        fill = i % 2 == 0
        pdf.set_fill_color(245, 248, 255) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.cell(60, 5, col[:28], border=1, fill=fill)
        pdf.cell(35, 5, str(df[col].dtype)[:14], border=1, fill=fill)
        pdf.cell(30, 5, str(df[col].notnull().sum()), border=1, fill=fill)
        pdf.cell(30, 5, str(df[col].isnull().sum()), border=1, fill=fill)
        pdf.cell(35, 5, str(df[col].nunique()), border=1, fill=fill, ln=True)
        if pdf.get_y() > 260:
            pdf.add_page()
            pdf.set_fill_color(255, 255, 255)
            pdf.rect(0, 0, 210, 297, "F")

    # ── Statistical Summary ───────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_fill_color(255, 255, 255)
    pdf.rect(0, 0, 210, 297, "F")
    pdf.section_title("3. Statistical Summary (Numeric Columns)")

    num_df = df.select_dtypes(include=np.number)
    if not num_df.empty:
        desc = num_df.describe().T[["mean", "std", "min", "50%", "max"]].round(2)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(200, 220, 255)
        headers = ["Column", "Mean", "Std", "Min", "Median", "Max"]
        widths = [50, 28, 28, 25, 25, 25]
        for h, w in zip(headers, widths):
            pdf.cell(w, 6, h, border=1, fill=True)
        pdf.ln()
        pdf.set_font("Helvetica", "", 8)
        for i, (col, row) in enumerate(desc.iterrows()):
            fill = i % 2 == 0
            pdf.set_fill_color(245, 248, 255) if fill else pdf.set_fill_color(255, 255, 255)
            pdf.cell(50, 5, str(col)[:22], border=1, fill=fill)
            for w, val in zip(widths[1:], row.values):
                pdf.cell(w, 5, str(val), border=1, fill=fill)
            pdf.ln()
            if pdf.get_y() > 260:
                pdf.add_page()
                pdf.set_fill_color(255, 255, 255)
                pdf.rect(0, 0, 210, 297, "F")

    # ── Churn Analysis ────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_fill_color(255, 255, 255)
    pdf.rect(0, 0, 210, 297, "F")
    pdf.section_title("4. Churn Analysis")
    churn_col = next((c for c in df.columns if c.lower() in ("churn", "churned")), None)
    if churn_col:
        churn_mask = df[churn_col].astype(str).str.lower().isin(["yes", "1"])
        churn_count = int(churn_mask.sum())
        active_count = len(df) - churn_count
        churn_rate = round(churn_count / len(df) * 100, 2)
        pdf.kv_row("Total Customers", str(len(df)))
        pdf.kv_row("Active Customers", str(active_count))
        pdf.kv_row("Churned Customers", str(churn_count))
        pdf.kv_row("Churn Rate", f"{churn_rate}%")
        pdf.kv_row("Retention Rate", f"{100 - churn_rate}%")
    else:
        pdf.bullet("No churn column detected in the dataset.")
    pdf.ln(5)

    # ── ML Results ────────────────────────────────────────────────────────────
    pdf.section_title("5. Machine Learning Results")
    results = st.session_state.get("churn_results")
    best_name = st.session_state.get("best_model_name")
    if results:
        pdf.kv_row("Best Model", best_name)
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(200, 220, 255)
        for h, w in zip(["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"],
                        [55, 28, 28, 25, 25, 25]):
            pdf.cell(w, 6, h, border=1, fill=True)
        pdf.ln()
        pdf.set_font("Helvetica", "", 8)
        for i, (name, res) in enumerate(results.items()):
            fill = i % 2 == 0
            pdf.set_fill_color(245, 248, 255) if fill else pdf.set_fill_color(255, 255, 255)
            pdf.cell(55, 5, name[:24], border=1, fill=fill)
            for w, v in zip([28, 28, 25, 25, 25],
                            [res["accuracy"], res["precision"], res["recall"], res["f1"], res["auc_score"]]):
                pdf.cell(w, 5, str(v), border=1, fill=fill)
            pdf.ln()
    else:
        pdf.bullet("Run Churn Prediction models to include ML results in this report.")

    # ── Business Insights ─────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_fill_color(255, 255, 255)
    pdf.rect(0, 0, 210, 297, "F")
    pdf.section_title("6. Business Insights & Recommendations")

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 80, 160)
    pdf.cell(0, 8, "Retention Strategies:", ln=True)
    strategies = [
        "Launch personalized email campaigns targeting high-risk customer segments.",
        "Introduce a tiered loyalty rewards program for long-term retention.",
        "Implement proactive customer support outreach for customers with 3+ complaints.",
        "Offer 15-20% discounts to convert month-to-month customers to annual contracts.",
        "Deploy AI-powered product recommendation engines to boost average order value.",
    ]
    for s in strategies:
        pdf.bullet(s)

    pdf.ln(4)
    pdf.set_text_color(30, 80, 160)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 8, "Growth Suggestions:", ln=True)
    growth = [
        "Increase digital advertising targeting the dominant age group (25-35 years).",
        "Launch win-back campaigns within 90 days of customer churn.",
        "Invest in mobile app personalization for higher engagement rates.",
        "Expand product bundles in the top-performing purchasing categories.",
        "Implement dynamic pricing strategies based on customer spending scores.",
    ]
    for g in growth:
        pdf.bullet(g)

    return bytes(pdf.output())


def show():
    st.markdown("## 📄 Download Report")

    df = _get_df()
    if df is None:
        st.warning("⚠️ No dataset loaded. Please upload data first.")
        return

    st.markdown("""
    <div style="background:rgba(30,136,229,0.1);border:1px solid rgba(30,136,229,0.2);
         border-radius:12px;padding:20px 24px;margin-bottom:20px;">
        <h4 style="color:#90CAF9;margin:0 0 10px;">📋 Report Contents</h4>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
            <div style="color:#B0BEC5;font-size:0.85rem;">✅ Executive Summary</div>
            <div style="color:#B0BEC5;font-size:0.85rem;">✅ Dataset Statistics</div>
            <div style="color:#B0BEC5;font-size:0.85rem;">✅ Column Overview</div>
            <div style="color:#B0BEC5;font-size:0.85rem;">✅ Statistical Summary</div>
            <div style="color:#B0BEC5;font-size:0.85rem;">✅ Churn Analysis</div>
            <div style="color:#B0BEC5;font-size:0.85rem;">✅ ML Model Results</div>
            <div style="color:#B0BEC5;font-size:0.85rem;">✅ Business Insights</div>
            <div style="color:#B0BEC5;font-size:0.85rem;">✅ Recommendations</div>
        </div>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # ── PDF Download ──────────────────────────────────────────────────────────
    with col1:
        st.markdown("#### 📕 PDF Report")
        if FPDF_AVAILABLE:
            if st.button("🔄 Generate PDF Report", use_container_width=True):
                with st.spinner("Generating PDF…"):
                    try:
                        pdf_bytes = generate_pdf(df)
                        st.download_button(
                            label="⬇️ Download PDF",
                            data=pdf_bytes,
                            file_name=f"customer_analytics_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                        st.success("✅ PDF ready!")
                    except Exception as e:
                        st.error(f"PDF generation error: {e}")
        else:
            st.error("Install fpdf2: `pip install fpdf2`")

    # ── CSV Download ──────────────────────────────────────────────────────────
    with col2:
        st.markdown("#### 📗 CSV Export")
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Dataset CSV",
            data=csv_data,
            file_name=f"customer_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if st.session_state.get("segmented_df") is not None:
            seg_csv = st.session_state["segmented_df"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Segmented Data CSV",
                data=seg_csv,
                file_name="segmented_customers.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ── Excel Download ────────────────────────────────────────────────────────
    with col3:
        st.markdown("#### 📘 Excel Export")
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Raw Data", index=False)
            df.describe(include="all").T.round(3).to_excel(writer, sheet_name="Statistics")

            if st.session_state.get("segmented_df") is not None:
                st.session_state["segmented_df"].to_excel(
                    writer, sheet_name="Segmented Data", index=False
                )

            if st.session_state.get("churn_results"):
                results = st.session_state["churn_results"]
                ml_rows = [
                    {"Model": name, "Accuracy": r["accuracy"], "Precision": r["precision"],
                     "Recall": r["recall"], "F1": r["f1"], "AUC": r["auc_score"]}
                    for name, r in results.items()
                ]
                pd.DataFrame(ml_rows).to_excel(writer, sheet_name="ML Results", index=False)

        st.download_button(
            "⬇️ Download Excel Workbook",
            data=buf.getvalue(),
            file_name=f"customer_analytics_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    # ── Preview section ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Report Preview")
    with st.expander("Dataset Summary", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows", f"{len(df):,}")
        c2.metric("Total Columns", len(df.columns))
        c3.metric("Missing Values", int(df.isnull().sum().sum()))
        c4.metric("Duplicates", int(df.duplicated().sum()))
        st.dataframe(df.describe(include="all").T.round(3), use_container_width=True)

    if st.session_state.get("churn_results"):
        with st.expander("ML Results Preview"):
            results = st.session_state["churn_results"]
            ml_df = pd.DataFrame([
                {"Model": n, "Accuracy": r["accuracy"], "Precision": r["precision"],
                 "Recall": r["recall"], "F1": r["f1"], "AUC": r["auc_score"]}
                for n, r in results.items()
            ])
            st.dataframe(ml_df, use_container_width=True)
