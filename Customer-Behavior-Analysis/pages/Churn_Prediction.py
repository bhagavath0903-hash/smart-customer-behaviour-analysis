"""
Churn Prediction Page – Churn_Prediction.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.preprocessing import prepare_churn_features
from utils.prediction import (
    train_all_models, select_best_model, save_model, predict_single,
    get_feature_importances,
)
from utils.visualization import (
    plot_confusion_matrix, plot_roc_curve, plot_feature_importance,
    PRIMARY_PALETTE, _base_layout,
)

THEME = "plotly_dark"
BG = "rgba(20,26,46,0.95)"


def _get_df():
    df = st.session_state.get("clean_df")
    if df is None:
        df = st.session_state.get("raw_df")
    return df


def _guard():
    df = _get_df()
    if df is None:
        st.warning("⚠️ Please upload and clean a dataset first.")
        if st.button("Go to Upload"):
            st.session_state["current_page"] = "📂 Upload Dataset"
            st.rerun()
        return False
    return True


def show():
    st.markdown("## 🤖 Churn Prediction")
    if not _guard():
        return

    df = _get_df()

    # ── Check for churn column ────────────────────────────────────────────────
    churn_col_found = any(c.lower() in ("churn", "churned", "is_churn", "customer_churn")
                          for c in df.columns)
    if not churn_col_found:
        st.error("❌ No 'Churn' column detected in the dataset. "
                 "Please ensure your data has a 'Churn' column (Yes/No or 1/0).")
        return

    # ── Train button ──────────────────────────────────────────────────────────
    st.markdown("### 🏋️ Model Training")
    col_inf, col_btn = st.columns([3, 1])
    with col_inf:
        st.markdown("""
        <div style="background:rgba(30,136,229,0.1);border:1px solid rgba(30,136,229,0.2);
             border-radius:10px;padding:14px 20px;">
            Trains four models simultaneously: <strong>Logistic Regression</strong>,
            <strong>Decision Tree</strong>, <strong>Random Forest</strong>, and
            <strong>Gradient Boosting</strong>. The best-performing model is selected
            automatically by F1 Score.
        </div>""", unsafe_allow_html=True)
    with col_btn:
        train_btn = st.button("🚀  Train All Models", use_container_width=True)

    results_ready = st.session_state.get("churn_results") is not None

    if train_btn:
        with st.spinner("Preparing features…"):
            try:
                X, y, feat_names = prepare_churn_features(df)
            except ValueError as e:
                st.error(str(e))
                return

        with st.spinner(f"Training on {len(X):,} samples…"):
            results = train_all_models(X, y)
            best_name, best_result = select_best_model(results)
            save_model(best_result["model"], best_result["scaler"])

        st.session_state["churn_results"] = results
        st.session_state["best_model_name"] = best_name
        st.session_state["best_model"] = best_result["model"]
        st.session_state["best_scaler"] = best_result["scaler"]
        st.session_state["feature_names"] = feat_names
        results_ready = True
        st.success(f"✅ Training complete! Best Model: **{best_name}** (F1 = {best_result['f1']:.4f})")
        st.balloons()

    if not results_ready:
        st.markdown("""
        <div style="text-align:center;padding:50px;">
            <div style="font-size:3rem;">🤖</div>
            <p style="color:#546E7A;">Click <strong>Train All Models</strong> to begin.</p>
        </div>""", unsafe_allow_html=True)
        return

    results = st.session_state["churn_results"]
    best_name = st.session_state["best_model_name"]

    # ── Model Comparison ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Model Comparison")
    metrics_df = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": res["accuracy"],
            "Precision": res["precision"],
            "Recall": res["recall"],
            "F1 Score": res["f1"],
            "AUC": res["auc_score"],
        }
        for name, res in results.items()
    ])
    metrics_df["Best"] = metrics_df["Model"] == best_name

    m_cols = st.columns(len(results))
    for i, (name, res) in enumerate(results.items()):
        with m_cols[i]:
            is_best = name == best_name
            border = "rgba(30,136,229,0.6)" if is_best else "rgba(30,136,229,0.15)"
            badge = "🏆 BEST" if is_best else ""
            st.markdown(f"""
            <div style="
                background:linear-gradient(135deg,rgba(30,58,95,0.85),rgba(13,27,42,0.9));
                border:1px solid {border}; border-radius:14px; padding:18px 14px;
                text-align:center; margin-bottom:4px;">
                <div style="font-size:0.75rem;color:#90A4AE;font-weight:600;letter-spacing:.5px;">
                    {badge}
                </div>
                <div style="font-size:1.0rem;font-weight:700;color:#90CAF9;margin:6px 0;">
                    {name}
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px;">
                    <div>
                        <div style="color:#78909C;font-size:0.7rem;">Accuracy</div>
                        <div style="color:#E8F0FE;font-weight:600;">{res['accuracy']:.3f}</div>
                    </div>
                    <div>
                        <div style="color:#78909C;font-size:0.7rem;">F1 Score</div>
                        <div style="color:#43A047;font-weight:600;">{res['f1']:.3f}</div>
                    </div>
                    <div>
                        <div style="color:#78909C;font-size:0.7rem;">Precision</div>
                        <div style="color:#E8F0FE;font-weight:600;">{res['precision']:.3f}</div>
                    </div>
                    <div>
                        <div style="color:#78909C;font-size:0.7rem;">Recall</div>
                        <div style="color:#E8F0FE;font-weight:600;">{res['recall']:.3f}</div>
                    </div>
                </div>
                <div style="margin-top:10px;color:#1E88E5;font-size:0.8rem;">AUC: {res['auc_score']:.3f}</div>
            </div>""", unsafe_allow_html=True)

    # ── Detailed results per model ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Detailed Model Analysis")
    model_sel = st.selectbox("Inspect Model", list(results.keys()), index=list(results.keys()).index(best_name))
    res = results[model_sel]

    tab1, tab2, tab3 = st.tabs(["📉 Confusion Matrix & ROC", "📊 Feature Importance", "📋 Classification Report"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_confusion_matrix(res["cm"]), use_container_width=True)
        with c2:
            st.plotly_chart(plot_roc_curve(res["fpr"], res["tpr"], res["auc_score"]),
                            use_container_width=True)

    with tab2:
        feat_names = st.session_state.get("feature_names", [])
        fi_df = get_feature_importances(res["model"], feat_names)
        if not fi_df.empty:
            st.plotly_chart(
                plot_feature_importance(fi_df["Feature"].tolist(), fi_df["Importance"].tolist()),
                use_container_width=True,
            )
            st.dataframe(fi_df, use_container_width=True)
        else:
            st.info("Feature importances not available for this model type.")

    with tab3:
        report = res.get("report", {})
        report_df = pd.DataFrame(report).T.drop(index=["accuracy"], errors="ignore").round(3)
        st.dataframe(report_df, use_container_width=True)

    # ── Manual Prediction ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Predict Churn for a New Customer")
    feat_names = st.session_state.get("feature_names", [])
    best_model = st.session_state.get("best_model")
    best_scaler = st.session_state.get("best_scaler")

    if not feat_names or best_model is None:
        st.info("Train models first to enable manual prediction.")
        return

    df_num = _get_df().select_dtypes(include=np.number)

    st.markdown(f"<p style='color:#78909C;'>Model used: <strong>{best_name}</strong></p>",
                unsafe_allow_html=True)

    input_vals = {}
    cols_inp = st.columns(min(4, len(feat_names)))
    for i, feat in enumerate(feat_names):
        with cols_inp[i % 4]:
            col_data = df_num[feat] if feat in df_num.columns else pd.Series([0])
            val = st.number_input(
                feat,
                value=float(col_data.median()),
                min_value=float(col_data.min()),
                max_value=float(col_data.max()),
                step=float(max(col_data.std() / 10, 0.01)),
                key=f"inp_{feat}",
            )
            input_vals[feat] = val

    pred_btn = st.button("🔮  Predict Churn", use_container_width=False)
    if pred_btn:
        feature_vals = [input_vals[f] for f in feat_names]
        label, prob = predict_single(best_model, best_scaler, feature_vals)
        churn_pct = prob * 100
        stay_pct = (1 - prob) * 100

        col_l, col_r = st.columns(2)
        with col_l:
            color = "#E53935" if "Churn" in label else "#43A047"
            st.markdown(f"""
            <div style="
                background:linear-gradient(135deg,{color}22,rgba(13,27,42,0.9));
                border:2px solid {color}66; border-radius:18px;
                padding:32px; text-align:center; margin-top:10px;">
                <div style="font-size:2.5rem; margin-bottom:10px;">{label.split()[0]}</div>
                <div style="font-size:1.8rem; font-weight:800; color:{color};">{label}</div>
                <div style="font-size:1rem; color:#90A4AE; margin-top:8px;">
                    Churn Probability: <strong style="color:{color};">{churn_pct:.1f}%</strong>
                </div>
            </div>""", unsafe_allow_html=True)

        with col_r:
            fig_prob = px.pie(
                values=[churn_pct, stay_pct],
                names=["Churn Risk", "Retention"],
                color_discrete_sequence=["#E53935", "#43A047"],
                hole=0.55,
                template=THEME,
            )
            fig_prob.update_layout(paper_bgcolor=BG, font=dict(color="#B0BEC5"), height=280)
            st.plotly_chart(fig_prob, use_container_width=True)

    # ── Download predictions ──────────────────────────────────────────────────
    st.markdown("---")
    if st.button("⬇️ Generate Predictions for Entire Dataset"):
        try:
            X_all, y_all, fnames = prepare_churn_features(_get_df())
            X_all_s = best_scaler.transform(X_all.fillna(X_all.median()))
            preds = best_model.predict(X_all_s)
            probs = best_model.predict_proba(X_all_s)[:, 1]
            pred_df = _get_df().copy()
            pred_df["Predicted_Churn"] = ["Yes" if p == 1 else "No" for p in preds]
            pred_df["Churn_Probability"] = np.round(probs, 4)
            csv_pred = pred_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Predictions CSV",
                data=csv_pred,
                file_name="churn_predictions.csv",
                mime="text/csv",
            )
            st.dataframe(pred_df[["Predicted_Churn", "Churn_Probability"]].head(20),
                         use_container_width=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")
