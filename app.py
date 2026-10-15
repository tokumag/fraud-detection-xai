"""
Streamlit app for interactive fraud detection with explainability.

Provides:
- Real-time transaction scoring
- SHAP-based explanations for each prediction
- Global model performance dashboard
- Feature importance visualization
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

from src.data_loader import FEATURE_NAMES, TARGET_NAME
from src.preprocessor import Preprocessor
from src.model import load_model
from src.explainer import FraudExplainer


# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection XAI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────
MODEL_PATH = "models/fraud_model.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"
METRICS_PATH = "models/metrics.json"


@st.cache_resource
def load_artifacts():
    """Load model, preprocessor, and metrics."""
    model = load_model(MODEL_PATH)
    preprocessor = Preprocessor.load(PREPROCESSOR_PATH)

    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    return model, preprocessor, metrics


@st.cache_resource
def load_explainer(_model, _preprocessor):
    """Initialize SHAP explainer with background data."""
    # Generate a small background dataset
    from src.data_loader import generate_synthetic_data
    bg_data = generate_synthetic_data(n_samples=1000, fraud_ratio=0.02, random_seed=99)
    X_bg = _preprocessor.transform(bg_data)
    return FraudExplainer(_model, X_bg, max_samples=500)


def render_metric_card(label: str, value: str, delta: str = ""):
    """Render a styled metric card."""
    st.metric(label=label, value=value, delta=delta)


def render_prediction_gauge(probability: float) -> go.Figure:
    """Create a gauge chart for fraud probability."""
    color = "green" if probability < 0.3 else "orange" if probability < 0.7 else "red"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={"text": "Fraud Probability (%)"},
        number={"suffix": "%", "font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30], "color": "#e8f5e9"},
                {"range": [30, 70], "color": "#fff3e0"},
                {"range": [70, 100], "color": "#ffebee"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    fig.update_layout(height=300, margin=dict(t=50, b=0, l=30, r=30))
    return fig


def render_shap_waterfall(contributions: dict, top_k: int = 10) -> go.Figure:
    """Create a horizontal bar chart of SHAP contributions."""
    items = list(contributions.items())[:top_k]
    features = [x[0].replace("_", " ").title() for x in items]
    values = [x[1] for x in items]
    colors = ["#ef5350" if v > 0 else "#66bb6a" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Feature Contributions (SHAP Values)",
        xaxis_title="Impact on Fraud Prediction →",
        yaxis=dict(autorange="reversed"),
        height=400,
        margin=dict(l=10, r=10, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Main App ──────────────────────────────────────────────────────
def main():
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        st.error("⚠️ No trained model found. Run `python train.py` first.")
        st.code("python train.py", language="bash")
        return

    model, preprocessor, metrics = load_artifacts()
    explainer = load_explainer(model, preprocessor)

    # ── Sidebar ───────────────────────────────────────────────────
    st.sidebar.title("🔍 Fraud Detection XAI")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Transaction Analyzer", "📊 Model Dashboard"],
    )

    if page == "🏠 Transaction Analyzer":
        render_analyzer(model, preprocessor, explainer)
    else:
        render_dashboard(metrics, model, preprocessor, explainer)


def render_analyzer(model, preprocessor, explainer):
    """Render the transaction analysis page."""
    st.title("🔍 Transaction Fraud Analyzer")
    st.markdown("Enter transaction details below to get a fraud prediction with AI explanations.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("💳 Transaction Details")
        amount = st.number_input("Transaction Amount ($)", 1.0, 50000.0, 150.0, step=10.0)
        merchant = st.slider("Merchant Category", 0, 14, 5)
        hour = st.slider("Transaction Hour", 0, 23, 14)
        day = st.selectbox("Day of Week", list(range(7)),
                          format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        online = st.checkbox("Online Order", value=False)
        international = st.checkbox("International", value=False)

    with col2:
        st.subheader("📍 Location & Security")
        dist_home = st.number_input("Distance from Home (km)", 0.0, 1000.0, 15.0)
        dist_last = st.number_input("Distance from Last Transaction (km)", 0.0, 500.0, 5.0)
        ratio_median = st.number_input("Ratio to Median Purchase", 0.1, 20.0, 1.0, step=0.1)
        repeat = st.checkbox("Repeat Retailer", value=True)
        chip = st.checkbox("Used Chip", value=True)
        pin = st.checkbox("Used PIN", value=True)

    with col3:
        st.subheader("📈 Account History")
        acct_age = st.number_input("Account Age (days)", 1, 5000, 800)
        txn_24h = st.number_input("Transactions Last 24h", 0, 50, 3)
        txn_7d = st.number_input("Transactions Last 7 Days", 0, 200, 15)
        avg_7d = st.number_input("Avg Amount Last 7 Days ($)", 1.0, 10000.0, 50.0)
        std_7d = st.number_input("Std Dev Amount Last 7 Days", 0.0, 5000.0, 20.0)
        max_30d = st.number_input("Max Amount Last 30 Days ($)", 1.0, 50000.0, 150.0)
        time_since = st.number_input("Hours Since Last Transaction", 0.0, 500.0, 12.0)

    is_weekend = 1 if day >= 5 else 0

    # Build feature vector
    features = pd.DataFrame([{
        "transaction_amount": amount,
        "merchant_category": merchant,
        "transaction_hour": hour,
        "transaction_day_of_week": day,
        "distance_from_home": dist_home,
        "distance_from_last_transaction": dist_last,
        "ratio_to_median_purchase": ratio_median,
        "repeat_retailer": int(repeat),
        "used_chip": int(chip),
        "used_pin": int(pin),
        "online_order": int(online),
        "account_age_days": acct_age,
        "num_transactions_last_24h": txn_24h,
        "num_transactions_last_7d": txn_7d,
        "avg_amount_last_7d": avg_7d,
        "std_amount_last_7d": std_7d,
        "max_amount_last_30d": max_30d,
        "time_since_last_transaction": time_since,
        "is_weekend": is_weekend,
        "is_international": int(international),
    }])

    st.markdown("---")

    if st.button("🔎 Analyze Transaction", type="primary", use_container_width=True):
        X_scaled = preprocessor.transform(features)
        prob = model.predict_proba(X_scaled)[0][1]
        pred = "🚨 FRAUD" if prob >= 0.5 else "✅ LEGITIMATE"

        # Results
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.plotly_chart(render_prediction_gauge(prob), use_container_width=True)
            if prob >= 0.5:
                st.error(f"**Verdict: {pred}** — This transaction has a {prob:.1%} probability of being fraudulent.")
            else:
                st.success(f"**Verdict: {pred}** — This transaction appears legitimate ({prob:.1%} fraud probability).")

        with res_col2:
            contributions = explainer.explain_single(X_scaled)
            st.plotly_chart(render_shap_waterfall(contributions), use_container_width=True)

        # Detailed explanation
        with st.expander("📋 Detailed Feature Contributions"):
            contrib_df = pd.DataFrame([
                {"Feature": k.replace("_", " ").title(), "SHAP Value": f"{v:+.4f}", "Direction": "↑ Fraud" if v > 0 else "↓ Legitimate"}
                for k, v in contributions.items()
            ])
            st.dataframe(contrib_df, use_container_width=True, hide_index=True)


def render_dashboard(metrics, model, preprocessor, explainer):
    """Render the model performance dashboard."""
    st.title("📊 Model Performance Dashboard")
    st.markdown("---")

    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    with col2:
        st.metric("PR-AUC", f"{metrics['pr_auc']:.4f}")
    with col3:
        st.metric("Precision", f"{metrics['precision']:.4f}")
    with col4:
        st.metric("Recall", f"{metrics['recall']:.4f}")
    with col5:
        st.metric("F1 Score", f"{metrics['f1_score']:.4f}")

    st.markdown("---")

    # Confusion matrix
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Confusion Matrix")
        cm = metrics["confusion_matrix"]
        cm_data = np.array([
            [cm["true_negatives"], cm["false_positives"]],
            [cm["false_negatives"], cm["true_positives"]],
        ])
        fig = px.imshow(
            cm_data,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Legitimate", "Fraud"],
            y=["Legitimate", "Fraud"],
            text_auto=True,
            color_continuous_scale="RdYlGn_r",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Global Feature Importance (SHAP)")
        from src.data_loader import generate_synthetic_data
        sample_data = generate_synthetic_data(n_samples=1000, fraud_ratio=0.02, random_seed=99)
        X_sample = preprocessor.transform(sample_data)
        importance_df = explainer.global_feature_importance(X_sample, top_k=15)

        fig = px.bar(
            importance_df,
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            title="Mean |SHAP| Value",
            color="mean_abs_shap",
            color_continuous_scale="Reds",
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            height=400,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Additional metrics
    st.markdown("---")
    st.subheader("📋 Detailed Metrics")
    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        st.json({
            "Accuracy": metrics["accuracy"],
            "Specificity": metrics["specificity"],
            "False Positive Rate": metrics["false_positive_rate"],
        })
    with detail_col2:
        st.markdown("""
        **Why these metrics matter for fraud detection:**
        - **PR-AUC** is more informative than ROC-AUC for imbalanced datasets
        - **Recall** (sensitivity) measures how many actual frauds we catch
        - **Precision** tells us what fraction of flagged transactions are truly fraud
        - **Specificity** shows how well we avoid blocking legitimate transactions
        """)


if __name__ == "__main__":
    main()
