import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Page config & styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Marketing Campaign Prediction Dashboard",
    page_icon="🚀",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #ff4b4b, #ff8f4b, #ffd24b);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-title { color: #8a8f98; margin-top: 0; margin-bottom: 1.2rem; }
    .result-card {
        padding: 1.2rem 1.4rem; border-radius: 16px; color: white;
        box-shadow: 0 6px 18px rgba(0,0,0,0.25); text-align: center;
    }
    .rev-card   { background: linear-gradient(135deg, #1e3c72, #2a5298); }
    .profit-card{ background: linear-gradient(135deg, #11998e, #38ef7d); }
    .loss-card  { background: linear-gradient(135deg, #cb2d3e, #ef473a); }
    .result-card h4 { margin: 0; font-weight: 500; opacity: .9; }
    .result-card h1 { margin: .3rem 0 0 0; font-size: 2.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-title">🚀 Marketing Campaign Prediction Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Predict campaign Revenue (regression) and Profit / Loss (classification) from your campaign inputs.</p>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# File paths
# ----------------------------------------------------------------------------
REG_MODEL_PATH = "models/revenue_regression_model.pkl"
REG_COLS_PATH = "models/regression_feature_columns.pkl"
CLF_MODEL_PATH = "models/profit_classification_model.pkl"
CLF_COLS_PATH = "models/classification_feature_columns.pkl"
DATA_PATH = "data/marketing_campaign_features.csv"


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    reg_model = joblib.load(REG_MODEL_PATH)
    reg_cols = joblib.load(REG_COLS_PATH)
    clf_model = joblib.load(CLF_MODEL_PATH) if os.path.exists(CLF_MODEL_PATH) else None
    clf_cols = joblib.load(CLF_COLS_PATH) if os.path.exists(CLF_COLS_PATH) else reg_cols
    return reg_model, list(reg_cols), clf_model, list(clf_cols)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


if not (os.path.exists(REG_MODEL_PATH) and os.path.exists(DATA_PATH)):
    st.error("Model/data files not found! Make sure the 'models' and 'data' folders are next to app.py")
    st.stop()

reg_model, reg_cols, clf_model, clf_cols = load_models()
features_df = load_data()
all_cols = list(dict.fromkeys(reg_cols + clf_cols))


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def build_input_row(user_values: dict) -> pd.DataFrame:
    """Create a one-row DataFrame aligned to training feature columns.
    Numeric inputs come from the user, one-hot columns are set from
    the selected categories, everything else uses dataset median/mode."""
    row = {}
    for c in all_cols:
        if c in user_values:
            row[c] = user_values[c]
        elif pd.api.types.is_numeric_dtype(features_df[c]) if c in features_df else True:
            row[c] = features_df[c].median() if c in features_df else 0
        else:
            row[c] = features_df[c].mode()[0]

    # Engineered features (applied only if the model expects them)
    imp = row.get("Impressions", 0)
    clk = row.get("Clicks", 0)
    spend = row.get("Spend", row.get("Acquisition_Cost", 0))
    conv = row.get("Conversions", 0)
    derived = {
        "CTR": clk / imp if imp else 0,
        "Conversion_Rate": conv / clk if clk else 0,
        "CPC": spend / clk if clk else 0,
        "Cost_Per_Conversion": spend / conv if conv else 0,
        "Log_Impressions": np.log1p(imp),
        "Log_Clicks": np.log1p(clk),
    }
    for k, v in derived.items():
        if k in all_cols and k not in user_values:
            row[k] = v

    # One-hot encoding for selected categorical options
    for key, val in user_values.items():
        if key.startswith("__cat__"):
            prefix = key.replace("__cat__", "")
            for c in all_cols:
                if c.startswith(prefix + "_"):
                    row[c] = 1 if c == f"{prefix}_{val}" else 0

    return pd.DataFrame([row])[all_cols]


def detect_categories(prefix_hint: str):
    """Find one-hot columns containing a hint (e.g. 'channel') and return (prefix, options)."""
    cols = [c for c in all_cols if prefix_hint in c.lower() and "_" in c]
    if not cols:
        return None, []
    prefix = cols[0].split("_")[0] if cols[0].lower().startswith(prefix_hint) else cols[0].rsplit("_", 1)[0]
    options = [c[len(prefix) + 1:] for c in cols if c.startswith(prefix + "_")]
    return prefix, options


# ----------------------------------------------------------------------------
# Sidebar - Inputs
# ----------------------------------------------------------------------------
st.sidebar.title("🧭 Campaign Inputs")


def num_input(label, col, default, step=1.0, minv=0.0, maxv=None):
    if col in features_df.columns:
        minv = float(features_df[col].min())
        maxv = float(features_df[col].max())
        default = float(features_df[col].median())
    maxv = maxv if maxv is not None else default * 10 + 1000
    return st.sidebar.number_input(label, min_value=minv, max_value=maxv, value=default, step=step)


user_values = {}
if "Impressions" in all_cols:
    user_values["Impressions"] = num_input("👁️ Impressions", "Impressions", 10000.0, 100.0)
if "Clicks" in all_cols:
    user_values["Clicks"] = num_input("🖱️ Clicks", "Clicks", 500.0, 10.0)
if "Conversions" in all_cols:
    user_values["Conversions"] = num_input("✅ Conversions", "Conversions", 50.0, 1.0)
for spend_col in ["Spend", "Acquisition_Cost"]:
    if spend_col in all_cols:
        user_values[spend_col] = num_input(f"💸 {spend_col.replace('_', ' ')}", spend_col, 1000.0, 10.0)
if "Engagement_Score" in all_cols:
    user_values["Engagement_Score"] = num_input("⭐ Engagement Score", "Engagement_Score", 5.0, 0.1)

for hint, label in [("channel", "📱 Channel"), ("campaign_type", "🎯 Campaign Type"),
                    ("segment", "👥 Customer Segment"), ("location", "📍 Location")]:
    prefix, options = detect_categories(hint)
    if options:
        user_values[f"__cat__{prefix}"] = st.sidebar.selectbox(label, options)

predict_clicked = st.sidebar.button("🔮 Predict", use_container_width=True, type="primary")

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_pred, tab_viz, tab_data = st.tabs(["🔮 Prediction", "📊 Data Visualizations", "🗂️ Dataset"])

# ----------------------------- Prediction tab -------------------------------
with tab_pred:
    if predict_clicked:
        input_df = build_input_row(user_values)
        X_reg = input_df[reg_cols]
        revenue = float(reg_model.predict(X_reg)[0])

        spend_val = user_values.get("Spend", user_values.get("Acquisition_Cost", 0))

        if clf_model is not None:
            X_clf = input_df[clf_cols]
            pred_class = int(clf_model.predict(X_clf)[0])
            proba = clf_model.predict_proba(X_clf)[0] if hasattr(clf_model, "predict_proba") else None
            is_profit = pred_class == 1
            confidence = float(np.max(proba)) if proba is not None else None
        else:
            is_profit = revenue > spend_val
            confidence = None

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="result-card rev-card"><h4>💰 Predicted Revenue</h4><h1>${revenue:,.2f}</h1></div>', unsafe_allow_html=True)
        with c2:
            cls = "profit-card" if is_profit else "loss-card"
            txt = "📈 PROFIT" if is_profit else "📉 LOSS"
            st.markdown(f'<div class="result-card {cls}"><h4>Profit / Loss</h4><h1>{txt}</h1></div>', unsafe_allow_html=True)
        with c3:
            conf_txt = f"{confidence * 100:.1f}%" if confidence is not None else "N/A"
            st.markdown(f'<div class="result-card rev-card"><h4>🎯 Model Confidence</h4><h1>{conf_txt}</h1></div>', unsafe_allow_html=True)

        st.markdown("---")
        g1, g2 = st.columns(2)

        with g1:
            st.subheader("💵 Revenue vs Spend")
            fig = go.Figure(go.Bar(
                x=["Spend", "Predicted Revenue"],
                y=[spend_val, revenue],
                marker_color=["#ff4b4b", "#2a5298"],
                text=[f"${spend_val:,.0f}", f"${revenue:,.0f}"],
                textposition="outside",
            ))
            fig.update_layout(height=380, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with g2:
            st.subheader("🧩 Your Key Inputs")
            numeric_inputs = {k: v for k, v in user_values.items() if not k.startswith("__cat__")}
            if numeric_inputs and features_df is not None:
                # Compare each input to dataset average (as % of average)
                labels, pct = [], []
                for k, v in numeric_inputs.items():
                    if k in features_df.columns and features_df[k].mean() != 0:
                        labels.append(k)
                        pct.append(v / features_df[k].mean() * 100)
                fig2 = go.Figure(go.Scatterpolar(r=pct + pct[:1], theta=labels + labels[:1],
                                                 fill="toself", line_color="#ff8f4b"))
                fig2.update_layout(height=380, margin=dict(t=20, b=20),
                                   polar=dict(radialaxis=dict(visible=True)),
                                   showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)
                st.caption("Inputs shown as % of the dataset average (100% = average campaign).")

        if confidence is not None and proba is not None and len(proba) == 2:
            st.subheader("📊 Profit / Loss Probability")
            fig3 = px.pie(values=[proba[0], proba[1]], names=["Loss", "Profit"], hole=0.55,
                          color_discrete_sequence=["#ef473a", "#38ef7d"])
            fig3.update_layout(height=340, margin=dict(t=10, b=10))
            st.plotly_chart(fig3, use_container_width=True)

        with st.expander("🔍 View processed model input"):
            st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)
    else:
        st.info("👈 Enter campaign details in the sidebar and click **Predict**.")

    st.markdown("---")
    st.subheader("🎲 Try a Random Campaign")
    if st.button("🎲 Pick Random Campaign & Predict"):
        sample = features_df.sample(1)
        st.dataframe(sample[reg_cols], use_container_width=True)
        pred = reg_model.predict(sample[reg_cols])[0]
        st.success(f"💰 **Predicted Revenue:** ${pred:,.2f}")
        if clf_model is not None:
            label = clf_model.predict(sample[clf_cols])[0]
            st.success(f"📈 **Predicted Outcome:** {'Profit' if label == 1 else 'Loss'}")

# ----------------------------- Visualization tab ----------------------------
with tab_viz:
    st.subheader("📊 Data Visualizations")

    if {"Impressions", "Clicks"}.issubset(features_df.columns):
        st.write("📈 **Impressions vs Clicks Relation**")
        fig = px.scatter(features_df, x="Impressions", y="Clicks", opacity=0.6,
                         color_discrete_sequence=["#ff4b4b"], trendline="ols")
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        channel_col = [c for c in features_df.columns if "channel" in c.lower()]
        if channel_col:
            col_name = channel_col[0]
            st.write(f"📱 **Marketing Channels Breakdown ({col_name})**")
            counts = features_df[col_name].value_counts().reset_index()
            counts.columns = [col_name, "Count"]
            st.plotly_chart(px.bar(counts, x=col_name, y="Count", color="Count",
                                   color_continuous_scale="Sunset"), use_container_width=True)
        elif "Conversions" in features_df.columns:
            st.write("📱 **Campaign Conversions Distribution**")
            counts = features_df["Conversions"].value_counts().head(10).reset_index()
            counts.columns = ["Conversions", "Count"]
            st.plotly_chart(px.bar(counts, x="Conversions", y="Count", color="Count",
                                   color_continuous_scale="Sunset"), use_container_width=True)

    with right:
        if {"Acquisition_Cost", "Engagement_Score"}.issubset(features_df.columns):
            st.write("📈 **Acquisition Cost vs Engagement Score Trend**")
            chart_data = features_df[["Engagement_Score", "Acquisition_Cost"]].head(50).sort_values("Engagement_Score")
            st.plotly_chart(px.line(chart_data, x="Engagement_Score", y="Acquisition_Cost", markers=True,
                                    color_discrete_sequence=["#2a5298"]), use_container_width=True)

    numeric_df = features_df.select_dtypes(include=np.number)
    if numeric_df.shape[1] > 1:
        st.write("🔥 **Correlation Heatmap**")
        corr = numeric_df.corr().round(2)
        st.plotly_chart(px.imshow(corr, color_continuous_scale="RdBu_r", aspect="auto"),
                        use_container_width=True)

# ----------------------------- Dataset tab ----------------------------------
with tab_data:
    st.subheader("🗂️ Campaign Features Dataset")
    st.dataframe(features_df, use_container_width=True, height=450)
    st.caption(f"{features_df.shape[0]:,} rows × {features_df.shape[1]} columns")
    st.download_button("⬇️ Download CSV", features_df.to_csv(index=False).encode(),
                       "marketing_campaign_features.csv", "text/csv")
