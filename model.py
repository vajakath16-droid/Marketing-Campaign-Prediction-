
import streamlit as st
import joblib
import pandas as pd
import os

st.title("🚀 Marketing Campaign Prediction Dashboard")

# Load models safely using paths relative to the zip structure
@st.cache_resource
def load_models():
    reg_model = joblib.load("models/revenue_regression_model.pkl")
    reg_cols = joblib.load("models/regression_feature_columns.pkl")
    return reg_model, reg_cols

# Check if files exist before running to avoid errors
if os.path.exists("models/revenue_regression_model.pkl"):
    reg_model, reg_cols = load_models()
    features_df = pd.read_csv("data/marketing_campaign_features.csv")

    if st.button("🎲 Pick a Random Campaign & Predict"):
        sample = features_df.sample(1)
        st.subheader("📊 Selected Campaign Data:")
        st.dataframe(sample[reg_cols])
        
        X_reg_sample = sample[reg_cols]
        predicted_revenue = reg_model.predict(X_reg_sample)
        st.success(f"💰 **Predicted Revenue:** ${predicted_revenue[0]:,.2f}")
else:
    st.error("Model files not found! Make sure the 'models' folder is next to app.py")


if os.path.exists("models/revenue_regression_model.pkl"):

    st.markdown("---")
    st.subheader("📊 Data Visualizations")
        
        # 1. Scatter Plot: Impressions vs Clicks
    st.write("📈 **Impressions vs Clicks Relation**")
    st.scatter_chart(data=features_df, x="Impressions", y="Clicks", color="#ff4b4b")
        
        # 2. Bar Chart (ஸ்மார்ட் கோடு)
    channel_col = [col for col in features_df.columns if 'channel' in col.lower()]
    
    if channel_col:
        col_name = channel_col[0]
        st.write(f"📱 **Marketing Channels Breakdown ({col_name})**")
        st.bar_chart(features_df[col_name].value_counts())
    else:
        st.write("📱 **Campaign Conversions Distribution**")
        st.bar_chart(features_df["Conversions"].value_counts().head(10))

    # 3. Line Chart: Cost vs Engagement
    if "Acquisition_Cost" in features_df.columns and "Engagement_Score" in features_df.columns:
        st.write("📈 **Acquisition Cost vs Engagement Score Trend**")
        chart_data = features_df[["Engagement_Score", "Acquisition_Cost"]].head(50)
        st.line_chart(data=chart_data, x="Engagement_Score", y="Acquisition_Cost")