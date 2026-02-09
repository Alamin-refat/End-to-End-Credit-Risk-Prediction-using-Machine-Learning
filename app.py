# =====================================================
# Imports
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# =====================================================
# Page Config
# =====================================================
st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="🏦",
    layout="centered"
)

# =====================================================
# Load Model Artifacts
# =====================================================
MODEL_PATH = "model/credit_risk_model.pkl"
SCALER_PATH = "model/scaler.pkl"
FEATURE_PATH = "model/feature_columns.pkl"

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Please check model folder.")
        st.stop()

    model = joblib.load(MODEL_PATH)

    scaler = None
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)

    feature_columns = None
    if os.path.exists(FEATURE_PATH):
        feature_columns = joblib.load(FEATURE_PATH)

    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

# =====================================================
# Header
# =====================================================
st.title("🏦 Credit Risk Prediction Dashboard")
st.markdown("### Machine Learning Powered Loan Default Risk Analysis")
st.divider()

# =====================================================
# Input Section
# =====================================================
st.subheader("📋 Customer Financial Profile")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 30)
    income = st.number_input("Annual Income", 0, 1_000_000, 50000)
    loan_amount = st.number_input("Loan Amount", 0, 500_000, 20000)
    loan_term = st.number_input("Loan Term (Months)", 1, 360, 36)

with col2:
    credit_score = st.number_input("Credit Score", 300, 900, 700)
    employment_length = st.number_input("Employment Length (Years)", 0, 50, 5)
    existing_loans = st.number_input("Number of Existing Loans", 0, 20, 1)

# =====================================================
# Data Preparation
# =====================================================
def prepare_input():
    data = {
        "age": age,
        "income": income,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "credit_score": credit_score,
        "employment_length": employment_length,
        "existing_loans": existing_loans
    }

    df = pd.DataFrame([data])

    if feature_columns is not None:
        df = df.reindex(columns=feature_columns, fill_value=0)

    if scaler is not None:
        df = scaler.transform(df)

    return df

# =====================================================
# Prediction Section
# =====================================================
st.divider()

if st.button("🔍 Analyze Credit Risk", use_container_width=True):

    input_df = prepare_input()

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Risk Assessment Result")

    # ==========================
    # Risk Status
    # ==========================
    if prediction == 1:
        st.error("⚠️ HIGH CREDIT RISK")
    else:
        st.success("✅ LOW CREDIT RISK")

    # ==========================
    # Probability Progress Bar
    # ==========================
    st.write("### Default Probability")
    st.progress(float(probability))
    st.metric("Probability of Default", f"{probability*100:.2f}%")

    # ==========================
    # Risk Gauge Style Indicator
    # ==========================
    st.write("### Risk Level Gauge")

    if probability < 0.3:
        st.success("🟢 Low Risk Zone")
    elif probability < 0.6:
        st.warning("🟡 Medium Risk Zone")
    else:
        st.error("🔴 High Risk Zone")

    # ==========================
    # Probability Visualization Chart
    # ==========================
    st.write("### Risk Probability Visualization")

    fig = plt.figure()
    plt.bar(["Default Risk", "Safe Probability"], [probability, 1-probability])
    plt.title("Credit Risk Probability Distribution")
    plt.ylabel("Probability")
    st.pyplot(fig)

# =====================================================
# Sidebar Info
# =====================================================
st.sidebar.title("ℹ️ Model Information")
st.sidebar.write("Model: Gradient Boosting Classifier")
st.sidebar.write("Business Use: Loan Default Prediction")
st.sidebar.write("Type: Binary Classification")

st.sidebar.divider()
st.sidebar.write("Production ML Portfolio Project")

# =====================================================
# Footer
# =====================================================
st.divider()
st.caption("End-to-End Machine Learning | Credit Risk Prediction | Deployment Ready")

