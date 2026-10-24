import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="CreditWise | AI Credit Risk Assessment",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling (CSS)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 55%, #0F3758 100%);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
        color: #FFFFFF;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0 0 0.5rem 0;
        background: linear-gradient(90deg, #FFFFFF 0%, #93C5FD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 1.02rem;
        color: #94A3B8;
        margin: 0;
        line-height: 1.5;
        max-width: 760px;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(96, 165, 250, 0.35);
        color: #93C5FD;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 1rem;
    }

    /* Form Container Cards */
    .section-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 1.5rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
        margin-bottom: 1.5rem;
    }

    .section-header {
        font-size: 1.12rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 1.25rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid #F1F5F9;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Result Cards */
    .result-card-good {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 2px solid #10B981;
        border-radius: 16px;
        padding: 2.2rem;
        text-align: center;
        box-shadow: 0 10px 20px -3px rgba(16, 185, 129, 0.2);
    }

    .result-card-bad {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        border-radius: 16px;
        padding: 2.2rem;
        text-align: center;
        box-shadow: 0 10px 20px -3px rgba(239, 68, 68, 0.2);
    }

    .result-heading {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0.6rem 0;
    }

    .good-theme {
        color: #065F46;
    }

    .bad-theme {
        color: #991B1B;
    }

    .result-note {
        font-size: 0.98rem;
        color: #475569;
        margin-top: 0.4rem;
        line-height: 1.5;
    }

    /* Button Polish */
    div.stButton > button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: #FFFFFF;
        font-weight: 700;
        font-size: 1.05rem;
        border: none;
        border-radius: 10px;
        padding: 0.8rem 2rem;
        transition: all 0.25s ease;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.28);
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.45);
        color: #FFFFFF;
        transform: translateY(-1px);
    }

    .footer {
        text-align: center;
        color: #94A3B8;
        font-size: 0.85rem;
        padding: 2.5rem 0 1.5rem 0;
        border-top: 1px solid #E2E8F0;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Model & Encoder Loader (Cached for Performance)
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_encoders():
    model = joblib.load("extra_trees_credict_model.pkl")
    encoders = {
        col: joblib.load(f"{col}_encoder.pkl")
        for col in ["Sex", "Housing", "Saving accounts", "Checking account"]
    }
    return model, encoders

try:
    model, encoders = load_model_and_encoders()
    load_ready = True
except Exception as e:
    load_ready = False
    load_error_msg = str(e)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 💳 CreditWise")
    st.markdown("**Enterprise Credit Risk Analytics**")
    st.caption("AI-powered underwriting assessment tool")
    st.divider()

    st.markdown("### ⚙️ Model Details")
    st.markdown("- **Algorithm:** Extra Trees Classifier")
    st.markdown("- **Dataset:** German Credit Benchmark")
    st.markdown("- **Evaluation:** Binary Risk Classification")
    st.markdown("- **Features Analyzed:** 8 Financial & Demographic Variables")
    st.divider()

    st.markdown("### ℹ️ Job Code Reference")
    st.caption("""
    - **0:** Unskilled (Non-resident)
    - **1:** Unskilled (Resident)
    - **2:** Skilled Employee
    - **3:** Management / Highly Qualified
    """)
    st.divider()

    st.markdown("### 🛡️ System Status")
    if load_ready:
        st.success("● Inference Engine Active")
    else:
        st.error("● Model Artifacts Offline")

# ---------------------------------------------------------
# Main Page Content
# ---------------------------------------------------------
# Hero Banner
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">Credit Risk Assessment Portal</div>
    <p class="hero-subtitle">
        Enter applicant demographic details, financial standing, and credit particulars 
        to evaluate creditworthiness using an ensemble machine learning model.
    </p>
    <div class="status-badge">
        <span>✨ ExtraTrees Predictive Engine</span> • <span>Production Ready</span>
    </div>
</div>
""", unsafe_allow_html=True)

if not load_ready:
    st.error(f"❌ Failed to load model or encoders: {load_error_msg}")
    st.stop()

# Form Inputs arranged into 3 responsive columns
st.markdown("#### 📋 Applicant Evaluation Form")

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown("""
    <div class="section-card">
        <div class="section-header">👤 Demographics & Personal</div>
    </div>
    """, unsafe_allow_html=True)
    
    age = st.number_input(
        "Age (Years)", 
        min_value=18, 
        max_value=80, 
        value=30, 
        help="Applicant age must be between 18 and 80"
    )
    
    sex = st.selectbox(
        "Sex", 
        ["male", "female"], 
        format_func=lambda s: s.capitalize(),
        help="Applicant gender"
    )
    
    housing = st.selectbox(
        "Housing Status", 
        ["own", "rent", "free"], 
        format_func=lambda h: h.capitalize(),
        help="Residential status of the applicant"
    )

with col2:
    st.markdown("""
    <div class="section-card">
        <div class="section-header">💼 Financial & Employment</div>
    </div>
    """, unsafe_allow_html=True)
    
    job = st.number_input(
        "Job Classification (0-3)", 
        min_value=0, 
        max_value=3, 
        value=1,
        help="0: Unskilled non-resident, 1: Unskilled resident, 2: Skilled, 3: Management/Highly qualified"
    )
    
    saving_accounts = st.selectbox(
        "Saving Accounts", 
        ["little", "moderate", "rich", "quite rich"], 
        format_func=lambda s: s.title(),
        help="Savings account balance tier"
    )
    
    checking_accounts = st.selectbox(
        "Checking Accounts", 
        ["little", "moderate", "rich"], 
        format_func=lambda c: c.title(),
        help="Checking account balance tier"
    )

with col3:
    st.markdown("""
    <div class="section-card">
        <div class="section-header">💰 Loan Terms & Duration</div>
    </div>
    """, unsafe_allow_html=True)
    
    credit_amount = st.number_input(
        "Credit Amount (DM / USD)", 
        min_value=0, 
        value=1000, 
        step=250,
        help="Total principal amount requested"
    )
    
    duration = st.number_input(
        "Duration (Months)", 
        min_value=1, 
        value=12, 
        step=1,
        help="Loan repayment tenure in months"
    )

    # Derived indicator for underwriter insight
    monthly_est = round(credit_amount / duration, 2) if duration > 0 else 0
    st.metric(
        label="Estimated Monthly Installment", 
        value=f"${monthly_est:,.2f}"
    )

# ---------------------------------------------------------
# Applicant Data Preview Card
# ---------------------------------------------------------
with st.expander("🔎 Review Applicant Data Snapshot", expanded=False):
    summary_df = pd.DataFrame({
        "Attribute": [
            "Age", "Sex", "Job Tier", "Housing", 
            "Savings Balance", "Checking Balance", 
            "Credit Amount", "Duration (Months)"
        ],
        "Value": [
            f"{age} yrs", sex.capitalize(), f"Tier {job}", housing.capitalize(),
            saving_accounts.capitalize(), checking_accounts.capitalize(),
            f"${credit_amount:,.2f}", f"{duration} mos"
        ]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# DataFrame Preparation (Identical functional encoding)
# ---------------------------------------------------------
input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [job],
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account": [encoders["Checking account"].transform([checking_accounts])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration]
})

# ---------------------------------------------------------
# Prediction & Results Section
# ---------------------------------------------------------
_, btn_col, _ = st.columns([1, 2, 1])

with btn_col:
    predict_clicked = st.button("🔍 Assess Credit Risk", use_container_width=True)

if predict_clicked:
    with st.spinner("Executing ExtraTrees risk model inference..."):
        pred = model.predict(input_df)[0]
        
        # Check if model supports class probabilities
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(input_df)[0]
            except Exception:
                proba = None

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Assessment Outcome")

    if pred == 1:
        st.markdown("""
        <div class="result-card-good">
            <span style="font-size: 3rem;">✅</span>
            <div class="result-heading good-theme">PREDICTED RISK: GOOD (LOW RISK)</div>
            <p class="result-note">
                The applicant profile meets standard financial credibility criteria.<br>
                Low probability of default detected. Recommended for credit approval.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown("""
        <div class="result-card-bad">
            <span style="font-size: 3rem;">⚠️</span>
            <div class="result-heading bad-theme">PREDICTED RISK: BAD (HIGH RISK)</div>
            <p class="result-note">
                The applicant profile exhibits financial risk factors associated with default.<br>
                Underwriting review, collateral security, or restructured terms are advised.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # If model probabilities are available, show a breakdown
    if proba is not None and len(proba) == 2:
        st.markdown("<br>", unsafe_allow_html=True)
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.metric("Risk Level: Bad Confidence", f"{proba[0] * 100:.1f}%")
            st.progress(float(proba[0]))
        with p_col2:
            st.metric("Risk Level: Good Confidence", f"{proba[1] * 100:.1f}%")
            st.progress(float(proba[1]))

# Footer
st.markdown("""
<div class="footer">
    CreditWise Automated Underwriting System • Built with Streamlit & Scikit-Learn • Production Build
</div>
""", unsafe_allow_html=True)
