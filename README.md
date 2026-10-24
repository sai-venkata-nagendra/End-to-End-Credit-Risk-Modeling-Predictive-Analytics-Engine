# 💳 CreditWise — AI Credit Risk Assessment System

CreditWise is an enterprise-grade Credit Risk Prediction & Underwriting Assistant powered by machine learning. It provides loan underwriters and credit analysts with instant risk evaluations based on applicant demographics, banking relationships, and loan particulars.

# **Live project link** [Streamlit]([https://streamlit.io/](https://end-to-end-credit-risk-modeling-predictive-analytics-engine-vu.streamlit.app/))

---

## 🚀 Key Features

- **Executive UI Design**: Built with a sleek, responsive interface utilizing custom CSS, modern typography (`Plus Jakarta Sans`), and a dark-theme hero header.
- **Structured 3-Tier Input Form**:
  - **👤 Personal & Demographics**: Age, Gender, Housing status.
  - **💼 Financial & Employment**: Job classification (tiers 0–3), Checking account balance tier, Savings account balance tier.
  - **💰 Loan Terms**: Credit principal amount and duration, with real-time **Estimated Monthly Installment** calculation.
- **Machine Learning Inference**: Powered by a pre-trained **Extra Trees Classifier** with cached model loading (`@st.cache_resource`) for sub-second predictions.
- **Risk Assessment & Confidence Scoring**:
  - **Good Credit Risk**: Instant approval indicators with probability confidence gauges.
  - **Bad Credit Risk**: Cautionary advisory flagging default risks for further manual underwriting review.
- **Applicant Data Snapshot**: An expandable review card allowing underwriters to verify all inputs before running inference.

---

## 📂 Project Structure

```text
credit risk analysis/
│
├── app.py                           # Main Streamlit web application
├── german_credit_data.csv           # Benchmark dataset (German Credit Data)
├── extra_trees_credict_model.pkl    # Serialized Extra Trees model
│
├── Sex_encoder.pkl                  # Label encoder for 'Sex'
├── Housing_encoder.pkl              # Label encoder for 'Housing'
├── Saving accounts_encoder.pkl      # Label encoder for 'Saving accounts'
├── Checking account_encoder.pkl     # Label encoder for 'Checking account'
│
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation & setup guide
```

---

## 🛠️ Tech Stack & Dependencies

- **Language:** Python 3.9+
- **Frontend / Framework:** [Streamlit](https://streamlit.io/)
- **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/) (`ExtraTreesClassifier`)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/)
- **Model Serialization:** [Joblib](https://joblib.readthedocs.io/)

---

## 📥 Installation & Setup

### 1. Clone or Open the Project
Navigate to the project root directory:
```bash
cd "d:/credit risk analysis"
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Libraries
```bash
pip install -r requirements.txt
```
*(Or manually install: `pip install streamlit pandas scikit-learn joblib`)*

---

## ▶️ Running the Application

Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```

Once started, Streamlit will provide local and network URLs (typically `http://localhost:8501`). Open this URL in any modern web browser.

---

## 📊 Feature Specifications & Reference

| Feature | Type | Values / Categories | Description |
| :--- | :--- | :--- | :--- |
| **Age** | Numerical | 18 – 80 | Applicant age in years |
| **Sex** | Categorical | `male`, `female` | Biological gender |
| **Job** | Numerical | 0, 1, 2, 3 | **0:** Unskilled non-resident<br>**1:** Unskilled resident<br>**2:** Skilled employee<br>**3:** Management / Highly qualified |
| **Housing** | Categorical | `own`, `rent`, `free` | Residential arrangement |
| **Saving accounts** | Categorical | `little`, `moderate`, `rich`, `quite rich` | Liquid savings reserve tier |
| **Checking account**| Categorical | `little`, `moderate`, `rich` | Operating bank balance tier |
| **Credit amount** | Numerical | $\ge 0$ (Default: 1,000) | Total credit principal requested |
| **Duration** | Numerical | $\ge 1$ month (Default: 12) | Loan tenure in months |

---

## 🧪 Model Stress-Testing (Tricky Test Case)

To test whether the model is making intelligent, risk-adjusted decisions rather than naive heuristic guesses, test this high-conflict edge case:

| Field | Value | Underlying Logic |
| :--- | :--- | :--- |
| **Age** | `44` | Prime employment age |
| **Sex** | `female` | Standard profile |
| **Housing** | `free` | No owned real estate asset |
| **Job** | `3` | Executive / Management tier |
| **Saving accounts** | `little` | Minimal emergency reserve buffer |
| **Checking account**| `moderate`| Modest operating cash flow |
| **Credit amount** | `12579` | High-risk, oversized borrowing request |
| **Duration** | `24` | 24-month accelerated repayment |

> **Observed Behavior:** Despite the executive job title (`Tier 3`) and prime age (`44`), the model flags this applicant as **BAD RISK**, correctly recognizing that excessive leverage without a savings buffer outweighs job title prestige.

---

## 📄 License & Disclaimer

This software is designed as a decision-support tool for credit analysts. Automated scores should complement, rather than replace, human underwriter discretion, institutional policy, and applicable fair lending regulations.
