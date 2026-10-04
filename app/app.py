from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "churn_model.joblib"

st.set_page_config(page_title="Churn Predictor", page_icon="📉")


@st.cache_resource
def load_model():
    art = joblib.load(MODEL_PATH)
    pipe = art["pipeline"]
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    return pipe, art["threshold"], explainer


pipe, threshold, explainer = load_model()

st.title("📉 Credit Card Customer Churn Predictor")
st.write("Enter a customer's details to see how likely they are to leave, and why.")

# ---------- Inputs ----------
st.sidebar.header("Customer details")

age = st.sidebar.slider("Age", 26, 73, 46)
gender = st.sidebar.selectbox("Gender", ["F", "M"])
dependents = st.sidebar.slider("Dependents", 0, 5, 2)
education = st.sidebar.selectbox(
    "Education", ["Graduate", "High School", "Unknown", "Uneducated",
                  "College", "Post-Graduate", "Doctorate"])
marital = st.sidebar.selectbox("Marital status", ["Married", "Single", "Divorced", "Unknown"])
income = st.sidebar.selectbox(
    "Income", ["Unknown", "Less than $40K", "$40K - $60K",
               "$60K - $80K", "$80K - $120K", "$120K +"])
card = st.sidebar.selectbox("Card type", ["Blue", "Silver", "Gold", "Platinum"])
months_on_book = st.sidebar.slider("Months with bank", 13, 56, 36)
relationship = st.sidebar.slider("Number of products held", 1, 6, 4)
inactive = st.sidebar.slider("Inactive months (last 12)", 0, 6, 2)
contacts = st.sidebar.slider("Contacts with bank (last 12 months)", 0, 6, 2)
credit_limit = st.sidebar.number_input("Credit limit", 1400.0, 35000.0, 4500.0, step=100.0)
revolving = st.sidebar.number_input("Revolving balance", 0.0, 2600.0, 1200.0, step=50.0)
amt_chg = st.sidebar.slider("Spend change Q4 vs Q1 (ratio)", 0.0, 3.5, 0.74)
trans_amt = st.sidebar.number_input("Total transaction amount (12 months)", 500.0, 19000.0, 3900.0, step=100.0)
trans_ct = st.sidebar.slider("Total transactions (12 months)", 10, 140, 67)
ct_chg = st.sidebar.slider("Transaction count change Q4 vs Q1 (ratio)", 0.0, 3.7, 0.70)

utilization = min(revolving / credit_limit, 1.0)

row = pd.DataFrame([{
    "Customer_Age": age,
    "Gender": gender,
    "Dependent_count": dependents,
    "Education_Level": education,
    "Marital_Status": marital,
    "Income_Category": income,
    "Card_Category": card,
    "Months_on_book": months_on_book,
    "Total_Relationship_Count": relationship,
    "Months_Inactive_12_mon": inactive,
    "Contacts_Count_12_mon": contacts,
    "Credit_Limit": credit_limit,
    "Total_Revolving_Bal": revolving,
    "Total_Amt_Chng_Q4_Q1": amt_chg,
    "Total_Trans_Amt": trans_amt,
    "Total_Trans_Ct": trans_ct,
    "Total_Ct_Chng_Q4_Q1": ct_chg,
    "Avg_Utilization_Ratio": utilization,
}])

# ---------- Prediction ----------
proba = pipe.predict_proba(row)[0, 1]

if proba >= threshold:
    label, action = "High risk", "Contact this customer soon with a retention offer."
elif proba >= threshold / 2:
    label, action = "Medium risk", "Keep an eye on this customer and watch for falling activity."
else:
    label, action = "Low risk", "No action needed."

col1, col2 = st.columns(2)
col1.metric("Churn probability", f"{proba:.1%}")
col2.metric("Risk level", label)
st.info(f"**Suggested action:** {action}")
st.caption(f"Customers above {threshold:.0%} are flagged as high risk "
           "(threshold chosen from a cost-benefit analysis).")

# ---------- Explanation ----------
st.subheader("Why this prediction?")

prep = pipe.named_steps["prep"]
Xt = prep.transform(row)
if hasattr(Xt, "toarray"):
    Xt = Xt.toarray()
names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]
sv = explainer(pd.DataFrame(Xt, columns=names))

contrib = pd.Series(sv.values[0], index=names)
top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)[::-1]

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(top.index, top.values, color=["#d62728" if v > 0 else "#1f77b4" for v in top.values])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Effect on churn risk  (red = raises risk, blue = lowers risk)")
plt.tight_layout()
st.pyplot(fig)

st.caption("Built with XGBoost and SHAP. Trained on the Kaggle Credit Card Customers dataset.")