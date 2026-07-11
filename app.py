import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

# --- Load model once, cached across reruns ---
@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

model = load_model()

st.title("📉 Customer Churn Predictor")
st.write(
    "Enter a customer's details and the model estimates their probability of churning. "
    "Trained on a synthetic telecom dataset with a Random Forest pipeline."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
    total_charges = st.number_input(
        "Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges)
    )
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    )

with col2:
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

input_df = pd.DataFrame(
    [
        {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Contract": contract,
            "InternetService": internet_service,
            "TechSupport": tech_support,
            "OnlineSecurity": online_security,
            "PaymentMethod": payment_method,
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "PaperlessBilling": paperless_billing,
        }
    ]
)

st.divider()

if st.button("Predict Churn Risk", type="primary", use_container_width=True):
    proba = model.predict_proba(input_df)[0, 1]
    prediction = "Likely to churn" if proba >= 0.5 else "Likely to stay"

    st.metric("Churn Probability", f"{proba:.1%}")

    if proba >= 0.5:
        st.error(f"⚠️ {prediction}")
    else:
        st.success(f"✅ {prediction}")

    st.progress(min(proba, 1.0))

    with st.expander("See the customer profile sent to the model"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}))

st.caption(
    "Note: this model is trained on synthetic data for demonstration purposes. "
    "Swap in your own labeled dataset (run generate_data.py → train_model.py) to make it real."
)
