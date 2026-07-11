"""
Generates a synthetic telecom customer churn dataset.
Mimics the structure of the classic Telco Customer Churn dataset
so the pipeline below is realistic and transferable to real data.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 3000

tenure = np.random.randint(0, 72, n)
monthly_charges = np.round(np.random.normal(65, 30, n).clip(18, 120), 2)
contract = np.random.choice(
    ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20]
)
internet_service = np.random.choice(
    ["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.20]
)
tech_support = np.random.choice(["Yes", "No"], n, p=[0.35, 0.65])
online_security = np.random.choice(["Yes", "No"], n, p=[0.35, 0.65])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n
)
senior_citizen = np.random.choice([0, 1], n, p=[0.84, 0.16])
partner = np.random.choice(["Yes", "No"], n, p=[0.5, 0.5])
dependents = np.random.choice(["Yes", "No"], n, p=[0.3, 0.7])
paperless_billing = np.random.choice(["Yes", "No"], n, p=[0.6, 0.4])

total_charges = np.round(monthly_charges * tenure * np.random.uniform(0.9, 1.0, n), 2)

# Build churn probability from a logical combination of features
# (this is what makes the "signal" learnable rather than pure noise)
churn_logit = (
    -1.5
    + 1.8 * (contract == "Month-to-month")
    - 1.0 * (contract == "Two year")
    + 1.2 * (internet_service == "Fiber optic")
    - 0.8 * (tech_support == "Yes")
    - 0.6 * (online_security == "Yes")
    - 0.03 * tenure
    + 0.01 * monthly_charges
    + 0.5 * (payment_method == "Electronic check")
    + 0.3 * senior_citizen
    - 0.3 * (partner == "Yes")
    + np.random.normal(0, 0.6, n)
)
churn_prob = 1 / (1 + np.exp(-churn_logit))
churn = (np.random.rand(n) < churn_prob).astype(int)
churn_label = np.where(churn == 1, "Yes", "No")

df = pd.DataFrame(
    {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "InternetService": internet_service,
        "TechSupport": tech_support,
        "OnlineSecurity": online_security,
        "PaymentMethod": payment_method,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "PaperlessBilling": paperless_billing,
        "Churn": churn_label,
    }
)

df.to_csv("churn_data.csv", index=False)
print(df.shape)
print(df["Churn"].value_counts(normalize=True))
