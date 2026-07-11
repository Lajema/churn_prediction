"""
Trains a churn prediction model and saves it (+ preprocessing pipeline)
to a single .pkl file that the Streamlit app loads at inference time.
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("churn_data.csv")

X = df.drop(columns=["Churn"])
y = (df["Churn"] == "Yes").astype(int)

numeric_features = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
categorical_features = [
    "Contract",
    "InternetService",
    "TechSupport",
    "OnlineSecurity",
    "PaymentMethod",
    "Partner",
    "Dependents",
    "PaperlessBilling",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300, max_depth=8, random_state=42, class_weight="balanced"
            ),
        ),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

# Save the whole pipeline (preprocessing + model) as one artifact.
# The Streamlit app just loads this file — it doesn't need to know
# about encoders or scalers separately.
joblib.dump(model, "churn_model.pkl")
print("\nSaved trained pipeline to churn_model.pkl")
