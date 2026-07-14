# 📉 Churn Compass: Customer Retention Predictor

An interactive Streamlit application that turns a customer's account profile into a live churn-risk score, built on a scikit-learn pipeline trained on telecom-style subscriber data.

## 🎨 Visual Philosophy: Signal-First Minimalism
Unlike a dense analytics dashboard, the Churn Compass app leans into a **decision-support framework** — the interface exists to answer one question fast: *is this customer at risk, and why?*
* **Two-Column Input Grid:** Account attributes (tenure, charges, contract) and service/demographic attributes (support, security, household) are split into separate columns so the form reads like a customer profile card, not a spreadsheet.
* **Traffic-Light Verdicts:** Prediction outcomes use a strict red/green convention (`st.error` for churn risk, `st.success` for retained) so the result is legible at a glance, no legend required.
* **Progressive Disclosure:** The raw feature vector sent to the model is tucked into a collapsed expander — visible on demand for debugging or trust-building, but out of the way for a quick read.

---

## 📖 The Project Storyline

The project is structured as a three-stage pipeline, mirroring how a churn model actually gets built and shipped:

### 1. 🔍 Exploration (The Notebook)
* **The Story:** `churn_model_training.ipynb` walks through the data before touching a model — class balance, tenure and monthly-charge distributions split by churn outcome, and churn rate by contract type.
* **The Translation:** Establishes *why* the model should work before building it. Month-to-month customers churn more because there's no lock-in; short-tenure customers churn more because loyalty hasn't formed yet. The model should end up leaning on exactly these signals.

### 2. ⚙️ Training (The Pipeline)
* **The Story:** `train_model.py` bundles preprocessing (scaling + one-hot encoding) and a class-weighted Random Forest into a single `Pipeline` object, evaluated on a held-out test set.
* **The Translation:** Because churn is imbalanced (~33% positive class), raw accuracy would be misleading — a model that just predicts "no churn" for everyone would look deceptively good. Evaluation instead centers on ROC-AUC and per-class recall, so the model is actually held accountable for catching churners, not just being right on average.

### 3. 🖥️ Deployment (The App)
* **The Story:** `app.py` loads the saved pipeline (`churn_model.pkl`) and exposes it as a live form — no retraining, no notebook, just instant predictions from saved weights.
* **The Translation:** This is the handoff moment: the same artifact produced in the notebook becomes the thing a non-technical stakeholder can actually click through, with `@st.cache_resource` ensuring the model loads once per session rather than on every interaction.

---

## 🛠️ Key Design Decisions

### Bundling Preprocessing Inside the Model Artifact
Rather than saving the encoder, scaler, and classifier as separate files, `train_model.py` wraps all three in one `sklearn.Pipeline` before calling `joblib.dump()`.

**Why this matters:**
1. **Single Source of Truth:** The app only ever calls `model.predict_proba(input_df)` — it never needs to know that `OneHotEncoder` or `StandardScaler` exist.
2. **No Train/Serve Skew:** Any change to preprocessing logic automatically ships with the next retrain; there's no risk of the app using stale encoding rules.
3. **Portability:** `churn_model.pkl` is the only artifact required to run predictions anywhere — Streamlit Cloud, a notebook, or another script.

### Handling Unseen Categories Gracefully
`OneHotEncoder(handle_unknown="ignore")` prevents the app from crashing if a user-entered category wasn't seen during training, instead of raising an error mid-prediction.

---

## 🚀 Technical Requirements & Execution

### 1. Data & Model Files
Running `generate_data.py` produces `churn_data.csv`; running `train_model.py` produces `churn_model.pkl`. Both are already included, so the app runs out of the box — regenerate them only if you swap in real customer data.

### 2. Running the Application
Install dependencies and launch the dashboard:

```bash
pip install -r requirements.txt
streamlit run app.py
```

### 3. Deploying to Streamlit Community Cloud
Push the project folder to a GitHub repo, connect it at share.streamlit.io, and point the deploy at `app.py`. Dependencies build automatically from `requirements.txt`.

