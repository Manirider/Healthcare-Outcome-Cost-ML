from src.config import (
    MODELS_DIR, DATA_FILE, FIGURES_DIR, COST_ASSUMPTIONS,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES
)
from src.feature_engineering import engineer_all_features
from src.pipeline import prepare_data, build_preprocessor
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
import sys
from src.logger import get_logger
logger = get_logger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Healthcare Outcome Predictor",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .big-font {
        font-size:24px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""")

@st.cache_resource
def load_resources():
    try:
        pipeline = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    except:
        st.warning("Best model not found, falling back to XGBoost.")
        pipeline = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))

    model = pipeline.named_steps.get("model") or pipeline.steps[-1][1]

    return pipeline, model

@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_FILE)
    return df

pipeline, model = load_resources()
df_ref = load_dataset()

st.title("Healthcare Outcome Analysis Dashboard")
st.markdown("### Predict patient outcomes and evaluate treatment cost-effectiveness")

st.sidebar.header("Patient Profile")

def user_input_features():

    age = st.sidebar.slider("Age (years)", 18, 90, 55)
    height = st.sidebar.slider("Height (cm)", 120, 220, 170)
    weight = st.sidebar.slider("Weight (kg)", 30, 250, 80)
    sbp = st.sidebar.slider("Systolic BP (mmHg)", 90, 200, 130)
    dbp = st.sidebar.slider("Diastolic BP (mmHg)", 50, 130, 85)
    heart_rate = st.sidebar.slider("Heart Rate (bpm)", 40, 180, 75)
    creatinine = st.sidebar.slider("Creatinine (mg/dL)", 0.3, 10.0, 1.0)
    hba1c = st.sidebar.slider("HbA1c (%)", 4.0, 15.0, 6.5)
    ldl = st.sidebar.slider("LDL Cholesterol (mg/dL)", 50, 250, 110)
    exercise = st.sidebar.slider("Exercise Freq (days/week)", 0, 7, 2)

    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    smoking = st.sidebar.selectbox(
        "Smoking Status", ["Never", "Former", "Current"])
    treatment = st.sidebar.selectbox(
        "Treatment Arm", ["Standard", "Control", "Enhanced"])

    data = {
        "age": age,
        "height_cm": height,
        "weight_kg": weight,
        "sbp": sbp,
        "dbp": dbp,
        "heart_rate": heart_rate,
        "creatinine": creatinine,
        "hba1c": hba1c,
        "ldl": ldl,
        "exercise_freq": exercise,
        "sex": sex,
        "smoking_status": smoking,
        "treatment_arm": treatment,
        "treatment_cost": 0.0
    }

    if treatment == "Enhanced":
        data["treatment_cost"] = COST_ASSUMPTIONS["treatment_cost_enhanced"]
    elif treatment == "Standard":
        data["treatment_cost"] = COST_ASSUMPTIONS["treatment_cost_standard"]
    else:
        data["treatment_cost"] = COST_ASSUMPTIONS["treatment_cost_control"]

    return pd.DataFrame([data])

input_df = user_input_features()

input_engineered = engineer_all_features(input_df.copy())

prob = 0.5
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Prediction & Interpretation")

    try:
        prob = pipeline.predict_proba(input_engineered)[0][1]
        decision = "Good Outcome" if prob > 0.5 else "Poor Outcome"

        color = "green" if prob > 0.5 else "red"

        st.markdown(f"""
        <div class="metric-card">
            <h2 style='color:{color};'>{decision}</h2>
            <p class="big-font">Probability: {prob:.2%}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Why this prediction?")

        preprocessor = pipeline.named_steps.get(
            "preprocessor") or pipeline.steps[0][1]
        X_proc = preprocessor.transform(input_engineered)

        try:
            feats = preprocessor.get_feature_names_out()
            feature_names = [str(f).replace(
                "num__", "").replace("cat__", "") for f in feats]
        except:
            feature_names = [f"Feature {i}" for i in range(X_proc.shape[1])]

        try:
            try:

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_proc)
            except Exception:

                background = df_ref.sample(min(100, len(df_ref)))
                background_eng = engineer_all_features(background.copy())
                background_proc = preprocessor.transform(background_eng)

                explainer = shap.LinearExplainer(model, background_proc)
                shap_values = explainer.shap_values(X_proc)

            if isinstance(shap_values, list):
                shap_val = shap_values[1][0]
                expected_val = explainer.expected_value[1]
            else:
                shap_val = shap_values[0]
                expected_val = explainer.expected_value

            explanation_df = pd.DataFrame({
                "feature": feature_names,
                "impact": shap_val
            }).sort_values("impact", key=abs, ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(8, 5))
            colors_bar = ["#2ecc71" if x >
                          0 else "#e74c3c" for x in explanation_df["impact"]]
            ax.barh(explanation_df["feature"],
                    explanation_df["impact"], color=colors_bar)
            ax.set_xlabel("SHAP Value (Impact on Log-Odds)")
            ax.set_title("Top Clinical Drivers for this Patient")
            plt.gca().invert_yaxis()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"SHAP explanation failed: {e}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

with col2:
    st.subheader("Financial Impact")

    wtp = st.number_input("Willingness to Pay ($/QALY)",
                          value=50000, step=5000)

    q_good = COST_ASSUMPTIONS["qaly_good_outcome"]
    q_bad = COST_ASSUMPTIONS["qaly_bad_outcome"]
    exp_qaly = prob * q_good + (1 - prob) * q_bad

    c_good = COST_ASSUMPTIONS["cost_good_outcome"]
    c_bad = COST_ASSUMPTIONS["cost_bad_outcome"]
    t_cost = input_engineered["treatment_cost"].iloc[0]

    exp_downstream = prob * c_good + (1 - prob) * c_bad
    total_cost = t_cost + exp_downstream

    nmb = wtp * exp_qaly - total_cost

    st.metric("Expected QALYs", f"{exp_qaly:.2f}")
    st.metric("Total Expected Cost", f"${total_cost:,.0f}")
    st.metric("Net Monetary Benefit", f"${nmb:,.0f}")

    st.markdown("---")
    st.markdown("#### Scenario Analysis")
    cost_modifier = st.slider("Adjust Treatment Cost (%)", -50, 50, 0)
    new_t_cost = t_cost * (1 + cost_modifier/100)
    new_total = new_t_cost + exp_downstream
    new_nmb = wtp * exp_qaly - new_total

    delta = new_nmb - nmb
    st.write(f"New NMB: **${new_nmb:,.0f}**")
    st.write(f"Change: **${delta:+,.0f}**")

st.markdown("---")
st.markdown("Interactive dashboard for treatment outcome analysis and cost-effectiveness.")

