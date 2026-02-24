from src.pipeline import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET
from src.feature_engineering import engineer_all_features
from src.config import (
    MODELS_DIR, DATA_FILE, FIGURES_DIR, TABLES_DIR,
    COST_ASSUMPTIONS, WTP_THRESHOLDS, CLINICAL_RANGES
)
import os
import sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Healthcare Outcome Risk Assessment",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a5276;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5d6d7e;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .risk-low { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .risk-medium { background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); }
    .risk-high { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }
    /* Hide the running/cycling/stop status animation in the top-right corner */
    .stStatusWidget, [data-testid="stStatusWidget"] {
        display: none !important;
    }
</style>

    flags = []
    bmi = patient_data["weight_kg"] / (patient_data["height_cm"] / 100) ** 2
    sbp = patient_data["sbp"]
    dbp = patient_data["dbp"]
    hba1c = patient_data["hba1c"]
    ldl = patient_data["ldl"]
    hr = patient_data["heart_rate"]
    creatinine = patient_data["creatinine"]

    if sbp >= 180 or dbp >= 120:
        flags.append(("Hypertensive Crisis", f"SBP {sbp} / DBP {dbp} mmHg", "critical"))
    elif sbp >= 140 or dbp >= 90:
        flags.append(("Stage 2 Hypertension", f"SBP {sbp} / DBP {dbp} mmHg", "high"))
    elif sbp >= 130 or dbp >= 80:
        flags.append(("Stage 1 Hypertension", f"SBP {sbp} / DBP {dbp} mmHg", "moderate"))
    elif sbp >= 120:
        flags.append(("Elevated BP", f"SBP {sbp} mmHg", "mild"))

    if hba1c >= 9.0:
        flags.append(("Severe Uncontrolled Diabetes", f"HbA1c {hba1c}%", "critical"))
    elif hba1c >= 6.5:
        flags.append(("Diabetes", f"HbA1c {hba1c}%", "high"))
    elif hba1c >= 5.7:
        flags.append(("Prediabetes", f"HbA1c {hba1c}%", "moderate"))

    if ldl >= 190:
        flags.append(("Very High LDL", f"LDL {ldl} mg/dL", "critical"))
    elif ldl >= 160:
        flags.append(("High LDL", f"LDL {ldl} mg/dL", "high"))
    elif ldl >= 130:
        flags.append(("Borderline High LDL", f"LDL {ldl} mg/dL", "moderate"))

    if bmi >= 40:
        flags.append(("Class III Obesity", f"BMI {bmi:.1f}", "critical"))
    elif bmi >= 35:
        flags.append(("Class II Obesity", f"BMI {bmi:.1f}", "high"))
    elif bmi >= 30:
        flags.append(("Class I Obesity", f"BMI {bmi:.1f}", "moderate"))
    elif bmi >= 25:
        flags.append(("Overweight", f"BMI {bmi:.1f}", "mild"))
    elif bmi < 18.5:
        flags.append(("Underweight", f"BMI {bmi:.1f}", "moderate"))

    if hr > 120:
        flags.append(("Tachycardia", f"HR {hr} bpm", "high"))
    elif hr > 100:
        flags.append(("Elevated Heart Rate", f"HR {hr} bpm", "moderate"))
    elif hr < 50:
        flags.append(("Bradycardia", f"HR {hr} bpm", "moderate"))

    if creatinine >= 2.0:
        flags.append(("Elevated Creatinine", f"{creatinine} mg/dL — possible renal impairment", "high"))
    elif creatinine >= 1.3:
        flags.append(("Borderline Creatinine", f"{creatinine} mg/dL", "moderate"))

    if patient_data["smoking_status"] == "Current":
        flags.append(("Active Smoker", "Current smoker — increased CVD risk", "moderate"))

    if patient_data["exercise_freq"] == 0:
        flags.append(("Sedentary", "No exercise — below WHO recommendation", "mild"))
    elif patient_data["exercise_freq"] <= 1:
        flags.append(("Low Activity", f"{patient_data['exercise_freq']} session/week", "mild"))

    return flags

def render_clinical_flags(flags):

    if not flags:
        st.success("All clinical values are within normal ranges.")
        return

    severity_styles = {
        "critical": {"bg": "#e74c3c", "icon": "🔴"},
        "high":     {"bg": "#e67e22", "icon": "🟠"},
        "moderate": {"bg": "#f1c40f", "icon": "🟡"},
        "mild":     {"bg": "#3498db", "icon": "🔵"},
    }

    html_parts = []
    for label, detail, severity in flags:
        style = severity_styles.get(severity, severity_styles["mild"])
        html_parts.append(
            f'<span style="display:inline-block; background:{style["bg"]}; color:white; '
            f'padding:4px 12px; border-radius:16px; margin:4px; font-size:0.85rem; '
            f'font-weight:600;">{style["icon"]} {label}: {detail}</span>'
        )

    st.markdown("".join(html_parts), unsafe_allow_html=True)

def prediction_section(patient_data, model):

    df_patient = pd.DataFrame([patient_data])
    df_patient = engineer_all_features(df_patient)

    from src.pipeline import prepare_data
    X_patient, _ = prepare_data(df_patient)

    prob = model.predict_proba(X_patient)[:, 1][0]
    risk_cat, risk_class, risk_color = get_risk_category(prob)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card {risk_class}">
            <h2 style="margin:0; font-size:2.5rem;">{prob:.1%}</h2>
            <p style="margin:0; font-size:1.1rem;">Predicted Success Probability</p>
        </div>

        <div class="metric-card {risk_class}">
            <h2 style="margin:0; font-size:2rem;">{risk_cat}</h2>
            <p style="margin:0; font-size:1.1rem;">Risk Category</p>
        </div>

        <div class="metric-card">
            <h2 style="margin:0; font-size:2rem;">${total_expected:,.0f}</h2>
            <p style="margin:0; font-size:1.1rem;">Expected Total Cost</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Clinical Flags")
    flags = get_clinical_flags(patient_data)
    render_clinical_flags(flags)

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Treatment Cost", f"${patient_data['treatment_cost']:,.0f}")
    c2.metric("Expected Downstream", f"${expected_downstream:,.0f}")
    c3.metric("Potential Saving", f"${potential_saving:,.0f}")

    bmi = patient_data["weight_kg"] / (patient_data["height_cm"] / 100) ** 2
    c4.metric("BMI", f"{bmi:.1f}")

    return prob

def population_tab(df):

    st.subheader("Population Overview")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="age", color="outcome_binary", barmode="overlay",
                           labels={"outcome_binary": "Outcome"}, title="Age Distribution by Outcome",
                           color_discrete_map={0: "#e74c3c", 1: "#2ecc71"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        outcome_by_arm = df.groupby("treatment_arm")[
            "outcome_binary"].mean().reset_index()
        fig = px.bar(outcome_by_arm, x="treatment_arm", y="outcome_binary",
                     title="Outcome Rate by Treatment Arm",
                     labels={"outcome_binary": "Success Rate",
                             "treatment_arm": "Treatment"},
                     color="outcome_binary", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.box(df, x="outcome_binary", y="hba1c", color="outcome_binary",
                     title="HbA1c by Outcome",
                     labels={"outcome_binary": "Outcome",
                             "hba1c": "HbA1c (%)"},
                     color_discrete_map={0: "#e74c3c", 1: "#2ecc71"})
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.scatter(df, x="sbp", y="hba1c", color="outcome_binary",
                         title="SBP vs HbA1c (colored by outcome)",
                         labels={"sbp": "SBP (mmHg)", "hba1c": "HbA1c (%)"},
                         color_discrete_map={0: "#e74c3c", 1: "#2ecc71"},
                         opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)

def feature_importance_tab():

    st.subheader("Model Feature Importance")

    shap_path = os.path.join(TABLES_DIR, "shap_importance.csv")
    if os.path.exists(shap_path):
        shap_df = pd.read_csv(shap_path).sort_values(
            "mean_abs_shap", ascending=True).tail(15)
        fig = px.bar(shap_df, x="mean_abs_shap", y="feature", orientation="h",
                     title="SHAP Feature Importance (Top 15)",
                     labels={"mean_abs_shap": "Mean |SHAP Value|",
                             "feature": "Feature"},
                     color="mean_abs_shap", color_continuous_scale="Viridis")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the interpretation module first to generate SHAP values.")

    col1, col2 = st.columns(2)
    beeswarm_path = os.path.join(FIGURES_DIR, "shap_beeswarm.png")
    if os.path.exists(beeswarm_path):
        col1.image(beeswarm_path, caption="SHAP Beeswarm Plot")
    dep_path = os.path.join(FIGURES_DIR, "shap_dependence_top8.png")
    if os.path.exists(dep_path):
        col2.image(dep_path, caption="SHAP Dependence (Top 8)")

def cost_effectiveness_tab(df, model):

    st.subheader("Cost-Effectiveness Analysis")

    nmb_path = os.path.join(FIGURES_DIR, "nmb_curves.png")
    if os.path.exists(nmb_path):
        st.image(nmb_path, caption="NMB Acceptability Curves")

    col1, col2 = st.columns(2)

    icer_path = os.path.join(TABLES_DIR, "icer_analysis.csv")
    if os.path.exists(icer_path):
        with col1:
            st.markdown("### ICER Analysis")
            icer_df = pd.read_csv(icer_path)
            st.dataframe(icer_df, use_container_width=True)

    seg_path = os.path.join(TABLES_DIR, "actionable_segments.csv")
    if os.path.exists(seg_path):
        with col2:
            st.markdown("### Actionable Segments")
            seg_df = pd.read_csv(seg_path).head(10)
            st.dataframe(seg_df, use_container_width=True)

    tornado_path = os.path.join(FIGURES_DIR, "tornado_sensitivity.png")
    if os.path.exists(tornado_path):
        st.image(tornado_path, caption="Tornado Sensitivity Analysis")

def sensitivity_tab():

    st.subheader("Interactive Sensitivity Analysis")

    st.markdown(
        "Adjust cost assumptions below to see the impact on expected outcomes:")

    col1, col2 = st.columns(2)
    with col1:
        cost_good = st.slider("Cost - Good Outcome ($)", 500, 10000,
                              COST_ASSUMPTIONS["cost_good_outcome"], 500)
        cost_bad = st.slider("Cost - Bad Outcome ($)", 10000, 100000,
                             COST_ASSUMPTIONS["cost_bad_outcome"], 5000)
        qaly_good = st.slider("QALY - Good Outcome", 0.3, 1.0,
                              COST_ASSUMPTIONS["qaly_good_outcome"], 0.05)
    with col2:
        qaly_bad = st.slider("QALY - Bad Outcome", 0.1, 0.7,
                             COST_ASSUMPTIONS["qaly_bad_outcome"], 0.05)
        wtp = st.slider("Willingness to Pay ($/QALY)",
                        10000, 200000, 50000, 10000)

    delta_qaly = qaly_good - qaly_bad
    delta_cost = cost_bad - cost_good
    nmb_per_improvement = wtp * delta_qaly - delta_cost

    st.markdown("### Results Under Modified Assumptions")
    m1, m2, m3 = st.columns(3)
    m1.metric("QALY Gain (Good vs Bad)", f"{delta_qaly:.2f}")
    m2.metric("Cost Difference (Bad - Good)", f"${delta_cost:,.0f}")
    m3.metric("NMB per Outcome Improvement", f"${nmb_per_improvement:,.0f}")

    if delta_qaly > 0:
        break_even_wtp = delta_cost / delta_qaly
        st.info(f"Break-even WTP threshold: **${break_even_wtp:,.0f}/QALY** "
                f"(current WTP: ${wtp:,}/QALY)")
    else:
        st.warning(
            "No QALY gain between outcomes - cost-effectiveness analysis not applicable.")

def main():

    st.markdown('<p class="main-header">Healthcare Outcome Risk Assessment Dashboard</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predictive analytics for treatment outcome and cost-effectiveness</p>',
                unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.error("No trained model found. Run `python -m src.train` first.")
        return

    df = load_population_data()

    patient_data = build_patient_input()

    st.markdown("---")
    st.markdown("## Individual Patient Prediction")
    prob = prediction_section(patient_data, model)

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Population Summary", "Feature Importance",
        "Cost-Effectiveness", "Sensitivity Analysis"
    ])

    with tab1:
        population_tab(df)
    with tab2:
        feature_importance_tab()
    with tab3:
        cost_effectiveness_tab(df, model)
    with tab4:
        sensitivity_tab()

    st.markdown("---")
    st.caption("Healthcare Outcome Prediction System | "
               "Model predictions are for analytical purposes only. "
               "Not intended for clinical decision-making without physician oversight.")

if __name__ == "__main__":
    main()

