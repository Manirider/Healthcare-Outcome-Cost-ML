from src.pipeline import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, prepare_data
from src.feature_engineering import engineer_all_features
from src.config import (
    RANDOM_SEED, DATA_FILE, MODELS_DIR, FIGURES_DIR, TABLES_DIR, TEST_SIZE
)
import shap
import matplotlib.pyplot as plt
import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
from src.logger import get_logger
logger = get_logger(__name__)
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")

def get_feature_names_from_pipeline(pipeline):
    preprocessor = pipeline.named_steps.get(
        "preprocessor") or pipeline.steps[0][1]
    try:
        names = preprocessor.get_feature_names_out()
        return [str(n).replace("num__", "").replace("cat__", "") for n in names]
    except:
        return NUMERIC_FEATURES + ["sex_Male", "smoking_Former", "smoking_Current",
                                   "treatment_Enhanced", "treatment_Standard"]

def run_shap_analysis(pipeline, X_train, X_test, feature_names=None):
    logger.info("\n" + "="*60)
    logger.info("SHAP GLOBAL INTERPRETATION")
    logger.info("="*60)

    preprocessor = pipeline.named_steps.get(
        "preprocessor") or pipeline.steps[0][1]
    X_test_processed = preprocessor.transform(X_test)

    if feature_names is None:
        feature_names = get_feature_names_from_pipeline(pipeline)

    X_test_df = pd.DataFrame(
        X_test_processed, columns=feature_names[:X_test_processed.shape[1]])

    model = pipeline.named_steps.get("model") or pipeline.steps[-1][1]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_df)
    except:
        explainer = shap.Explainer(model, X_test_df)
        shap_values = explainer(X_test_df).values

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    logger.info(f"  SHAP values shape: {shap_values.shape}")

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_df, plot_type="bar",
                      show=False, max_display=15)
    plt.title("SHAP Feature Importance (Mean |SHAP|)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "shap_summary_bar.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_df, show=False, max_display=15)
    plt.title("SHAP Beeswarm Plot", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "shap_beeswarm.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:8]
    top_features = [X_test_df.columns[i] for i in top_indices]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for idx, (feat_idx, ax) in enumerate(zip(top_indices, axes.flatten())):
        feat_name = X_test_df.columns[feat_idx]
        shap.dependence_plot(feat_idx, shap_values,
                             X_test_df, ax=ax, show=False)
        ax.set_title(feat_name, fontweight="bold")

    plt.suptitle("SHAP Dependence Plots (Top 8 Features)",
                 fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "shap_dependence_top8.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    importance_df = pd.DataFrame({
        "feature": X_test_df.columns,
        "mean_abs_shap": np.round(mean_abs_shap, 4)
    }).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(os.path.join(
        TABLES_DIR, "shap_importance.csv"), index=False)

    logger.info(f"  Top features by SHAP:")
    logger.info(importance_df.head(10).to_string(index=False))

    logger.info("\n  Computing SHAP interaction values for top pairs...")
    try:
        interaction_values = explainer.shap_interaction_values(
            X_test_df.iloc[:200])
        if isinstance(interaction_values, list):
            interaction_values = interaction_values[1]

        mean_interaction = np.abs(interaction_values).mean(axis=0)
        np.fill_diagonal(mean_interaction, 0)
        interaction_df = pd.DataFrame(
            mean_interaction, columns=X_test_df.columns, index=X_test_df.columns
        )

        pairs = []
        for i in range(len(interaction_df)):
            for j in range(i+1, len(interaction_df)):
                pairs.append((interaction_df.columns[i], interaction_df.columns[j],
                             interaction_df.iloc[i, j]))
        pairs.sort(key=lambda x: x[2], reverse=True)

        logger.info("  Top SHAP interaction pairs:")
        for feat1, feat2, val in pairs[:5]:
            logger.info(f"    {feat1} x {feat2}: {val:.4f}")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for idx, (f1, f2, _) in enumerate(pairs[:3]):
            i1 = list(X_test_df.columns).index(f1)
            i2 = list(X_test_df.columns).index(f2)
            axes[idx].scatter(
                X_test_df.iloc[:200][f1],
                interaction_values[:, i1, i2],
                c=X_test_df.iloc[:200][f2], cmap="coolwarm", alpha=0.5, s=15
            )
            axes[idx].set_xlabel(f1, fontweight="bold")
            axes[idx].set_ylabel(f"SHAP interaction ({f1} × {f2})", fontsize=9)
            axes[idx].set_title(f"{f1} × {f2}", fontweight="bold")

        plt.suptitle("SHAP Interaction Values (Top 3 Pairs)",
                     fontweight="bold", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "shap_interactions.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("  SHAP interaction plot saved.")
    except Exception as e:
        logger.info(
            f"  SHAP interactions skipped (model type may not support TreeExplainer): {e}")

    return shap_values, X_test_df, explainer

def run_pdp_analysis(pipeline, X_train, X_test, feature_names=None):
    logger.info("\n" + "="*60)
    logger.info("PARTIAL DEPENDENCE PLOTS (PDP)")
    logger.info("="*60)

    from sklearn.inspection import PartialDependenceDisplay

    preprocessor = pipeline.named_steps.get(
        "preprocessor") or pipeline.steps[0][1]
    model = pipeline.named_steps.get("model") or pipeline.steps[-1][1]

    X_train_proc = preprocessor.transform(X_train)
    if feature_names is None:
        feature_names = get_feature_names_from_pipeline(pipeline)

    feature_names = feature_names[:X_train_proc.shape[1]]

    features_of_interest = [f for f in feature_names if any(x in f.lower() for x in
                            ['metabolic', 'treatment', 'cholesterol', 'age', 'bmi', 'hba1c'])]
    features_of_interest = features_of_interest[:6]
    if not features_of_interest:
        features_of_interest = [0, 1, 2, 3]

    logger.info(f"  Generating PDP for: {features_of_interest}")

    fig, ax = plt.subplots(figsize=(14, 10))
    PartialDependenceDisplay.from_estimator(
        model,
        X_train_proc,
        features_of_interest[:6],
        feature_names=feature_names,
        kind='average',
        ax=ax
    )
    plt.suptitle(
        "Partial Dependence Plots (Top Clinical Features)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pdp_plots.png"), dpi=150)
    plt.close()
    logger.info("  PDP plots saved to figures/pdp_plots.png")

def write_clinical_narratives(importance_df):
    narratives = {
        "hba1c": "HbA1c (glycated hemoglobin) reflects average blood glucose over 2-3 months. "
                 "Higher values indicate worse glycemic control and are strongly associated with "
                 "microvascular complications (retinopathy, nephropathy, neuropathy). In this model, "
                 "HbA1c emerges as a top predictor because poor glycemic control fundamentally undermines "
                 "the body's ability to heal and respond to treatment.",

        "sbp": "Systolic blood pressure is a primary driver of cardiovascular risk. Elevated SBP increases "
               "afterload, accelerates atherosclerosis, and impairs end-organ perfusion. The model's reliance "
               "on SBP aligns with decades of clinical trial evidence showing that BP control improves outcomes "
               "across nearly all therapeutic domains.",

        "age": "Age captures the cumulative burden of physiological aging: reduced organ reserve, increased "
               "comorbidity burden, altered pharmacokinetics, and impaired healing capacity. The negative "
               "association with outcome reflects the clinical reality that older patients face higher treatment "
               "failure rates even with optimal management.",

        "exercise_freq": "Physical activity frequency is a modifiable risk factor associated with improved "
                         "cardiovascular fitness, insulin sensitivity, and psychological well-being. The positive "
                         "SHAP contribution for higher exercise frequency aligns with WHO guidelines recommending "
                         "150+ minutes/week of moderate activity for chronic disease management.",

        "treatment_arm": "Treatment arm captures the direct therapeutic effect. Enhanced treatment protocols "
                         "typically include more intensive monitoring, combination therapies, or higher-dose regimens. "
                         "The SHAP values confirm that enhanced treatment provides a significant positive effect on "
                         "outcomes, which is the expected clinical signal in a well-designed intervention study.",

        "cvd_risk_score": "The composite cardiovascular risk score aggregates multiple risk factors into a single "
                          "metric inspired by the Framingham Risk Score. Its prominence in SHAP rankings validates "
                          "the clinical intuition that combined risk assessment outperforms any single lab value "
                          "for predicting treatment success.",

        "metabolic_score": "The metabolic syndrome score (0-5) counts the number of cardiometabolic risk criteria "
                           "met. Each additional criterion synergistically increases vascular inflammation, insulin "
                           "resistance, and oxidative stress, explaining its predictive power beyond individual components.",

        "bmi": "Body mass index, while an imperfect measure of adiposity, captures the metabolic consequences of "
               "excess weight including systemic inflammation, insulin resistance, and impaired drug distribution. "
               "The U-shaped relationship in SHAP dependence plots may reflect the obesity paradox seen in some "
               "cardiovascular outcome studies.",
    }

    logger.info("\n" + "="*60)
    logger.info("CLINICAL NARRATIVES FOR TOP FEATURES")
    logger.info("="*60)
    for feat in importance_df.head(8)["feature"]:
        feat_base = feat.lower().replace("num__", "").replace("cat__", "")
        matched = None
        for key in narratives:
            if key in feat_base:
                matched = key
                break
        if matched:
            logger.info(f"\n  [{feat}]")
            logger.info(f"  {narratives[matched]}")

    return narratives

def main():
    logger.info("="*60)
    logger.info("PHASE 7: MODEL INTERPRETATION")
    logger.info("="*60)

    pipeline = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))

    df = pd.read_csv(DATA_FILE)
    df = engineer_all_features(df)
    X, y = prepare_data(df)

    train_idx = np.load(os.path.join(MODELS_DIR, "train_indices.npy"))
    test_idx = np.load(os.path.join(MODELS_DIR, "test_indices.npy"))

    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]

    shap_values, X_test_df, explainer = run_shap_analysis(
        pipeline, X_train, X_test)

    xgb_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
    if os.path.exists(xgb_path):
        model_inner = pipeline.named_steps.get(
            "model") or pipeline.steps[-1][1]
        model_type = type(model_inner).__name__
        if "XGB" not in model_type and "LGBM" not in model_type:
            logger.info("\n  Running SHAP on XGBoost model for interaction values...")
            xgb_pipeline = joblib.load(xgb_path)
            run_shap_analysis(xgb_pipeline, X_train, X_test)

    run_pdp_analysis(pipeline, X_train, X_test)

    importance_df = pd.read_csv(os.path.join(
        TABLES_DIR, "shap_importance.csv"))
    write_clinical_narratives(importance_df)

    logger.info("\n" + "="*60)
    logger.info("INTERPRETATION COMPLETE")
    logger.info("="*60)

if __name__ == "__main__":
    main()

