from src.config import (
    DATA_FILE, ENGINEERED_DATA_FILE, TABLES_DIR, METABOLIC_SYNDROME_THRESHOLDS, RANDOM_SEED
)
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from src.logger import get_logger
logger = get_logger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def compute_bmi(df):
    if "weight_kg" in df.columns and "height_cm" in df.columns:
        df["bmi"] = np.round(df["weight_kg"] / (df["height_cm"] / 100) ** 2, 2)
    return df

def compute_map(df):
    df["map_pressure"] = np.round(df["dbp"] + (df["sbp"] - df["dbp"]) / 3, 2)
    return df

def compute_pulse_pressure(df):
    df["pulse_pressure"] = df["sbp"] - df["dbp"]
    return df

def compute_metabolic_syndrome_score(df):
    thresholds = METABOLIC_SYNDROME_THRESHOLDS
    score = (
        (df["sbp"] >= thresholds["sbp"]).astype(int) +
        (df["dbp"] >= thresholds["dbp"]).astype(int) +
        (df["hba1c"] >= thresholds["hba1c"]).astype(int) +
        (df["bmi"] >= thresholds["bmi"]).astype(int)
    )
    ldl_component = (df["ldl"] >= thresholds["ldl"]).astype(float)
    ldl_component = ldl_component.fillna(0)
    score = score + ldl_component

    df["metabolic_score"] = score.astype(int)
    return df

def compute_age_hba1c_interaction(df):
    age_z = (df["age"] - df["age"].mean()) / df["age"].std()
    hba1c_z = (df["hba1c"] - df["hba1c"].mean()) / df["hba1c"].std()
    df["age_hba1c_interaction"] = np.round(age_z * hba1c_z, 4)
    return df

def compute_cvd_risk_composite(df):
    age_z = (df["age"] - df["age"].mean()) / df["age"].std()
    sbp_z = (df["sbp"] - df["sbp"].mean()) / df["sbp"].std()
    bmi_z = (df["bmi"] - df["bmi"].mean()) / df["bmi"].std()
    hba1c_z = (df["hba1c"] - df["hba1c"].mean()) / df["hba1c"].std()

    ldl_filled = df["ldl"].fillna(df["ldl"].median())
    ldl_z = (ldl_filled - ldl_filled.mean()) / ldl_filled.std()

    smoke_score = df["smoking_status"].map(
        {"Never": 0, "Former": 0.4, "Current": 1.0})

    df["cvd_risk_score"] = np.round(
        0.20 * age_z +
        0.25 * sbp_z +
        0.15 * ldl_z +
        0.20 * smoke_score +
        0.10 * bmi_z +
        0.10 * hba1c_z, 4
    )
    return df

def compute_treatment_intensity(df):
    risk_pctile = df["cvd_risk_score"].rank(pct=True)
    df["treatment_intensity_ratio"] = np.round(
        df["treatment_cost"] / (1 + risk_pctile), 2
    )
    return df

def compute_exercise_deficit(df):
    ex_freq = df["exercise_freq"].fillna(df["exercise_freq"].median())
    deficit = np.maximum(0, 3 - ex_freq)
    df["exercise_deficit"] = np.round(deficit * (df["bmi"] / 25), 2)
    return df

def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_bmi(df)
    df = compute_map(df)
    df = compute_pulse_pressure(df)
    df = compute_metabolic_syndrome_score(df)
    df = compute_age_hba1c_interaction(df)
    df = compute_cvd_risk_composite(df)
    df = compute_treatment_intensity(df)
    df = compute_exercise_deficit(df)
    return df

ENGINEERED_FEATURES = [
    "bmi", "map_pressure", "pulse_pressure", "metabolic_score",
    "age_hba1c_interaction", "cvd_risk_score",
    "treatment_intensity_ratio", "exercise_deficit"
]

def univariate_analysis(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("\n" + "="*60)
    logger.info("UNIVARIATE ANALYSIS OF ENGINEERED FEATURES")
    logger.info("="*60)

    results = []

    target = df["outcome_binary"]

    for feat in ENGINEERED_FEATURES:
        if feat not in df.columns:
            continue

        values = df[feat].dropna()
        target_aligned = target.loc[values.index]

        rho, p_spearman = stats.spearmanr(values, target_aligned)

        rpb, p_pb = stats.pointbiserialr(target_aligned, values)

        try:
            auc = roc_auc_score(target_aligned, values)

            if auc < 0.5:
                auc = 1 - auc
        except:
            auc = 0.5

        result = {
            "feature": feat,
            "spearman_rho": round(rho, 4),
            "spearman_p": p_spearman,
            "point_biserial_r": round(rpb, 4),
            "point_biserial_p": p_pb,
            "univariate_auc": round(auc, 4)
        }
        results.append(result)

        logger.info(f"  {feat:30s}  rho={rho:+.3f}  rpb={rpb:+.3f}  AUC={auc:.3f}")

    results_df = pd.DataFrame(results).sort_values(
        "univariate_auc", ascending=False)
    results_df.to_csv(os.path.join(
        TABLES_DIR, "feature_univariate_analysis.csv"), index=False)
    return results_df

def main():
    logger.info("="*60)
    logger.info("PHASE 4: FEATURE ENGINEERING")
    logger.info("="*60)

    df = pd.read_csv(DATA_FILE)
    logger.info(f"Original shape: {df.shape}")

    df = engineer_all_features(df)
    logger.info(f"After engineering: {df.shape}")
    logger.info(f"New features: {ENGINEERED_FEATURES}")

    uni_df = univariate_analysis(df)

    df.to_csv(ENGINEERED_DATA_FILE, index=False)
    logger.info(f"\nEngineered dataset saved to {ENGINEERED_DATA_FILE}")

    logger.info("\n" + "="*60)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("="*60)

    return df

if __name__ == "__main__":
    main()

