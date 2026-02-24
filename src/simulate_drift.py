import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_FILE, RANDOM_SEED
from src.logger import get_logger

logger = get_logger(__name__)

def simulate_data_drift():

    logger.info("="*60)
    logger.info("PRODUCTION DRIFT DETECTION SIMULATION")
    logger.info("="*60)

    try:
        df_reference = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        logger.error(f"Cannot find reference data at {DATA_FILE}.")
        return

    df_current = df_reference.sample(n=min(2000, len(df_reference)), random_state=RANDOM_SEED).copy()

    df_current["age"] = df_current["age"] + np.random.normal(5, 2, len(df_current))
    df_current["weight_kg"] = df_current["weight_kg"] * 1.05

    features_to_check = ["age", "weight_kg", "hba1c", "sbp", "ldl", "heart_rate"]

    drift_detected = False
    logger.info(f"Comparing Reference Data (n={len(df_reference)}) to Current Batch (n={len(df_current)})")
    logger.info("-" * 40)

    for feat in features_to_check:
        if feat in df_reference.columns and feat in df_current.columns:
            ref_data = df_reference[feat].dropna()
            cur_data = df_current[feat].dropna()

            statistic, p_value = stats.ks_2samp(ref_data, cur_data)

            if p_value < 0.05:
                drift_detected = True
                status = "DRIFT DETECTED"
                logger.warning(f"  {feat:15s}: {status} (KS stat: {statistic:.4f}, p-value: {p_value:.4e})")
            else:
                status = "STABLE"
                logger.info(f"  {feat:15s}: {status} (KS stat: {statistic:.4f}, p-value: {p_value:.4f})")

    logger.info("-" * 40)
    if drift_detected:
        logger.error("ALERT: Significant data drift detected in incoming batch. Model retraining recommended.")
    else:
        logger.info("SUCCESS: Data distributions are stable. Model inference can proceed safely.")

    logger.info("="*60)

if __name__ == "__main__":
    simulate_data_drift()

