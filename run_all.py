import os
import sys
import time

import numpy as np
from src.logger import get_logger
logger = get_logger(__name__)
np.seterr(all='ignore')

os.environ["PYTHONUTF8"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_phase(name, func):
    logger.info(f"\n{'='*60}")
    logger.info(f"  PHASE: {name}")
    logger.info(f"{'='*60}")
    start = time.time()
    func()
    elapsed = time.time() - start
    logger.info(f"\n  [{name}] completed in {elapsed:.1f}s")
    return elapsed

def main():
    total_start = time.time()

    logger.info("="*60)
    logger.info("  HEALTHCARE OUTCOME ANALYSIS — FULL PIPELINE")
    logger.info("="*60)

    from src.generate_data import main as gen_main
    run_phase("Data Generation", gen_main)

    from src.eda import main as eda_main
    run_phase("Exploratory Data Analysis", eda_main)

    from src.hypothesis_tests import main as hyp_main
    run_phase("Statistical Hypothesis Testing", hyp_main)

    from src.feature_engineering import main as fe_main
    run_phase("Feature Engineering", fe_main)

    from src.train import main as train_main
    run_phase("Model Training & Evaluation", train_main)

    from src.interpretation import main as interp_main
    run_phase("Model Interpretation (SHAP + LIME)", interp_main)

    from src.simulate_drift import simulate_data_drift as drift_main
    run_phase("Data Drift Simulation", drift_main)

    from src.cost_effectiveness import main as cea_main
    run_phase("Cost-Effectiveness Analysis", cea_main)

    from src.generate_report import create_pdf as pdf_main
    run_phase("Executive Summary PDF", pdf_main)

    total = time.time() - total_start
    logger.info(f"\n{'='*60}")
    logger.info(
        f"  ALL PHASES COMPLETE — Total time: {total:.0f}s ({total/60:.1f}min)")
    logger.info(f"{'='*60}")
    logger.info("\n  Dashboard: streamlit run app.py")
    logger.info("  Report: outputs/reports/executive_summary.pdf")

if __name__ == "__main__":
    main()

