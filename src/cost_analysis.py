from src.cost_effectiveness import (
    main, compute_expected_costs, compute_nmb,
    compute_icer, tornado_sensitivity, identify_actionable_segments
)

from src.config import COST_ASSUMPTIONS, WTP_THRESHOLDS
from src.logger import get_logger
logger = get_logger(__name__)

def cost_per_successful_outcome(df):
    results = []
    for arm in df["treatment_arm"].unique():
        arm_df = df[df["treatment_arm"] == arm]
        total_cost = arm_df["treatment_cost"].sum()
        n_success = arm_df["outcome_binary"].sum()
        cost_per = total_cost / max(n_success, 1)
        results.append({
            "treatment_arm": arm,
            "total_patients": len(arm_df),
            "successful_outcomes": int(n_success),
            "success_rate": round(n_success / len(arm_df) * 100, 1),
            "total_cost": round(total_cost, 2),
            "cost_per_success": round(cost_per, 2)
        })
    return results

def savings_per_100_patients(df, model, threshold=0.5):
    from src.pipeline import prepare_data
    X, _ = prepare_data(df)
    probs = model.predict_proba(X)[:, 1]

    high_risk = probs < threshold
    n_total = len(df)

    current_cost = df["treatment_cost"].mean() * 100

    enhanced_cost = COST_ASSUMPTIONS["treatment_cost_enhanced"]
    standard_cost = COST_ASSUMPTIONS["treatment_cost_standard"]
    guided_cost = (high_risk.mean() * enhanced_cost +
                   (1 - high_risk.mean()) * standard_cost) * 100

    savings = current_cost - guided_cost

    return {
        "current_cost_per_100": round(current_cost, 2),
        "model_guided_cost_per_100": round(guided_cost, 2),
        "savings_per_100": round(savings, 2),
        "savings_pct": round(savings / current_cost * 100, 1)
    }

if __name__ == "__main__":
    main()

