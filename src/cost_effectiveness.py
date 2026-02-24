from src.pipeline import TARGET, prepare_data
from src.feature_engineering import engineer_all_features
from src.config import (
    RANDOM_SEED, DATA_FILE, MODELS_DIR, FIGURES_DIR, TABLES_DIR,
    COST_ASSUMPTIONS, WTP_THRESHOLDS, AGE_BINS, BMI_CATEGORIES
)
import seaborn as sns
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

def document_assumptions():
    logger.info("\n" + "="*60)
    logger.info("COST-EFFECTIVENESS ASSUMPTIONS (ALL SYNTHETIC)")
    logger.info("="*60)
    ca = COST_ASSUMPTIONS
    logger.info(f"""
    Treatment Costs:
      Standard treatment:     ${ca['treatment_cost_standard']:>10,} per patient
      Enhanced treatment:     ${ca['treatment_cost_enhanced']:>10,} per patient
      Control (monitoring):   ${ca['treatment_cost_control']:>10,} per patient

    Downstream Costs:
      Good outcome:           ${ca['cost_good_outcome']:>10,} per patient
      Bad outcome:            ${ca['cost_bad_outcome']:>10,} per patient

    QALYs:
      Good outcome:           {ca['qaly_good_outcome']:>10.2f}
      Bad outcome:            {ca['qaly_bad_outcome']:>10.2f}

    Economic Parameters:
      Discount rate:          {ca['discount_rate']*100:>9.0f}% per year
      Time horizon:           {ca['time_horizon_years']:>10} years

    WTP Thresholds:           {', '.join(f'${w:,}' for w in WTP_THRESHOLDS)} per QALY
    """)
    return ca

def compute_expected_costs(df, y_prob):
    ca = COST_ASSUMPTIONS

    expected_downstream = (
        y_prob * ca["cost_good_outcome"] +
        (1 - y_prob) * ca["cost_bad_outcome"]
    )

    baseline_expected = (
        df[TARGET].mean() * ca["cost_good_outcome"] +
        (1 - df[TARGET].mean()) * ca["cost_bad_outcome"]
    )

    potential_saving = (1 - y_prob) *        (ca["cost_bad_outcome"] - ca["cost_good_outcome"])

    expected_qaly = (
        y_prob * ca["qaly_good_outcome"] +
        (1 - y_prob) * ca["qaly_bad_outcome"]
    )

    return {
        "expected_downstream": expected_downstream,
        "potential_saving": potential_saving,
        "expected_qaly": expected_qaly,
        "baseline_expected": baseline_expected
    }

def compute_nmb(y_prob, treatment_cost, wtp_threshold):
    ca = COST_ASSUMPTIONS
    expected_qaly = (
        y_prob * ca["qaly_good_outcome"] +
        (1 - y_prob) * ca["qaly_bad_outcome"]
    )
    expected_downstream = (
        y_prob * ca["cost_good_outcome"] +
        (1 - y_prob) * ca["cost_bad_outcome"]
    )
    total_cost = treatment_cost + expected_downstream

    nmb = wtp_threshold * expected_qaly - total_cost
    return nmb

def compute_icer(df, y_prob):
    ca = COST_ASSUMPTIONS

    results = []

    for arm_compare, arm_ref in [("Enhanced", "Control"), ("Standard", "Control"), ("Enhanced", "Standard")]:
        mask_comp = df["treatment_arm"] == arm_compare
        mask_ref = df["treatment_arm"] == arm_ref

        if mask_comp.sum() < 10 or mask_ref.sum() < 10:
            continue

        cost_comp = df.loc[mask_comp, "treatment_cost"].mean() +            (y_prob[mask_comp] * ca["cost_good_outcome"] +
             (1 - y_prob[mask_comp]) * ca["cost_bad_outcome"]).mean()
        cost_ref = df.loc[mask_ref, "treatment_cost"].mean() +            (y_prob[mask_ref] * ca["cost_good_outcome"] +
             (1 - y_prob[mask_ref]) * ca["cost_bad_outcome"]).mean()

        qaly_comp = (y_prob[mask_comp] * ca["qaly_good_outcome"] +
                     (1 - y_prob[mask_comp]) * ca["qaly_bad_outcome"]).mean()
        qaly_ref = (y_prob[mask_ref] * ca["qaly_good_outcome"] +
                    (1 - y_prob[mask_ref]) * ca["qaly_bad_outcome"]).mean()

        delta_cost = cost_comp - cost_ref
        delta_qaly = qaly_comp - qaly_ref

        icer = delta_cost / delta_qaly if abs(delta_qaly) > 1e-6 else np.inf

        results.append({
            "comparison": f"{arm_compare} vs {arm_ref}",
            "cost_comparator": round(cost_comp, 2),
            "cost_reference": round(cost_ref, 2),
            "delta_cost": round(delta_cost, 2),
            "qaly_comparator": round(qaly_comp, 4),
            "qaly_reference": round(qaly_ref, 4),
            "delta_qaly": round(delta_qaly, 4),
            "icer": round(icer, 2) if not np.isinf(icer) else "Dominated/Inf"
        })

    return pd.DataFrame(results)

def tornado_sensitivity(df, y_prob, base_wtp=50_000):
    ca = COST_ASSUMPTIONS
    baseline_nmb = compute_nmb(
        y_prob, df["treatment_cost"].values, base_wtp).mean()

    params_to_vary = [
        ("cost_good_outcome", ca["cost_good_outcome"]),
        ("cost_bad_outcome", ca["cost_bad_outcome"]),
        ("qaly_good_outcome", ca["qaly_good_outcome"]),
        ("qaly_bad_outcome", ca["qaly_bad_outcome"]),
        ("treatment_cost_standard", ca["treatment_cost_standard"]),
        ("treatment_cost_enhanced", ca["treatment_cost_enhanced"]),
    ]

    tornado_data = []
    for param_name, base_value in params_to_vary:
        for multiplier, label in [(0.5, "Low (-50%)"), (1.5, "High (+50%)")]:
            ca_modified = COST_ASSUMPTIONS.copy()
            ca_modified[param_name] = base_value * multiplier

            expected_qaly = (
                y_prob * ca_modified.get("qaly_good_outcome", ca["qaly_good_outcome"]) +
                (1 - y_prob) *
                ca_modified.get("qaly_bad_outcome", ca["qaly_bad_outcome"])
            )
            expected_downstream = (
                y_prob * ca_modified.get("cost_good_outcome", ca["cost_good_outcome"]) +
                (1 - y_prob) *
                ca_modified.get("cost_bad_outcome", ca["cost_bad_outcome"])
            )
            total_cost = df["treatment_cost"].values + expected_downstream
            nmb_modified = (base_wtp * expected_qaly - total_cost).mean()

            tornado_data.append({
                "parameter": param_name,
                "scenario": label,
                "value": round(base_value * multiplier, 2),
                "nmb": round(nmb_modified, 2),
                "nmb_change": round(nmb_modified - baseline_nmb, 2)
            })

    tornado_df = pd.DataFrame(tornado_data)

    fig, ax = plt.subplots(figsize=(12, 8))
    params = tornado_df["parameter"].unique()
    y_positions = range(len(params))

    for i, param in enumerate(params):
        param_data = tornado_df[tornado_df["parameter"] == param]
        low_nmb = param_data[param_data["scenario"].str.contains(
            "Low")]["nmb_change"].values[0]
        high_nmb = param_data[param_data["scenario"].str.contains(
            "High")]["nmb_change"].values[0]

        left = min(low_nmb, high_nmb)
        width = abs(high_nmb - low_nmb)
        color = "#3498db" if high_nmb > low_nmb else "#e74c3c"

        ax.barh(i, width, left=left, height=0.6,
                color=color, edgecolor="gray", alpha=0.8)
        ax.text(left - 200, i, f"${param_data[param_data['scenario'].str.contains('Low')]['value'].values[0]:,.0f}",
                va="center", ha="right", fontsize=9)
        ax.text(left + width + 200, i, f"${param_data[param_data['scenario'].str.contains('High')]['value'].values[0]:,.0f}",
                va="center", ha="left", fontsize=9)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(params, fontsize=10)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Change in Net Monetary Benefit ($)", fontweight="bold")
    ax.set_title(
        f"Tornado Sensitivity Analysis (WTP=${base_wtp:,}/QALY)", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "tornado_sensitivity.png"), dpi=150)
    plt.close()

    tornado_df.to_csv(os.path.join(
        TABLES_DIR, "tornado_sensitivity.csv"), index=False)
    return tornado_df, baseline_nmb

def identify_actionable_segments(df, y_prob, nmb_values):
    logger.info("\n" + "="*60)
    logger.info("TOP ACTIONABLE SEGMENTS")
    logger.info("="*60)

    df_analysis = df.copy()
    df_analysis["pred_prob"] = y_prob
    df_analysis["nmb"] = nmb_values

    segments = []

    for arm in df_analysis["treatment_arm"].unique():
        mask = df_analysis["treatment_arm"] == arm
        seg = {
            "segment": f"Treatment: {arm}",
            "n": int(mask.sum()),
            "mean_pred_prob": round(df_analysis.loc[mask, "pred_prob"].mean(), 3),
            "mean_nmb": round(df_analysis.loc[mask, "nmb"].mean(), 2),
            "total_nmb": round(df_analysis.loc[mask, "nmb"].sum(), 2),
        }
        segments.append(seg)

    for lo, hi, label in AGE_BINS:
        mask = (df_analysis["age"] >= lo) & (df_analysis["age"] <= hi)
        if mask.sum() > 0:
            segments.append({
                "segment": f"Age: {label}",
                "n": int(mask.sum()),
                "mean_pred_prob": round(df_analysis.loc[mask, "pred_prob"].mean(), 3),
                "mean_nmb": round(df_analysis.loc[mask, "nmb"].mean(), 2),
                "total_nmb": round(df_analysis.loc[mask, "nmb"].sum(), 2),
            })

    df_analysis["risk_quartile"] = pd.qcut(1 - df_analysis["pred_prob"], 4,
                                           labels=["Low Risk Q1", "Q2", "Q3", "High Risk Q4"])
    for q in df_analysis["risk_quartile"].unique():
        mask = df_analysis["risk_quartile"] == q
        segments.append({
            "segment": f"Risk: {q}",
            "n": int(mask.sum()),
            "mean_pred_prob": round(df_analysis.loc[mask, "pred_prob"].mean(), 3),
            "mean_nmb": round(df_analysis.loc[mask, "nmb"].mean(), 2),
            "total_nmb": round(df_analysis.loc[mask, "nmb"].sum(), 2),
        })

    seg_df = pd.DataFrame(segments).sort_values("mean_nmb", ascending=False)
    seg_df.to_csv(os.path.join(
        TABLES_DIR, "actionable_segments.csv"), index=False)

    logger.info("\nTop 5 segments by mean NMB:")
    logger.info(seg_df.head(5).to_string(index=False))

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.RdYlGn((seg_df["mean_nmb"] - seg_df["mean_nmb"].min()) /
                           (seg_df["mean_nmb"].max() - seg_df["mean_nmb"].min() + 1e-6))
    ax.barh(seg_df["segment"], seg_df["mean_nmb"],
            color=colors, edgecolor="gray")
    ax.axvline(0, color="red", linewidth=1, linestyle="--")
    ax.set_xlabel("Mean Net Monetary Benefit ($)", fontweight="bold")
    ax.set_title(
        "Net Monetary Benefit by Patient Segment (WTP=$50,000/QALY)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "nmb_by_segment.png"), dpi=150)
    plt.close()

    return seg_df

def plot_nmb_curves(df, y_prob):
    ca = COST_ASSUMPTIONS

    fig, ax = plt.subplots(figsize=(10, 6))
    wtp_range = np.linspace(0, 200_000, 100)

    for arm in ["Enhanced", "Standard", "Control"]:
        mask = df["treatment_arm"] == arm
        if mask.sum() == 0:
            continue

        nmb_by_wtp = []
        for wtp in wtp_range:
            nmb = compute_nmb(
                y_prob[mask], df.loc[mask, "treatment_cost"].values, wtp)
            nmb_by_wtp.append(nmb.mean())

        ax.plot(wtp_range / 1000, nmb_by_wtp, linewidth=2, label=arm)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Willingness to Pay ($ thousands per QALY)",
                  fontweight="bold")
    ax.set_ylabel("Mean Net Monetary Benefit ($)", fontweight="bold")
    ax.set_title("Cost-Effectiveness Acceptability: NMB by WTP",
                 fontweight="bold", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "nmb_curves.png"), dpi=150)
    plt.close()

def main():
    logger.info("="*60)
    logger.info("PHASE 8: COST-EFFECTIVENESS ANALYSIS")
    logger.info("="*60)

    ca = document_assumptions()

    pipeline = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    df = pd.read_csv(DATA_FILE)
    df = engineer_all_features(df)
    X, y = prepare_data(df)

    y_prob = pipeline.predict_proba(X)[:, 1]

    cost_results = compute_expected_costs(df, y_prob)
    logger.info(
        f"\n  Mean expected downstream cost: ${cost_results['expected_downstream'].mean():,.2f}")
    logger.info(
        f"  Mean potential saving (targeted intervention): ${cost_results['potential_saving'].mean():,.2f}")
    logger.info(
        f"  Total potential savings (all patients): ${cost_results['potential_saving'].sum():,.2f}")

    default_wtp = 50_000
    nmb_values = compute_nmb(y_prob, df["treatment_cost"].values, default_wtp)
    logger.info(f"\n  Mean NMB (WTP=${default_wtp:,}): ${nmb_values.mean():,.2f}")

    logger.info("\n" + "="*60)
    logger.info("ICER ANALYSIS")
    logger.info("="*60)
    icer_df = compute_icer(df, y_prob)
    logger.info(icer_df.to_string(index=False))
    icer_df.to_csv(os.path.join(TABLES_DIR, "icer_analysis.csv"), index=False)

    tornado_df, baseline_nmb = tornado_sensitivity(
        df, y_prob, base_wtp=default_wtp)

    plot_nmb_curves(df, y_prob)

    seg_df = identify_actionable_segments(df, y_prob, nmb_values)

    logger.info("\n" + "="*60)
    logger.info("COST-EFFECTIVENESS ANALYSIS COMPLETE")
    logger.info("="*60)

if __name__ == "__main__":
    main()

