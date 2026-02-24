from src.config import (
    DATA_FILE, FIGURES_DIR, TABLES_DIR, CLINICAL_RANGES,
    AGE_BINS, BMI_CATEGORIES, RANDOM_SEED
)
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
from src.logger import get_logger
logger = get_logger(__name__)
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

CONTINUOUS_FEATURES = ["age", "height_cm", "weight_kg", "sbp", "dbp",
                       "hba1c", "ldl", "heart_rate", "creatinine",
                       "exercise_freq", "treatment_cost", "downstream_cost"]
CATEGORICAL_FEATURES = ["sex", "smoking_status", "treatment_arm"]
TARGET = "outcome_binary"

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    logger.info(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df

def analyze_missingness(df: pd.DataFrame) -> None:
    logger.info("\n" + "="*60)
    logger.info("MISSINGNESS ANALYSIS")
    logger.info("="*60)

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "count": missing,
        "percent": missing_pct
    }).query("count > 0").sort_values("percent", ascending=False)

    logger.info(missing_df.to_string())

    logger.info("\n--- MAR Mechanism Suspicion ---")
    for col in missing_df.index:
        is_missing = df[col].isnull().astype(int)

        if col != "age":
            age_missing = df.loc[is_missing == 1, "age"]
            age_present = df.loc[is_missing == 0, "age"]
            if len(age_missing) > 5:
                stat, p = stats.mannwhitneyu(
                    age_missing, age_present, alternative="two-sided")
                marker = "*** MAR (age-dep)" if p < 0.05 else "MCAR (no age assoc)"
                logger.info(f"  {col}: Missing ages mean={age_missing.mean():.1f} vs Present mean={age_present.mean():.1f}, "
                      f"p={p:.4f} -> {marker}")

    fig, ax = plt.subplots(figsize=(12, 3))
    missing_matrix = df[missing_df.index].isnull().astype(int).T
    sns.heatmap(missing_matrix.iloc[:, :200], cbar=False, yticklabels=True,
                cmap="YlOrRd", ax=ax)
    ax.set_title("Missing Data Pattern (first 200 observations)",
                 fontweight="bold")
    ax.set_xlabel("Observation Index")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "missingness_heatmap.png"), dpi=150)
    plt.close()

    missing_df.to_csv(os.path.join(TABLES_DIR, "missingness_summary.csv"))

def analyze_target(df: pd.DataFrame) -> None:
    logger.info("\n" + "="*60)
    logger.info("TARGET DISTRIBUTION")
    logger.info("="*60)

    outcome_rate = df[TARGET].mean()
    logger.info(
        f"Overall outcome rate: {outcome_rate:.4f} ({df[TARGET].sum()}/{len(df)})")
    logger.info(f"Class ratio (neg:pos): {1-outcome_rate:.3f}:{outcome_rate:.3f}")

    subgroup_results = []

    logger.info("\n--- By Treatment Arm ---")
    for arm, grp in df.groupby("treatment_arm"):
        rate = grp[TARGET].mean()
        n = len(grp)
        logger.info(f"  {arm}: {rate:.3f} (n={n})")
        subgroup_results.append(
            {"subgroup": f"Treatment: {arm}", "n": n, "outcome_rate": rate})

    logger.info("\n--- By Sex ---")
    for sex, grp in df.groupby("sex"):
        rate = grp[TARGET].mean()
        n = len(grp)
        logger.info(f"  {sex}: {rate:.3f} (n={n})")
        subgroup_results.append(
            {"subgroup": f"Sex: {sex}", "n": n, "outcome_rate": rate})

    logger.info("\n--- By Age Group ---")
    for lo, hi, label in AGE_BINS:
        grp = df[(df["age"] >= lo) & (df["age"] <= hi)]
        rate = grp[TARGET].mean()
        n = len(grp)
        logger.info(f"  {label} ({lo}-{hi}): {rate:.3f} (n={n})")
        subgroup_results.append(
            {"subgroup": f"Age: {label}", "n": n, "outcome_rate": rate})

    logger.info("\n--- By Smoking Status ---")
    for status, grp in df.groupby("smoking_status"):
        rate = grp[TARGET].mean()
        n = len(grp)
        logger.info(f"  {status}: {rate:.3f} (n={n})")
        subgroup_results.append(
            {"subgroup": f"Smoking: {status}", "n": n, "outcome_rate": rate})

    logger.info("\n--- By BMI Category ---")
    bmi = df["weight_kg"] / (df["height_cm"] / 100) ** 2
    for label, (lo, hi) in BMI_CATEGORIES.items():
        grp = df[(bmi >= lo) & (bmi < hi)]
        if len(grp) > 0:
            rate = grp[TARGET].mean()
            n = len(grp)
            logger.info(f"  {label}: {rate:.3f} (n={n})")
            subgroup_results.append(
                {"subgroup": f"BMI: {label}", "n": n, "outcome_rate": rate})

    sub_df = pd.DataFrame(subgroup_results)
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.RdYlGn(sub_df["outcome_rate"].values)
    bars = ax.barh(sub_df["subgroup"], sub_df["outcome_rate"],
                   color=colors, edgecolor="gray")
    ax.axvline(outcome_rate, color="red", linestyle="--",
               linewidth=1.5, label=f"Overall: {outcome_rate:.3f}")
    ax.set_xlabel("Outcome Rate (Treatment Success)", fontweight="bold")
    ax.set_title("Outcome Rate by Clinical Subgroup",
                 fontweight="bold", fontsize=14)
    ax.legend(fontsize=12)
    for bar, n in zip(bars, sub_df["n"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"n={n}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "outcome_by_subgroup.png"), dpi=150)
    plt.close()

    sub_df.to_csv(os.path.join(
        TABLES_DIR, "subgroup_outcome_rates.csv"), index=False)

def detect_clinical_outliers(df: pd.DataFrame) -> None:
    logger.info("\n" + "="*60)
    logger.info("CLINICAL OUTLIER DETECTION")
    logger.info("="*60)

    outlier_results = []
    for col, (lo, hi) in CLINICAL_RANGES.items():
        if col in df.columns:
            below = (df[col] < lo).sum()
            above = (df[col] > hi).sum()
            total_outlier = below + above
            pct = total_outlier / df[col].notna().sum() * 100
            logger.info(f"  {col}: {total_outlier} outliers ({pct:.2f}%) "
                  f"[below {lo}: {below}, above {hi}: {above}]")

            p01 = df[col].quantile(0.01)
            p99 = df[col].quantile(0.99)
            logger.info(
                f"    -> Recommended Winsorization bounds (1st/99th): [{p01:.2f}, {p99:.2f}]")

            outlier_results.append({
                "feature": col, "low_threshold": lo, "high_threshold": hi,
                "below": below, "above": above, "total": total_outlier, "pct": round(pct, 2),
                "p01": round(p01, 2), "p99": round(p99, 2)
            })

    bmi = df["weight_kg"] / (df["height_cm"] / 100) ** 2
    extreme_bmi = ((bmi < 15) | (bmi > 50)).sum()
    logger.info(f"  BMI (derived): {extreme_bmi} extreme values (BMI < 15 or > 50)")

    pp = df["sbp"] - df["dbp"]
    narrow_pp = (pp < 20).sum()
    wide_pp = (pp > 100).sum()
    logger.info(f"  Pulse pressure: {narrow_pp} narrow (<20), {wide_pp} wide (>100)")

    pd.DataFrame(outlier_results).to_csv(os.path.join(
        TABLES_DIR, "clinical_outliers.csv"), index=False)

def analyze_correlations(df: pd.DataFrame) -> None:
    logger.info("\n" + "="*60)
    logger.info("CORRELATION ANALYSIS")
    logger.info("="*60)

    numeric_cols = [c for c in CONTINUOUS_FEATURES if c in df.columns]
    numeric_cols.append(TARGET)

    corr = df[numeric_cols].corr()
    logger.info("\nTop correlations with outcome:")
    target_corr = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False)
    logger.info(target_corr.round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=axes[0], square=True, linewidths=0.5)
    axes[0].set_title("Pearson Correlation Matrix",
                      fontweight="bold", fontsize=13)

    df_mi = df[numeric_cols].dropna()
    X_mi = df_mi.drop(columns=[TARGET])
    y_mi = df_mi[TARGET]
    mi_scores = mutual_info_classif(
        X_mi, y_mi, random_state=RANDOM_SEED, n_neighbors=5)
    mi_df = pd.DataFrame({"feature": X_mi.columns, "MI": mi_scores}).sort_values(
        "MI", ascending=True)

    axes[1].barh(mi_df["feature"], mi_df["MI"],
                 color=sns.color_palette("viridis", len(mi_df)))
    axes[1].set_title("Mutual Information with Outcome",
                      fontweight="bold", fontsize=13)
    axes[1].set_xlabel("Mutual Information (bits)")

    plt.tight_layout()
    plt.savefig(os.path.join(
        FIGURES_DIR, "correlation_mi_heatmap.png"), dpi=150)
    plt.close()

    mi_df.to_csv(os.path.join(
        TABLES_DIR, "mutual_information.csv"), index=False)

def plot_outcome_stratified(df: pd.DataFrame) -> None:
    logger.info("\n" + "="*60)
    logger.info("OUTCOME-STRATIFIED DISTRIBUTIONS")
    logger.info("="*60)

    cont_features = [c for c in CONTINUOUS_FEATURES if c in df.columns and c not in [
        "treatment_cost", "downstream_cost"]]

    n_features = len(cont_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cont_features):
        ax = axes[i]
        for outcome, label, color in [(0, "Poor Outcome", "#e74c3c"), (1, "Good Outcome", "#2ecc71")]:
            data = df.loc[df[TARGET] == outcome, col].dropna()
            data.plot.kde(ax=ax, label=label, color=color, linewidth=2)
        ax.set_title(col, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_ylabel("Density")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Outcome-Stratified Kernel Density Estimates",
                 fontweight="bold", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "kde_by_outcome.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cont_features):
        ax = axes[i]
        sns.violinplot(data=df, x=TARGET, y=col, ax=ax, palette=["#e74c3c", "#2ecc71"],
                       inner="quartile", cut=0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Poor (0)", "Good (1)"])
        ax.set_title(col, fontweight="bold")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Outcome-Stratified Violin Plots",
                 fontweight="bold", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "violin_by_outcome.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("  -> KDE and violin plots saved.")

def plot_distributions(df: pd.DataFrame) -> None:

    cont = [c for c in CONTINUOUS_FEATURES if c in df.columns]

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    for i, col in enumerate(cont):
        if i < len(axes):
            df[col].dropna().hist(bins=40, ax=axes[i], color="#3498db",
                                  edgecolor="white", alpha=0.8)
            axes[i].set_title(col, fontweight="bold")
            axes[i].axvline(df[col].mean(), color="red",
                            linestyle="--", linewidth=1)

    for j in range(len(cont), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions (red line = mean)",
                 fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(
        FIGURES_DIR, "feature_distributions.png"), dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(CATEGORICAL_FEATURES):
        df[col].value_counts().plot.bar(ax=axes[i], color=sns.color_palette("Set2"),
                                        edgecolor="gray")
        axes[i].set_title(col, fontweight="bold")
        axes[i].tick_params(axis="x", rotation=45)

    plt.suptitle("Categorical Feature Distributions",
                 fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(
        FIGURES_DIR, "categorical_distributions.png"), dpi=150)
    plt.close()

def main():

    logger.info("="*60)
    logger.info("PHASE 2: EXPLORATORY DATA ANALYSIS")
    logger.info("="*60)

    df = load_data()

    logger.info(f"\nDataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"\nDescriptive statistics:")
    logger.info(df.describe().round(2).to_string())

    analyze_missingness(df)
    analyze_target(df)
    detect_clinical_outliers(df)
    analyze_correlations(df)
    plot_outcome_stratified(df)
    plot_distributions(df)

    try:
        import sweetviz as sv
        report = sv.analyze(df, target_feat=TARGET)
        report.show_html(
            os.path.join(FIGURES_DIR, "sweetviz_report.html"),
            open_browser=False
        )
        logger.info("\n  -> Sweetviz report saved.")
    except Exception as e:
        logger.info(f"\n  [!] Sweetviz report skipped: {e}")

    logger.info("\n" + "="*60)
    logger.info("EDA COMPLETE — all figures saved to outputs/figures/")
    logger.info("="*60)

if __name__ == "__main__":
    main()

