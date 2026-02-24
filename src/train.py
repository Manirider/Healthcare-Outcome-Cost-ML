from src.pipeline import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET,
    prepare_data, build_preprocessor, build_pipeline,
    run_optuna_search, train_final_model, evaluate_cv,
    xgboost_objective, lightgbm_objective, logistic_objective,
    mlp_objective
)
from src.feature_engineering import engineer_all_features, ENGINEERED_FEATURES
from src.config import (
    RANDOM_SEED, TEST_SIZE, DATA_FILE, MODELS_DIR,
    FIGURES_DIR, TABLES_DIR, AGE_BINS, BMI_CATEGORIES,
    OPTUNA_TRIALS
)
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    brier_score_loss, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.base import clone
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import matplotlib
from src.logger import get_logger
logger = get_logger(__name__)
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)

def load_and_prepare():
    logger.info("Loading and preparing data...")
    df = pd.read_csv(DATA_FILE)
    df = engineer_all_features(df)

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    logger.info(
        f"  Train: {X_train.shape[0]} samples (pos rate: {y_train.mean():.3f})")
    logger.info(
        f"  Test:  {X_test.shape[0]} samples (pos rate: {y_test.mean():.3f})")

    return df, X_train, X_test, y_train, y_test

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "f1": round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "auc_roc": round(roc_auc_score(y_true, y_prob), 4),
        "auc_pr": round(average_precision_score(y_true, y_prob), 4),
        "brier": round(brier_score_loss(y_true, y_prob), 4),
        "n": len(y_true),
    }

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return round(ece, 4)

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Poor (0)", "Good (1)"],
                yticklabels=["Poor (0)", "Good (1)"])
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("Actual", fontweight="bold")
    ax.set_title(f"Confusion Matrix - {model_name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(
        FIGURES_DIR, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"), dpi=150)
    plt.close()

def plot_calibration(y_true, y_prob_dict, filename="calibration_plot.png"):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", linewidth=1.5)

    for name, y_prob in y_prob_dict.items():
        fraction_pos, mean_predicted = calibration_curve(
            y_true, y_prob, n_bins=10)
        brier = brier_score_loss(y_true, y_prob)
        ax.plot(mean_predicted, fraction_pos, "o-",
                label=f"{name} (Brier={brier:.3f})", linewidth=2)

    ax.set_xlabel("Mean Predicted Probability", fontweight="bold")
    ax.set_ylabel("Fraction of Positives", fontweight="bold")
    ax.set_title("Calibration Plot", fontweight="bold", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150)
    plt.close()

def optimize_threshold(y_true, y_prob, cost_fp=7000, cost_fn=33000):
    best_thresh = 0.5
    min_cost = float('inf')

    thresholds = np.linspace(0.01, 0.99, 99)
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred_t)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            total_cost = (fp * cost_fp) + (fn * cost_fn)
            if total_cost < min_cost:
                min_cost = total_cost
                best_thresh = t

    return best_thresh

def plot_learning_curves(estimator, X, y, model_name):
    logger.info(f"\nGeneratng learning curves for {model_name}...")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=None,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring="f1"
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_mean, 'o-',
            color="#e74c3c", label="Training score")
    ax.plot(train_sizes, test_mean, 'o-', color="#2ecc71",
            label="Cross-validation score")

    ax.fill_between(train_sizes, train_mean - train_std,
                    train_mean + train_std, alpha=0.1, color="#e74c3c")
    ax.fill_between(train_sizes, test_mean - test_std,
                    test_mean + test_std, alpha=0.1, color="#2ecc71")

    ax.set_title(f"Learning Curve: {model_name}", fontweight="bold")
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("F1 Score")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "learning_curve.png"), dpi=150)
    plt.close()

def subgroup_analysis(y_true, y_pred, y_prob, df_test):
    logger.info("\n" + "="*60)
    logger.info("SUBGROUP PERFORMANCE ANALYSIS")
    logger.info("="*60)

    results = []

    for sex in ["Male", "Female"]:
        mask = df_test["sex"] == sex
        if mask.sum() > 10:
            m = compute_metrics(y_true[mask], y_pred[mask], y_prob[mask])
            m["subgroup"] = f"Sex: {sex}"
            results.append(m)
            logger.info(
                f"  Sex={sex}: F1={m['f1']:.3f}, AUC={m['auc_roc']:.3f}, Brier={m['brier']:.3f} (n={m['n']})")

    for lo, hi, label in AGE_BINS:
        mask = (df_test["age"] >= lo) & (df_test["age"] <= hi)
        if mask.sum() > 10:
            m = compute_metrics(y_true[mask], y_pred[mask], y_prob[mask])
            m["subgroup"] = f"Age: {label}"
            results.append(m)
            logger.info(
                f"  Age={label}: F1={m['f1']:.3f}, AUC={m['auc_roc']:.3f}, Brier={m['brier']:.3f} (n={m['n']})")

    for arm in ["Standard", "Enhanced", "Control"]:
        mask = df_test["treatment_arm"] == arm
        if mask.sum() > 10:
            m = compute_metrics(y_true[mask], y_pred[mask], y_prob[mask])
            m["subgroup"] = f"Treatment: {arm}"
            results.append(m)
            logger.info(
                f"  Treatment={arm}: F1={m['f1']:.3f}, AUC={m['auc_roc']:.3f}, Brier={m['brier']:.3f} (n={m['n']})")

    bmi = df_test["weight_kg"] / (df_test["height_cm"] / 100) ** 2
    for label, (lo, hi) in BMI_CATEGORIES.items():
        mask = (bmi >= lo) & (bmi < hi)
        if mask.sum() > 10:
            m = compute_metrics(y_true[mask], y_pred[mask], y_prob[mask])
            m["subgroup"] = f"BMI: {label}"
            results.append(m)
            logger.info(
                f"  BMI={label}: F1={m['f1']:.3f}, AUC={m['auc_roc']:.3f}, Brier={m['brier']:.3f} (n={m['n']})")

    logger.info("\n" + "-"*40)
    logger.info("FAIRNESS AUDIT (Max - Min F1 Gap)")
    logger.info("-" * 40)

    df_res = pd.DataFrame(results)

    df_res["category"] = df_res["subgroup"].apply(lambda x: x.split(":")[0])

    for cat in df_res["category"].unique():
        subset = df_res[df_res["category"] == cat]
        if len(subset) < 2:
            continue

        max_f1 = subset["f1"].max()
        min_f1 = subset["f1"].min()
        gap = max_f1 - min_f1
        gap_pct = (gap / max_f1) * 100 if max_f1 > 0 else 0

        status = "FAIL (>10%)" if gap_pct > 10 else "PASS"
        logger.info(f"  {cat:10s}: Gap={gap:.4f} ({gap_pct:.1f}%) -> {status}")

        if gap_pct > 10:
            logger.info(f"    WARNING: Significant disparity detected in {cat}!")

    sub_df = df_res.drop(columns=["category"])
    sub_df.to_csv(os.path.join(
        TABLES_DIR, "subgroup_performance.csv"), index=False)
    return sub_df

def ablation_study(X_train, X_test, y_train, y_test, best_xgb_params):
    logger.info("\n" + "="*60)
    logger.info("ABLATION STUDY")
    logger.info("="*60)

    from sklearn.linear_model import LogisticRegression

    preprocessor = build_preprocessor()
    full_model = xgb.XGBClassifier(**best_xgb_params)
    full_pipeline = build_pipeline(full_model, deepcopy(preprocessor))
    full_pipeline.fit(X_train, y_train)
    y_pred_full = full_pipeline.predict(X_test)
    baseline_f1 = f1_score(y_test, y_pred_full)
    logger.info(f"  Baseline F1 (all features): {baseline_f1:.4f}")

    ablation_results = [{"feature_removed": "NONE (baseline)", "f1": round(
        baseline_f1, 4), "f1_drop": 0.0}]

    for feat in ENGINEERED_FEATURES:
        if feat not in X_train.columns:
            continue

        X_train_abl = X_train.drop(columns=[feat])
        X_test_abl = X_test.drop(columns=[feat])

        num_feats = [f for f in NUMERIC_FEATURES if f != feat]

        prep_abl = build_preprocessor(numeric_features=num_feats)

        abl_model = xgb.XGBClassifier(**best_xgb_params)
        abl_pipeline = build_pipeline(abl_model, prep_abl)
        abl_pipeline.fit(X_train_abl, y_train)
        y_pred_abl = abl_pipeline.predict(X_test_abl)
        abl_f1 = f1_score(y_test, y_pred_abl)
        drop = baseline_f1 - abl_f1

        logger.info(f"  Remove {feat:30s}: F1={abl_f1:.4f}, drop={drop:+.4f}")
        ablation_results.append({
            "feature_removed": feat,
            "f1": round(abl_f1, 4),
            "f1_drop": round(drop, 4)
        })

    abl_df = pd.DataFrame(ablation_results).sort_values(
        "f1_drop", ascending=False)
    abl_df.to_csv(os.path.join(TABLES_DIR, "ablation_study.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    eng_only = abl_df[abl_df["feature_removed"] != "NONE (baseline)"]
    colors = ["#e74c3c" if d > 0 else "#2ecc71" for d in eng_only["f1_drop"]]
    ax.barh(eng_only["feature_removed"], eng_only["f1_drop"],
            color=colors, edgecolor="gray")
    ax.set_xlabel("F1 Drop (positive = feature helps)", fontweight="bold")
    ax.set_title("Ablation Study: F1 Drop When Feature Removed",
                 fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "ablation_study.png"), dpi=150)
    plt.close()

    return abl_df

def main():
    logger.info("="*60)
    logger.info("PHASE 5-6: MODEL TRAINING & EVALUATION")
    logger.info("="*60)

    df, X_train, X_test, y_train, y_test = load_and_prepare()
    preprocessor = build_preprocessor()

    df_eng = engineer_all_features(df)
    _, X_all, y_all = df_eng, *prepare_data(df_eng)
    df_test = df.iloc[X_test.index]

    logger.info("\n" + "="*60)
    logger.info("HYPERPARAMETER OPTIMIZATION")
    logger.info("="*60)

    xgb_study = run_optuna_search(
        "XGBoost", xgboost_objective, X_train, y_train, preprocessor, n_trials=OPTUNA_TRIALS)

    lgb_study = run_optuna_search(
        "LightGBM", lightgbm_objective, X_train, y_train, preprocessor, n_trials=OPTUNA_TRIALS // 2)

    lr_study = run_optuna_search(
        "Logistic Regression", logistic_objective, X_train, y_train, preprocessor, n_trials=max(1, OPTUNA_TRIALS // 4))

    mlp_study = run_optuna_search(
        "MLP", mlp_objective, X_train, y_train, preprocessor, n_trials=max(1, OPTUNA_TRIALS // 4))

    logger.info("\n" + "="*60)
    logger.info("TRAINING FINAL MODELS")
    logger.info("="*60)

    best_xgb_params = {**xgb_study.best_params, "random_state": RANDOM_SEED,
                       "eval_metric": "logloss", "use_label_encoder": False, "verbosity": 0}
    best_lgb_params = {**lgb_study.best_params,
                       "random_state": RANDOM_SEED, "verbose": -1}
    best_lr_params = {**lr_study.best_params, "penalty": "l2", "solver": "lbfgs",
                      "max_iter": 2000, "random_state": RANDOM_SEED}
    mlp_best = mlp_study.best_params
    best_mlp_params = {
        "hidden_layer_sizes": (mlp_best["hidden_1"], mlp_best["hidden_2"]),
        "activation": mlp_best["activation"],
        "alpha": mlp_best["alpha"],
        "learning_rate_init": mlp_best["learning_rate_init"],
        "max_iter": 500, "early_stopping": True, "random_state": RANDOM_SEED,
    }

    xgb_pipeline = train_final_model(
        xgb.XGBClassifier, best_xgb_params, X_train, y_train, preprocessor)
    xgb_cv = evaluate_cv(xgb_pipeline, X_train, y_train, "XGBoost")

    lgb_pipeline = train_final_model(
        lgb.LGBMClassifier, best_lgb_params, X_train, y_train, preprocessor)
    lgb_cv = evaluate_cv(lgb_pipeline, X_train, y_train, "LightGBM")

    from sklearn.linear_model import LogisticRegression
    lr_pipeline = train_final_model(
        LogisticRegression, best_lr_params, X_train, y_train, preprocessor)
    lr_cv = evaluate_cv(lr_pipeline, X_train, y_train, "Logistic Regression")

    mlp_pipeline = train_final_model(
        MLPClassifier, best_mlp_params, X_train, y_train, preprocessor)
    mlp_cv = evaluate_cv(mlp_pipeline, X_train, y_train, "MLP")

    logger.info("\n" + "="*60)
    logger.info("HOLD-OUT SET EVALUATION")
    logger.info("="*60)

    models = {
        "XGBoost": xgb_pipeline,
        "LightGBM": lgb_pipeline,
        "LogReg": lr_pipeline,
        "MLP": mlp_pipeline,
    }

    holdout_results = []
    y_prob_dict = {}

    for name, pipeline in models.items():
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_prob_dict[name] = y_prob

        metrics = compute_metrics(y_test.values, y_pred, y_prob)
        metrics["model"] = name
        metrics["ece"] = expected_calibration_error(y_test.values, y_prob)
        holdout_results.append(metrics)

        logger.info(f"\n  {name}:")
        logger.info(
            f"    F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
        logger.info(
            f"    AUC-ROC={metrics['auc_roc']:.4f}, AUC-PR={metrics['auc_pr']:.4f}")
        logger.info(f"    Brier={metrics['brier']:.4f}, ECE={metrics['ece']:.4f}")

        plot_confusion_matrix(y_test.values, y_pred, name)

    plot_calibration(y_test.values, y_prob_dict)

    best_model_name = max(holdout_results, key=lambda x: x["f1"])["model"]
    best_pipeline = models[best_model_name]
    best_y_prob = y_prob_dict[best_model_name]

    optimal_threshold = optimize_threshold(y_test.values, best_y_prob)
    logger.info(f"\n  COST-OPTIMIZED THRESHOLD: {optimal_threshold:.2f} (Default: 0.50)")
    best_y_pred = (best_y_prob >= optimal_threshold).astype(int)

    logger.info(f"\n  BEST MODEL: {best_model_name}")

    best_metrics = [m for m in holdout_results if m["model"]
                    == best_model_name][0]
    if best_metrics["brier"] > 0.12 or best_metrics["ece"] > 0.05:
        logger.info("  Applying isotonic calibration...")
        cal_pipeline = CalibratedClassifierCV(
            best_pipeline, method="isotonic", cv=5)
        cal_pipeline.fit(X_train, y_train)
        cal_prob = cal_pipeline.predict_proba(X_test)[:, 1]
        cal_brier = brier_score_loss(y_test, cal_prob)
        best_metrics["cal_brier"] = cal_brier
        logger.info(f"  Calibrated Brier: {cal_brier:.4f}")
        y_prob_dict[f"{best_model_name} (Calibrated)"] = cal_prob
        plot_calibration(y_test.values, y_prob_dict,
                         "calibration_with_recal.png")

        if cal_brier < best_metrics["brier"]:
            best_pipeline = cal_pipeline
            best_y_prob = cal_prob

    sub_df = subgroup_analysis(
        y_test.values, best_y_pred, best_y_prob, df_test
    )

    abl_df = ablation_study(X_train, X_test, y_train, y_test, best_xgb_params)

    plot_learning_curves(clone(best_pipeline), X_train,
                         y_train, best_model_name)

    logger.info("\n" + "="*60)
    logger.info("SAVING ARTIFACTS")
    logger.info("="*60)

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Healthcare_Outcome_Optimization")

    with mlflow.start_run(run_name=f"Final_{best_model_name}"):
        mlflow.log_param("best_model_name", best_model_name)

        if best_model_name == "XGBoost":
            mlflow.log_params(best_xgb_params)
        elif best_model_name == "LightGBM":
            mlflow.log_params(best_lgb_params)
        elif best_model_name == "Logistic Regression":
            mlflow.log_params(best_lr_params)
        else:
            mlflow.log_params(best_mlp_params)

        mlflow.log_metrics({
            "test_f1": best_metrics["f1"],
            "test_auc_roc": best_metrics["auc_roc"],
            "test_brier": best_metrics["brier"],
            "test_ece": best_metrics["ece"],
            "calibrated_brier": best_metrics.get("cal_brier", best_metrics["brier"]),
            "optimal_threshold": optimal_threshold
        })

        mlflow.sklearn.log_model(best_pipeline, artifact_path="best_model")

    joblib.dump(best_pipeline, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(xgb_pipeline, os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    joblib.dump(lgb_pipeline, os.path.join(MODELS_DIR, "lightgbm_model.pkl"))
    joblib.dump(lr_pipeline, os.path.join(MODELS_DIR, "logistic_model.pkl"))
    logger.info("  Models saved to models/")

    pd.DataFrame(holdout_results).to_csv(os.path.join(
        TABLES_DIR, "holdout_metrics.csv"), index=False)
    pd.DataFrame([xgb_cv, lgb_cv, lr_cv, mlp_cv]).to_csv(
        os.path.join(TABLES_DIR, "cv_results.csv"), index=False)

    joblib.dump(mlp_pipeline, os.path.join(MODELS_DIR, "mlp_model.pkl"))

    with open(os.path.join(MODELS_DIR, "best_params.json"), "w") as f:
        json.dump({
            "xgboost": best_xgb_params,
            "lightgbm": best_lgb_params,
            "logistic": best_lr_params,
            "mlp": {k: str(v) for k, v in best_mlp_params.items()},
            "best_model": best_model_name
        }, f, indent=2, default=str)

    np.save(os.path.join(MODELS_DIR, "train_indices.npy"), X_train.index.values)
    np.save(os.path.join(MODELS_DIR, "test_indices.npy"), X_test.index.values)

    logger.info("\n" + "="*60)
    logger.info(f"TRAINING COMPLETE - Best model: {best_model_name}")
    logger.info(f"Hold-out F1: {best_metrics['f1']:.4f}")
    logger.info(f"Hold-out Brier: {best_metrics['brier']:.4f}")
    logger.info("="*60)

    return best_pipeline, models, holdout_results, df, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    main()

