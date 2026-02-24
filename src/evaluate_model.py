import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, brier_score_loss, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve

from src.config import (
    MODELS_DIR, FIGURES_DIR, TABLES_DIR, DATA_FILE,
    RANDOM_SEED, TEST_SIZE, AGE_BINS, BMI_CATEGORIES
)
from src.feature_engineering import engineer_all_features, ENGINEERED_FEATURES
from src.logger import get_logger
logger = get_logger(__name__)

def load_best_model():

    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No model found at {model_path}. Run train_model.py first.")
    return joblib.load(model_path)

def load_all_models():

    models = {}
    for name in ["xgboost", "lightgbm", "logistic", "mlp"]:
        path = os.path.join(MODELS_DIR, f"{name}_model.pkl")
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

def compute_metrics(y_true, y_pred, y_prob):

    metrics = {
        "f1": round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "auc_roc": round(roc_auc_score(y_true, y_prob), 4),
        "auc_pr": round(average_precision_score(y_true, y_prob), 4),
        "brier": round(brier_score_loss(y_true, y_prob), 4),
    }

    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=10, strategy="uniform")
    bin_weights = np.histogram(y_prob, bins=10, range=(0, 1))[0] / len(y_prob)
    ece = np.sum(np.abs(prob_true - prob_pred) * bin_weights[:len(prob_true)])
    metrics["ece"] = round(ece, 4)

    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name, save_dir=None):

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Poor Outcome", "Good Outcome"],
                yticklabels=["Poor Outcome", "Good Outcome"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    plt.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(
            save_dir, f"confusion_matrix_{model_name.lower()}.png"), dpi=150)
    plt.close()
    return cm

def plot_roc_curve(y_true, y_prob, model_name, save_dir=None):

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(
            save_dir, f"roc_curve_{model_name.lower()}.png"), dpi=150)
    plt.close()
    return auc

def subgroup_analysis(y_true, y_pred, y_prob, df_test, model_name="Best"):

    results = []

    results = []

    for sex in df_test["sex"].unique():
        mask = df_test["sex"] == sex
        if mask.sum() > 20:
            metrics = compute_metrics(y_true[mask], y_pred[mask], y_prob[mask])
            metrics["subgroup"] = f"Sex: {sex}"
            metrics["n"] = int(mask.sum())
            results.append(metrics)

            results.append(metrics)

    for lo, hi, label in AGE_BINS:
        mask = (df_test["age"] >= lo) & (df_test["age"] <= hi)
        if mask.sum() > 20:
            metrics = compute_metrics(y_true[mask], y_pred[mask], y_prob[mask])
            metrics["subgroup"] = f"Age: {label}"
            metrics["n"] = int(mask.sum())
            results.append(metrics)

            results.append(metrics)

    for arm in df_test["treatment_arm"].unique():
        mask = df_test["treatment_arm"] == arm
        if mask.sum() > 20:
            metrics = compute_metrics(y_true[mask], y_pred[mask], y_prob[mask])
            metrics["subgroup"] = f"Treatment: {arm}"
            metrics["n"] = int(mask.sum())
            results.append(metrics)

    return pd.DataFrame(results)

def analyze_overfitting(model, X_train, y_train, X_test, y_test, threshold=0.05):

    logger.info("\n" + "="*60)
    logger.info("OVERFITTING ANALYSIS (Train vs Test)")
    logger.info("="*60)

    logger.info("="*60)

    y_pred_train = model.predict(X_train)
    f1_train = f1_score(y_train, y_pred_train)
    auc_train = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])

    auc_train = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])

    y_pred_test = model.predict(X_test)
    f1_test = f1_score(y_test, y_pred_test)
    auc_test = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    drop_f1 = f1_train - f1_test
    drop_auc = auc_train - auc_test

    logger.info(
        f"  Train F1: {f1_train:.4f}  |  Test F1: {f1_test:.4f}  |  Drop: {drop_f1:+.4f}")
    logger.info(
        f"  Train AUC: {auc_train:.4f} |  Test AUC: {auc_test:.4f} |  Drop: {drop_auc:+.4f}")

    if drop_f1 > threshold:
        logger.info(
            f"  WARNING: Significant overfitting detected (> {threshold*100:.0f}% drop in F1).")
        logger.info("  Recommendation: Increase regularization (alpha/lambda), reduce tree depth, or add dropout.")
    else:
        logger.info("  STATUS: Good generalization (no significant overfitting).")

    return {"train_f1": f1_train, "test_f1": f1_test, "drop": drop_f1}

def generate_evaluation_report(save_dir=None):

    if save_dir is None:
        save_dir = FIGURES_DIR

    model = load_best_model()
    df = pd.read_csv(DATA_FILE)
    df = engineer_all_features(df)

    df = engineer_all_features(df)

    test_idx = np.load(os.path.join(MODELS_DIR, "test_indices.npy"))
    train_idx = np.load(os.path.join(MODELS_DIR, "train_indices.npy"))

    df_test = df.iloc[test_idx]
    df_train = df.iloc[train_idx]

    from src.pipeline import prepare_data
    X_test, y_test = prepare_data(df_test)
    X_train, y_train = prepare_data(df_train)

    X_test, y_test = prepare_data(df_test)
    X_train, y_train = prepare_data(df_train)

    analyze_overfitting(model, X_train, y_train, X_test, y_test)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    y_pred = (y_prob >= 0.5).astype(int)

    metrics = compute_metrics(y_test, y_pred, y_prob)
    logger.info(f"\nBest Model Evaluation:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    plot_confusion_matrix(y_test, y_pred, "Best", save_dir)
    plot_roc_curve(y_test, y_prob, "Best", save_dir)

    logger.info(
        f"\n{classification_report(y_test, y_pred, target_names=['Poor', 'Good'])}")

    return metrics

if __name__ == "__main__":
    metrics = generate_evaluation_report()
    logger.info("\nEvaluation complete.")

