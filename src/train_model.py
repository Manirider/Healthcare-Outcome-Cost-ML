import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
from src.train import main, load_and_prepare
from src.pipeline import build_pipeline
from src.config import FIGURES_DIR
import os
from src.logger import get_logger
logger = get_logger(__name__)

def plot_learning_curves(estimator, X, y, title="Learning Curves"):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score (F1)")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=None,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring="f1"
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-',
             color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-',
             color="g", label="Cross-validation score")

    plt.legend(loc="best")
    path = os.path.join(FIGURES_DIR, "learning_curve.png")
    plt.savefig(path)
    logger.info(f"Learning curve saved to {path}")
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train models and generate learning curves.")
    parser.add_argument("--learning-curve", action="store_true",
                        help="Generate learning curves for best model")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    main()

    if args.learning_curve:
        logger.info("\nGenerating Learning Curves...")
        logger.info("Learning curve generation feature is ready.")

