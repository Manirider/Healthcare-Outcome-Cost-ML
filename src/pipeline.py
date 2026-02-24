from src.feature_engineering import engineer_all_features, ENGINEERED_FEATURES
from src.config import (
    RANDOM_SEED, TEST_SIZE, CV_FOLDS, CV_REPEATS,
    OPTUNA_TRIALS, MODELS_DIR, TABLES_DIR
)
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from copy import deepcopy

from sklearn.model_selection import (
    train_test_split, RepeatedStratifiedKFold, cross_val_score, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
import optuna
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    brier_score_loss, make_scorer
)
from src.preprocessing_utils import Winsorizer

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC
from src.logger import get_logger
logger = get_logger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")
np.seterr(all='ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

NUMERIC_FEATURES = [
    "age", "height_cm", "weight_kg", "sbp", "dbp", "hba1c", "ldl",
    "heart_rate", "creatinine", "exercise_freq", "treatment_cost",
] + ENGINEERED_FEATURES

CATEGORICAL_FEATURES = ["sex", "smoking_status", "treatment_arm"]
TARGET = "outcome_binary"

def get_feature_names():
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES

def prepare_data(df):
    features = get_feature_names()
    X = df[features].copy()
    y = df[TARGET].copy()
    return X, y

def build_preprocessor(numeric_features=None, categorical_features=None):

    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    numeric_transformer = Pipeline(steps=[
        ("imputer", IterativeImputer(max_iter=10, random_state=RANDOM_SEED)),
        ("winsorizer", Winsorizer(lower_quantile=0.01, upper_quantile=0.99)),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first",
         sparse_output=False, handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )
    return preprocessor

def build_pipeline(model, preprocessor, use_smote=True):

    if use_smote:
        n_numeric = len(NUMERIC_FEATURES)
        pipeline = ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_SEED, k_neighbors=5)),
            ("model", model)
        ])
    else:
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])
    return pipeline

def xgboost_objective(trial, X, y, preprocessor):

    try:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
            "random_state": RANDOM_SEED,
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "verbosity": 0,
        }

        model = xgb.XGBClassifier(**params)
        pipeline = build_pipeline(model, deepcopy(preprocessor))

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                             random_state=RANDOM_SEED)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1", n_jobs=1)

        return scores.mean()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

def lightgbm_objective(trial, X, y, preprocessor):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "random_state": RANDOM_SEED,
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(**params)
    pipeline = build_pipeline(model, deepcopy(preprocessor))

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_SEED)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1", n_jobs=1)

    return scores.mean()

def logistic_objective(trial, X, y, preprocessor):

    params = {
        "C": trial.suggest_float("C", 1e-4, 100, log=True),
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 2000,
        "random_state": RANDOM_SEED,
    }

    model = LogisticRegression(**params)
    pipeline = build_pipeline(model, deepcopy(preprocessor))

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_SEED)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1", n_jobs=1)

    return scores.mean()

def mlp_objective(trial, X, y, preprocessor):

    hidden_1 = trial.suggest_int("hidden_1", 32, 256)
    hidden_2 = trial.suggest_int("hidden_2", 16, 128)
    params = {
        "hidden_layer_sizes": (hidden_1, hidden_2),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        "max_iter": 500,
        "early_stopping": True,
        "random_state": RANDOM_SEED,
    }

    model = MLPClassifier(**params)
    pipeline = build_pipeline(model, deepcopy(preprocessor))

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_SEED)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1", n_jobs=1)

    return scores.mean()

def run_optuna_search(name, objective_func, X, y, preprocessor, n_trials=None):

    if n_trials is None:
        n_trials = OPTUNA_TRIALS

    logger.info(f"\n  Optimizing {name} ({n_trials} trials)...")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(
        lambda trial: objective_func(trial, X, y, preprocessor),
        n_trials=n_trials,
        show_progress_bar=False
    )

    logger.info(f"  Best F1: {study.best_value:.4f}")
    logger.info(f"  Best params: {study.best_params}")

    return study

def train_final_model(model_class, best_params, X_train, y_train, preprocessor, use_smote=True):

    model = model_class(**best_params)
    pipeline = build_pipeline(model, deepcopy(
        preprocessor), use_smote=use_smote)
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_cv(pipeline, X, y, name="Model"):

    cv = RepeatedStratifiedKFold(
        n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_SEED
    )

    f1_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1", n_jobs=1)
    auc_scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=1)

    logger.info(
        f"\n  {name} CV Results ({CV_FOLDS}x{CV_REPEATS} Repeated Stratified K-Fold):")
    logger.info(f"    F1:  {f1_scores.mean():.4f} +/- {f1_scores.std():.4f}")
    logger.info(f"    AUC: {auc_scores.mean():.4f} +/- {auc_scores.std():.4f}")

    return {
        "model": name,
        "f1_mean": round(f1_scores.mean(), 4),
        "f1_std": round(f1_scores.std(), 4),
        "auc_mean": round(auc_scores.mean(), 4),
        "auc_std": round(auc_scores.std(), 4),
    }

