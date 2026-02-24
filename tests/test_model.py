from src.feature_engineering import engineer_all_features, ENGINEERED_FEATURES
from src.config import MODELS_DIR, RANDOM_SEED
import os
import sys
import pytest
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

class TestModelLoading:

    def test_best_model_exists(self):

        path = os.path.join(MODELS_DIR, "best_model.pkl")
        assert os.path.exists(path), "best_model.pkl not found"

    def test_best_model_loads(self, best_model):

        assert best_model is not None

    def test_all_four_models_exist(self):

        for name in ["xgboost", "lightgbm", "logistic", "mlp"]:
            path = os.path.join(MODELS_DIR, f"{name}_model.pkl")
            assert os.path.exists(path), f"{name}_model.pkl not found"

    def test_indices_saved(self):

        assert os.path.exists(os.path.join(MODELS_DIR, "train_indices.npy"))
        assert os.path.exists(os.path.join(MODELS_DIR, "test_indices.npy"))

    def test_best_params_saved(self):

        assert os.path.exists(os.path.join(MODELS_DIR, "best_params.json"))

class TestModelPrediction:

    def test_predict_returns_array(self, best_model, engineered_df):

        from src.pipeline import prepare_data
        X, _ = prepare_data(engineered_df.head(10))
        probs = best_model.predict_proba(X)
        assert isinstance(probs, np.ndarray)

    def test_probabilities_in_range(self, best_model, engineered_df):

        from src.pipeline import prepare_data
        X, _ = prepare_data(engineered_df.head(50))
        probs = best_model.predict_proba(X)[:, 1]
        assert probs.min() >= 0.0, f"Min prob {probs.min()} below 0"
        assert probs.max() <= 1.0, f"Max prob {probs.max()} above 1"

    def test_prediction_shape(self, best_model, engineered_df):

        from src.pipeline import prepare_data
        n = 25
        X, _ = prepare_data(engineered_df.head(n))
        probs = best_model.predict_proba(X)
        assert probs.shape[0] == n

    def test_two_classes(self, best_model, engineered_df):

        from src.pipeline import prepare_data
        X, _ = prepare_data(engineered_df.head(10))
        probs = best_model.predict_proba(X)
        assert probs.shape[1] == 2

class TestModelPerformance:

    def test_f1_above_threshold(self, best_model, engineered_df):

        from sklearn.metrics import f1_score
        from src.pipeline import prepare_data

        test_idx = np.load(os.path.join(MODELS_DIR, "test_indices.npy"))
        df_test = engineered_df.iloc[test_idx]
        X, y = prepare_data(df_test)

        y_prob = best_model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        f1 = f1_score(y, y_pred)
        assert f1 >= 0.80, f"F1 = {f1:.4f}, below 0.80 threshold"

    def test_auc_above_threshold(self, best_model, engineered_df):

        from sklearn.metrics import roc_auc_score
        from src.pipeline import prepare_data

        test_idx = np.load(os.path.join(MODELS_DIR, "test_indices.npy"))
        df_test = engineered_df.iloc[test_idx]
        X, y = prepare_data(df_test)

        y_prob = best_model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_prob)
        assert auc >= 0.90, f"AUC = {auc:.4f}, below 0.90 threshold"

    def test_all_models_above_f1_threshold(self, all_models, engineered_df):

        from sklearn.metrics import f1_score
        from src.pipeline import prepare_data

        test_idx = np.load(os.path.join(MODELS_DIR, "test_indices.npy"))
        df_test = engineered_df.iloc[test_idx]
        X, y = prepare_data(df_test)

        import warnings
        for name, model in all_models.items():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
                y_prob = model.predict_proba(X)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            f1 = f1_score(y, y_pred)
            assert f1 >= 0.80, f"{name} F1 = {f1:.4f}, below 0.80"

