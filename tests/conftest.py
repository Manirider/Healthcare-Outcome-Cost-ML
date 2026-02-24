from src.config import (
    DATA_FILE, ENGINEERED_DATA_FILE, MODELS_DIR, RANDOM_SEED,
    N_SAMPLES, NUMERIC_FEATURES, CATEGORICAL_FEATURES
)
import os
import sys
import pytest
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

@pytest.fixture(scope="session")
def raw_df():

    if not os.path.exists(DATA_FILE):
        pytest.skip(
            f"Raw data not found at {DATA_FILE}. Run `python run_all.py` first.")
    return pd.read_csv(DATA_FILE)

@pytest.fixture(scope="session")
def engineered_df():

    if not os.path.exists(ENGINEERED_DATA_FILE):
        pytest.skip(
            f"Engineered data not found. Run `python run_all.py` first.")
    return pd.read_csv(ENGINEERED_DATA_FILE)

@pytest.fixture(scope="session")
def best_model():

    path = os.path.join(MODELS_DIR, "best_model.pkl")
    if not os.path.exists(path):
        pytest.skip(
            f"Model not found at {path}. Run `python run_all.py` first.")
    return joblib.load(path)

@pytest.fixture(scope="session")
def all_models():

    models = {}
    for name in ["xgboost", "lightgbm", "logistic", "mlp"]:
        path = os.path.join(MODELS_DIR, f"{name}_model.pkl")
        if os.path.exists(path):
            models[name] = joblib.load(path)
    if not models:
        pytest.skip("No models found. Run `python run_all.py` first.")
    return models

@pytest.fixture
def sample_patient():

    return pd.DataFrame({
        "age": [55], "sex": ["Male"], "height_cm": [175], "weight_kg": [82],
        "sbp": [135], "dbp": [85], "hba1c": [6.2], "ldl": [120],
        "heart_rate": [72], "creatinine": [1.1], "smoking_status": ["Never"],
        "exercise_freq": [3], "treatment_arm": ["Standard"],
        "treatment_cost": [5000], "outcome_binary": [1]
    })

