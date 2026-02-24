from src.config import METABOLIC_SYNDROME_THRESHOLDS
from src.feature_engineering import (
    ENGINEERED_FEATURES, engineer_all_features,
    compute_bmi, compute_map, compute_pulse_pressure,
    compute_metabolic_syndrome_score
)
import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

class TestFeatureCreation:

    def test_all_features_created(self, raw_df):

        df = engineer_all_features(raw_df.copy())
        for feat in ENGINEERED_FEATURES:
            assert feat in df.columns, f"Missing engineered feature: {feat}"

    def test_no_nans_introduced(self, raw_df):

        df = engineer_all_features(raw_df.copy())
        for feat in ENGINEERED_FEATURES:
            nan_count = df[feat].isna().sum()
            assert nan_count == 0, f"{feat} has {nan_count} NaNs"

    def test_output_shape(self, raw_df):

        original_cols = len(raw_df.columns)
        df = engineer_all_features(raw_df.copy())
        assert len(df) == len(raw_df), "Row count changed"
        assert len(df.columns) == original_cols + len(ENGINEERED_FEATURES),            f"Expected {original_cols + len(ENGINEERED_FEATURES)} cols, got {len(df.columns)}"

class TestBMI:

    def test_bmi_formula(self):

        df = pd.DataFrame({"weight_kg": [80.0], "height_cm": [180.0]})
        df = compute_bmi(df)
        expected = 80.0 / (1.80 ** 2)
        assert abs(df["bmi"].iloc[0] - expected) < 0.01

    def test_bmi_positive(self, raw_df):

        df = compute_bmi(raw_df.copy())
        assert (df["bmi"] > 0).all()

class TestMAP:

    def test_map_formula(self):

        df = pd.DataFrame({"sbp": [120.0], "dbp": [80.0]})
        df = compute_map(df)
        expected = 80.0 + (120.0 - 80.0) / 3
        assert abs(df["map_pressure"].iloc[0] - expected) < 0.01

class TestPulsePressure:

    def test_pulse_pressure_formula(self):

        df = pd.DataFrame({"sbp": [130.0], "dbp": [80.0]})
        df = compute_pulse_pressure(df)
        assert df["pulse_pressure"].iloc[0] == 50.0

class TestMetabolicScore:

    def test_score_range(self, raw_df):

        df = compute_bmi(raw_df.copy())

        df = compute_metabolic_syndrome_score(df)
        assert df["metabolic_score"].min() >= 0
        assert df["metabolic_score"].max() <= 5

    def test_perfect_health_scores_zero(self):

        df = pd.DataFrame({
            "sbp": [110.0], "dbp": [70.0], "hba1c": [5.0],
            "ldl": [90.0], "bmi": [22.0]
        })
        df = compute_metabolic_syndrome_score(df)
        assert df["metabolic_score"].iloc[0] == 0

    def test_unhealthy_scores_high(self):

        df = pd.DataFrame({
            "sbp": [160.0], "dbp": [100.0], "hba1c": [9.0],
            "ldl": [200.0], "bmi": [38.0]
        })
        df = compute_metabolic_syndrome_score(df)
        assert df["metabolic_score"].iloc[0] >= 3

