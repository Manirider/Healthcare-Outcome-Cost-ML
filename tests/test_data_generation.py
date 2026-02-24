from src.config import N_SAMPLES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, RANDOM_SEED
import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

class TestDataSchema:

    def test_shape(self, raw_df):

        assert raw_df.shape[0] == N_SAMPLES, f"Expected {N_SAMPLES} rows, got {raw_df.shape[0]}"

    def test_numeric_columns_exist(self, raw_df):

        for col in NUMERIC_FEATURES:
            assert col in raw_df.columns, f"Missing numeric column: {col}"

    def test_categorical_columns_exist(self, raw_df):

        for col in CATEGORICAL_FEATURES:
            assert col in raw_df.columns, f"Missing categorical column: {col}"

    def test_target_column_exists(self, raw_df):

        assert "outcome_binary" in raw_df.columns

    def test_target_is_binary(self, raw_df):

        unique = set(raw_df["outcome_binary"].unique())
        assert unique.issubset(
            {0, 1}), f"Target has non-binary values: {unique}"

    def test_treatment_arms(self, raw_df):

        arms = set(raw_df["treatment_arm"].unique())
        assert arms == {"Standard", "Enhanced",
                        "Control"}, f"Unexpected arms: {arms}"

    def test_sex_categories(self, raw_df):

        assert set(raw_df["sex"].unique()).issubset({"Male", "Female"})

    def test_no_duplicate_ids(self, raw_df):

        if "patient_id" in raw_df.columns:
            assert raw_df["patient_id"].is_unique

class TestDataQuality:

    def test_age_range(self, raw_df):

        ages = raw_df["age"].dropna()
        assert ages.min() >= 18, f"Min age ({ages.min()}) below 18"
        assert ages.max() <= 100, f"Max age ({ages.max()}) above 100"

    def test_outcome_rate(self, raw_df):

        rate = raw_df["outcome_binary"].mean()
        assert 0.20 <= rate <= 0.45, f"Outcome rate {rate:.2%} outside expected bounds"

    def test_missingness_exists(self, raw_df):

        total_missing = raw_df.isnull().sum().sum()
        assert total_missing > 0, "No missing values found — MAR was not applied"

    def test_treatment_cost_positive(self, raw_df):

        if "treatment_cost" in raw_df.columns:
            assert (raw_df["treatment_cost"] > 0).all(
            ), "Negative treatment cost found"

    def test_clinical_ranges_reasonable(self, raw_df):

        if "sbp" in raw_df.columns:
            sbp = raw_df["sbp"].dropna()
            assert sbp.min() > 0, "SBP cannot be ≤ 0"
            assert sbp.max() < 300, "SBP suspiciously high"

class TestDataDeterminism:

    def test_seed_produces_same_output(self):

        from src.generate_data import generate_demographics

        np.random.seed(RANDOM_SEED)
        df1 = generate_demographics(100)

        np.random.seed(RANDOM_SEED)
        df2 = generate_demographics(100)

        pd.testing.assert_frame_equal(df1, df2)

