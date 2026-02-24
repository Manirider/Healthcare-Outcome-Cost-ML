from src.config import CLINICAL_RANGES, NUMERIC_FEATURES
from src.preprocessing_utils import Winsorizer
from src.preprocessing import (
    handle_missing_values, detect_outliers_iqr, treat_outliers
)
import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

class TestMICEImputation:

    def test_no_nans_after_imputation(self, raw_df):

        df = handle_missing_values(raw_df.copy())
        numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        for col in numeric_cols:
            assert df[col].isna().sum(
            ) == 0, f"{col} still has NaN after imputation"

    def test_shape_preserved(self, raw_df):

        df = handle_missing_values(raw_df.copy())
        assert df.shape == raw_df.shape

class TestOutlierDetection:

    def test_returns_dataframe(self, raw_df):

        report = detect_outliers_iqr(raw_df)
        assert isinstance(report, pd.DataFrame)

    def test_has_expected_columns(self, raw_df):

        report = detect_outliers_iqr(raw_df)
        assert "n_outliers" in report.columns
        assert "pct_outliers" in report.columns

class TestWinsorization:

    def test_values_within_bounds(self, raw_df):

        df = treat_outliers(raw_df.copy())
        for feat, (lo, hi) in CLINICAL_RANGES.items():
            if feat in df.columns:
                col = df[feat].dropna()
                assert col.min() >= lo - 0.01,                    f"{feat} min ({col.min()}) below clinical floor ({lo})"
                assert col.max() <= hi + 0.01,                    f"{feat} max ({col.max()}) above clinical ceiling ({hi})"

    def test_winsorizer_class(self):

        df = pd.DataFrame({"sbp": [50, 100, 150, 280]})
        w = Winsorizer(lower_quantile=0.1, upper_quantile=0.9)
        result = w.fit_transform(df)
        assert result["sbp"].min() >= df["sbp"].quantile(0.1)
        assert result["sbp"].max() <= df["sbp"].quantile(0.9)

    def test_shape_preserved(self, raw_df):

        df = treat_outliers(raw_df.copy())
        assert len(df) == len(raw_df)

