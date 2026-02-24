import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline

from src.config import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, CLINICAL_RANGES,
    RANDOM_SEED, DATA_FILE, ENGINEERED_DATA_FILE, PROCESSED_DATA_DIR
)
from src.preprocessing_utils import Winsorizer
from src.feature_engineering import engineer_all_features

import os
from src.logger import get_logger
logger = get_logger(__name__)

def load_raw_data():
    return pd.read_csv(DATA_FILE)

def handle_missing_values(df, strategy="mice"):
    numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    imputer = IterativeImputer(
        max_iter=10,
        random_state=RANDOM_SEED,
        sample_posterior=False
    )
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df

def detect_outliers_iqr(df, features=None, factor=1.5):
    if features is None:
        features = [c for c in NUMERIC_FEATURES if c in df.columns]

    outlier_report = {}
    for col in features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_report[col] = {
            "lower_bound": round(lower, 2),
            "upper_bound": round(upper, 2),
            "n_outliers": n_outliers,
            "pct_outliers": round(n_outliers / len(df) * 100, 2)
        }
    return pd.DataFrame(outlier_report).T

def treat_outliers(df, method="winsorize"):
    df = df.copy()
    for feat, (lo, hi) in CLINICAL_RANGES.items():
        if feat in df.columns:
            df[feat] = df[feat].clip(lower=lo, upper=hi)
    return df

def encode_categoricals(df, features=None):
    if features is None:
        features = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    return pd.get_dummies(df, columns=features, drop_first=False)

def scale_features(df, features=None):
    if features is None:
        features = [c for c in NUMERIC_FEATURES if c in df.columns]
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, scaler

def build_preprocessor(numeric_features=None, categorical_features=None):
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    numeric_pipeline = SkPipeline([
        ("imputer", IterativeImputer(max_iter=10, random_state=RANDOM_SEED)),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = SkPipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="passthrough"
    )
    return preprocessor

def preprocess_full_pipeline(df):
    logger.info("Step 1: Handling missing values (MICE imputation)...")
    df = handle_missing_values(df)

    logger.info("Step 2: Treating outliers (clinical Winsorization)...")
    df = treat_outliers(df)

    logger.info("Step 3: Engineering features (8 new clinical features)...")
    df = engineer_all_features(df)

    logger.info(f"Step 4: Saving processed dataset to {ENGINEERED_DATA_FILE}...")
    df.to_csv(ENGINEERED_DATA_FILE, index=False)

    return df

if __name__ == "__main__":
    df = load_raw_data()
    logger.info(f"Raw data loaded: {df.shape}")
    df = preprocess_full_pipeline(df)
    logger.info(f"Processed data shape: {df.shape}")

