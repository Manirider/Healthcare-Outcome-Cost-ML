import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from src.logger import get_logger
logger = get_logger(__name__)

pd.set_option('future.no_silent_downcasting', True)

class Winsorizer(BaseEstimator, TransformerMixin):

    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_limits_ = {}
        self.upper_limits_ = {}

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]

        X = pd.DataFrame(X, columns=self.feature_names_in_)

        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.lower_limits_[col] = X[col].quantile(self.lower_quantile)
                self.upper_limits_[col] = X[col].quantile(self.upper_quantile)
            else:
                self.lower_limits_[col] = -np.inf
                self.upper_limits_[col] = np.inf

        return self

    def transform(self, X):
        check_is_fitted(self, ["lower_limits_", "upper_limits_"])

        X_out = pd.DataFrame(X, columns=self.feature_names_in_).copy()

        for col, lower in self.lower_limits_.items():
            upper = self.upper_limits_[col]
            X_out[col] = X_out[col].clip(lower=lower, upper=upper).infer_objects(copy=False)

        return X_out

