import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline


# 1. Custom VarianceThreshold Wrapper (preserves DataFrame and metadata)
class VarianceThreshold(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        threshold=1e-8,
        metadata_cols=("uav_id", "flight_cycle", "RUL"),
        verbose=False,
    ):
        self.threshold = threshold
        self.metadata_cols = metadata_cols
        self.feature_cols_ = None
        self.cols_to_keep_ = None
        self.v_ = verbose

    def fit(self, X, y=None):
        feat_cols = [c for c in X.columns if c not in self.metadata_cols]
        variances = X[feat_cols].var()
        self.feature_cols_ = variances[variances > self.threshold].index.tolist()
        self.cols_to_keep_ = [
            c for c in X.columns if c in self.metadata_cols or c in self.feature_cols_
        ]
        return self

    def transform(self, X):
        if self.feature_cols_ is None or self.cols_to_keep_ is None:
            raise RuntimeError("VarianceThreshold must be fitted before transform")

        missing_features = [c for c in self.feature_cols_ if c not in X.columns]
        if missing_features:
            raise ValueError(f"Missing fitted feature columns: {missing_features}")

        cols_to_transform = [c for c in self.cols_to_keep_ if c in X.columns]

        if self.v_:
            print(
                f"VarianceThreshold:\n\tDropped Columns: {[c for c in X.columns if c not in X[cols_to_transform].columns]}"
            )

        return X[cols_to_transform].copy()


# Nullifies telemetries' outliers
class VerticalHampelFilter(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        rolling_window_size=30,
        n_sigmas=5.0,
        id_col="uav_id",
        metadata_cols=("uav_id", "flight_cycle", "RUL"),
        verbose=False,
    ):
        self.rolling_window_size = rolling_window_size
        self.n_sigmas = n_sigmas
        self.id_col = id_col
        self.metadata_cols = metadata_cols
        self.v_ = verbose

    def fit(self, X, y=None):
        self.feature_cols_ = [c for c in X.columns if c not in self.metadata_cols]
        return self

    def transform(self, X: pd.DataFrame):
        if not hasattr(self, "feature_cols_"):
            raise RuntimeError("VerticalHampelFilter must be fitted before transform")

        X_out = X.copy()
        feat_cols = self.feature_cols_

        for col in feat_cols:
            if self.id_col in X_out.columns:
                med = X_out.groupby(self.id_col)[col].transform(
                    lambda s: s.rolling(
                        self.rolling_window_size, center=True, min_periods=1
                    ).median()
                )
                mad = X_out.groupby(self.id_col)[col].transform(
                    lambda s: (
                        (
                            s
                            - s.rolling(
                                self.rolling_window_size, center=True, min_periods=1
                            ).median()
                        )
                        .abs()
                        .rolling(self.rolling_window_size, center=True, min_periods=1)
                        .median()
                    )
                )
            else:
                med = (
                    X_out[col]
                    .rolling(self.rolling_window_size, center=True, min_periods=1)
                    .median()
                )
                mad = (
                    (X_out[col] - med)
                    .abs()
                    .rolling(self.rolling_window_size, center=True, min_periods=1)
                    .median()
                )

            scale = 1.4826 * mad
            diff = (X_out[col] - med).abs()
            is_outlier = (scale > 1e-6) & (diff > self.n_sigmas * scale)
            if self.v_:
                print(f"Feature: {col}\n\tScale: {scale}\tOutliers: {is_outlier.sum()}")
                is_outlier.to_csv(
                    f"./data/outliers/a/train/{col}.csv"
                    if "RUL" in X_out.columns
                    else f"./data/outliers/a/test/{col}.csv",
                    index=False,
                )
            X_out.loc[is_outlier, col] = np.nan

        return X_out


# Nullifies faulty telemetry channels for UAVs
class HorizontalHampelFilter(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        init_window_size=10,
        n_sigmas=3.0,
        id_col="uav_id",
        metadata_cols=("uav_id", "flight_cycle", "RUL"),
        verbose=False,
    ):
        self.init_window_size = init_window_size
        self.n_sigmas = n_sigmas
        self.id_col = id_col
        self.metadata_cols = metadata_cols
        self.v_ = verbose

    def fit(self, X, y=None):
        self.feature_cols_ = [c for c in X.columns if c not in self.metadata_cols]
        return self

    def transform(self, X: pd.DataFrame):
        if not hasattr(self, "feature_cols_"):
            raise RuntimeError("HorizontalHampelFilter must be fitted before transform")

        X_out = X.copy()
        feat_cols = self.feature_cols_

        for col in feat_cols:
            init_windows = X_out.groupby(self.id_col)[col].apply(
                lambda x: x.iloc[: self.init_window_size].median()
            )
            med = init_windows.median()
            mad = (init_windows - med).abs().median()
            scale = 1.4826 * mad
            diff = (init_windows - med).abs()
            is_outlier = (scale > 1e-6) & (diff > self.n_sigmas * scale)

            if self.v_:
                print(
                    f"Feature: {col}\n\tFaulty UAVs: {init_windows[is_outlier].index.tolist()}"
                )

            faulty_mask = X_out[self.id_col].isin(init_windows[is_outlier].index)
            X_out.loc[faulty_mask, col] = np.nan

        return X_out


# 3. Iterative HistGB Imputer Wrapper
class HGBIImputer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        max_iter=5,
        random_state=42,
        metadata_cols=("uav_id", "flight_cycle", "RUL"),
        verbose=False,
    ):
        self.max_iter = max_iter
        self.random_state = random_state
        self.metadata_cols = metadata_cols
        self.imputer_ = None
        self.feature_cols_ = None
        self.v_ = verbose

    def fit(self, X, y=None):
        self.feature_cols_ = [c for c in X.columns if c not in self.metadata_cols]
        self.imputer_ = IterativeImputer(
            estimator=HistGradientBoostingRegressor(random_state=self.random_state),
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.imputer_.fit(X[self.feature_cols_])
        return self

    def _interpolate_by_trajectory(self, X):
        X_out = X.copy()
        if self.metadata_cols and "uav_id" in X_out.columns:
            order = X_out.sort_values(["uav_id", "flight_cycle"]).index
            ordered = X_out.loc[order]
            ordered[self.feature_cols_] = ordered.groupby("uav_id", sort=False)[
                self.feature_cols_
            ].transform(lambda series: series.interpolate(limit_direction="both"))
            X_out.loc[order, self.feature_cols_] = ordered[self.feature_cols_]
            return X_out

        X_out[self.feature_cols_] = X_out[self.feature_cols_].interpolate(
            limit_direction="both"
        )
        return X_out

    def transform(self, X):
        if self.imputer_ is None or self.feature_cols_ is None:
            raise RuntimeError("HGBIImputer must be fitted before transform")

        missing_features = [c for c in self.feature_cols_ if c not in X.columns]
        if missing_features:
            raise ValueError(f"Missing fitted feature columns: {missing_features}")

        X_out = self._interpolate_by_trajectory(X)
        imputed_array = self.imputer_.transform(X_out[self.feature_cols_])
        X_out[self.feature_cols_] = imputed_array
        return X_out


if __name__ == "__main__":
    TRAIN_PATH = "./data/raw/train.csv"
    TEST_PATH = "./data/raw/test.csv"
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    pl = Pipeline(
        [
            ("variance threshold", VarianceThreshold(verbose=True)),
            (
                "outlier nullifier",
                VerticalHampelFilter(rolling_window_size=30, n_sigmas=5, verbose=False),
            ),
            (
                "defect nullifier",
                HorizontalHampelFilter(init_window_size=40, n_sigmas=5, verbose=False),
            ),
            ("imputer", HGBIImputer()),
        ],
        verbose=True,
    )
    train = pl.fit_transform(train_df)
    test = pl.transform(test_df)
