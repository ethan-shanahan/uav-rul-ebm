from collections.abc import Hashable
from typing import cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


class FeatureVarianceThreshold(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        threshold: float = 1e-8,
        metadata_cols: tuple[str, ...] = ("uav_id", "flight_cycle", "RUL"),
        verbose: bool = False,
    ) -> None:
        self.threshold = threshold
        self.metadata_cols: tuple[str, ...] = metadata_cols
        self.verbose = verbose
        self.feature_cols_: list[str] | None = None
        self.vt = VarianceThreshold(threshold=threshold)
        self.vt.set_output(transform="pandas")

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_cols_ = [col for col in X.columns if col not in self.metadata_cols]
        self.vt.fit(X[self.feature_cols_], y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self.vt)

        X_features = X.filter(items=self.feature_cols_).copy()
        remaining_features = cast(pd.DataFrame, self.vt.transform(X_features))
        X_metadata = X.filter(items=self.metadata_cols).copy()
        return pd.concat([X_metadata, remaining_features], axis=1)


class PointOutlierFilter(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        rolling_window_size=30,
        n_sigma=5.0,
        metadata_cols: tuple[str, ...] = ("uav_id", "flight_cycle", "RUL"),
        verbose=False,
    ) -> None:
        self.rolling_window_size = rolling_window_size
        self.n_sigma = n_sigma
        self.metadata_cols: tuple[str, ...] = metadata_cols
        self.verbose = verbose
        self.id_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        self.id_ = next(col for col in X.columns if "id" in col)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "id_")

        X = X.copy()
        for c in [col for col in X.columns if col not in self.metadata_cols]:
            med = X.groupby(self.id_)[c].transform(
                lambda s: s.rolling(
                    self.rolling_window_size, center=True, min_periods=1
                ).median()
            )
            mad = X.groupby(self.id_)[c].transform(
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
            scale = 1.4826 * mad
            diff = (X[c] - med).abs()
            outlier_mask = (scale > 1e-6) & (diff > self.n_sigma * scale)
            X.loc[outlier_mask, c] = np.nan

        return X


class TrendOutlierFilter(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        init_window_size=30,
        n_sigma: float = 5,
        metadata_cols: tuple[str, ...] = ("uav_id", "flight_cycle", "RUL"),
        verbose=False,
    ) -> None:
        self.init_window_size: int = init_window_size
        self.n_sigma: float = n_sigma
        self.metadata_cols: tuple[str, ...] = metadata_cols
        self.verbose: bool = verbose
        self.id_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        self.id_ = next(col for col in X.columns if "id" in col)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "id_")

        X = X.copy()
        for c in [col for col in X.columns if col not in self.metadata_cols]:
            init_windows = X.groupby(self.id_)[c].apply(
                lambda x: x.iloc[: self.init_window_size].median()
            )
            med = init_windows.median()
            mad = (init_windows - med).abs().median()

            scale = 1.4826 * mad
            diff = (init_windows - med).abs()
            # print(f"feature: {c}\n\tdiff: {diff}\n\tscale*n: {self.n_sigma * scale}")
            outlier_mask = (scale > 1e-6) & (diff > self.n_sigma * scale)
            outlier_uav_mask = X[self.id_].isin(init_windows[outlier_mask].index)
            X.loc[outlier_uav_mask, c] = np.nan

        return X


class UAVPruning(BaseEstimator, TransformerMixin):
    def __init__(self, uav_feat_ids: list[tuple[str, str]], verbose=False) -> None:
        self.uav_feat_ids = uav_feat_ids
        self.verbose: bool = verbose
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "is_fitted_")

        X = X.copy()
        for uav_feat in self.uav_feat_ids:
            X.loc[uav_feat[0], uav_feat[1]] = np.nan

        return X


class HGBIImputer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        max_iter: int = 10,
        min_samples_leaf: int = 5,
        random_state: int = 42,
        metadata_cols: tuple[str, ...] = ("uav_id", "flight_cycle", "RUL"),
        verbose: bool = False,
    ) -> None:
        self.max_iter: int = max_iter
        self.min_samples_leaf = min_samples_leaf
        self.random_state: int = random_state
        self.metadata_cols: tuple[str, ...] = metadata_cols
        self.imputer_: IterativeImputer | None = None
        self.feature_cols_: list[str] | None = None
        self.verbose: bool = verbose

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None):
        """Fit the HistGB iterative imputer on the available feature columns."""
        self.feature_cols_ = [c for c in X.columns if c not in self.metadata_cols]
        self.imputer_ = IterativeImputer(
            estimator=HistGradientBoostingRegressor(
                random_state=self.random_state,
                min_samples_leaf=self.min_samples_leaf,
            ),
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.imputer_.fit(X[self.feature_cols_])
        return self

    def _interpolate_by_trajectory(self, X: pd.DataFrame) -> pd.DataFrame:
        """Interpolate missing feature values independently per UAV trajectory."""
        if self.feature_cols_ is None:
            raise RuntimeError("HGBIImputer must be fitted before interpolation")

        feature_cols = self.feature_cols_
        X_out = X.copy()
        if self.metadata_cols and "uav_id" in X_out.columns:
            order = X_out.sort_values(["uav_id", "flight_cycle"]).index
            ordered = X_out.loc[order]
            ordered[feature_cols] = ordered.groupby("uav_id", sort=False)[
                feature_cols
            ].transform(lambda series: series.interpolate(limit_direction="both"))
            X_out.loc[order, feature_cols] = ordered[feature_cols]
            return X_out

        X_out[feature_cols] = X_out[feature_cols].interpolate(limit_direction="both")
        return X_out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Interpolate and impute missing values using the fitted estimator."""
        if self.imputer_ is None or self.feature_cols_ is None:
            raise RuntimeError("HGBIImputer must be fitted before transform")

        missing_features = [c for c in self.feature_cols_ if c not in X.columns]
        if missing_features:
            raise ValueError(f"Missing fitted feature columns: {missing_features}")

        X_out = self._interpolate_by_trajectory(X)
        imputed_array = self.imputer_.transform(X_out[self.feature_cols_])
        X_out[self.feature_cols_] = imputed_array
        return X_out


class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        scaler: BaseEstimator | None = None,
        metadata_cols: tuple[str, ...] = ("uav_id", "flight_cycle", "RUL"),
    ) -> None:
        self.scaler = scaler
        self.metadata_cols = metadata_cols

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        self.feat_cols_ = [col for col in X.columns if col not in self.metadata_cols]

        # Use passed scaler or default to StandardScaler
        base_scaler = self.scaler if self.scaler is not None else StandardScaler()
        self.scaler_ = clone(base_scaler)

        if hasattr(self.scaler_, "set_output"):
            self.scaler_.set_output(transform="pandas")

        if self.feat_cols_:
            self.scaler_.fit(X[self.feat_cols_], y)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["scaler_", "feat_cols_"])

        X = X.copy()
        X_features = X[self.feat_cols_]
        scaled_features = cast(pd.DataFrame, self.scaler_.transform(X_features))
        X_metadata = X.filter(items=self.metadata_cols).copy()

        return pd.concat([X_metadata, scaled_features], axis=1)
