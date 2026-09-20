from itertools import chain

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TelemetryPruning(BaseEstimator, TransformerMixin):
    def __init__(self, telemetry_cols: list[str], verbose=False) -> None:
        self.telemetry_cols = telemetry_cols
        self.verbose: bool = verbose
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("TelemetryPruning must be fitted before transform.")

        return X.drop(columns=self.telemetry_cols)


class TelemetryMerging(BaseEstimator, TransformerMixin):
    def __init__(self, telemetry_mergers: list[list[str]], verbose=False) -> None:
        self.telemetry_mergers = telemetry_mergers
        self.verbose: bool = verbose
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("TelemetryPruning must be fitted before transform.")

        X = X.copy()
        for tels in self.telemetry_mergers:
            tel_ids = [t.split("_", maxsplit=1)[-1] for t in tels]
            X["telemetry_" + "_".join(tel_ids)] = X[tels].mean(axis=1)
        X = X.drop(columns=list(chain.from_iterable(self.telemetry_mergers)))

        return X


class EMAFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        alpha: float = 0.01,
        metadata_cols: tuple[str, ...] = ("uav_id", "flight_cycle", "RUL"),
        verbose=False,
    ) -> None:
        self.alpha = alpha
        self.metadata_cols: tuple[str, ...] = metadata_cols
        self.id_ = None
        self.verbose: bool = verbose

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        self.id_ = next(col for col in X.columns if "id" in col)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.id_:
            raise RuntimeError("TelemetryPruning must be fitted before transform.")

        X = X.copy()
        for c in [col for col in X.columns if col not in self.metadata_cols]:
            X[f"{c}_ema"] = X.groupby(self.id_)[c].transform(
                lambda x: x.ewm(alpha=self.alpha, adjust=False).mean()
            )

        return X
