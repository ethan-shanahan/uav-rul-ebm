from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from torch import nn
from torch.utils.data import DataLoader

from uav_rul_ebm.datasets import UAVWindowDataset, telemetry_columns
from uav_rul_ebm.models import Baseline
from uav_rul_ebm.utils import seed_everything


def _add_target(X: pd.DataFrame, y: Any) -> pd.DataFrame:
    """Attach row-aligned targets for grouped window construction."""
    if "RUL" in X.columns:
        return X
    if y is None:
        raise ValueError("Grouped regressors require RUL in X or a row-aligned y")
    values = np.asarray(y)
    if len(values) != len(X):
        raise ValueError("y must have one target per input row")
    X_out = X.copy()
    X_out["RUL"] = values
    return X_out


class UAVWindowRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-compatible CNN/LSTM regressor for grouped UAV trajectories."""

    def __init__(
        self,
        window_size: int = 30,
        stride: int = 1,
        batch_size: int = 64,
        epochs: int = 20,
        learning_rate: float = 1e-3,
        patience: int = 5,
        cnn_out_channels: int = 32,
        lstm_hidden_dim: int = 64,
        dropout_p: float = 0.3,
        seed: int = 42,
        device: str = "cpu",
        verbose: bool = True,
    ) -> None:
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.cnn_out_channels = cnn_out_channels
        self.lstm_hidden_dim = lstm_hidden_dim
        self.dropout_p = dropout_p
        self.seed = seed
        self.device = device
        self.verbose = verbose

    def _loader(
        self, X: pd.DataFrame, *, shuffle: bool, final_only: bool
    ) -> DataLoader:
        dataset = UAVWindowDataset(
            X,
            window_size=self.window_size,
            stride=self.stride,
            feature_cols=self.feature_cols_,
            require_target=True,
            final_window_only=final_only,
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, X: pd.DataFrame, y: Any = None) -> UAVWindowRegressor:
        """Fit the sequence model on all valid windows in ``X``."""
        seed_everything(self.seed)
        X = _add_target(X, y)
        self.feature_cols_ = telemetry_columns(X)
        loader = self._loader(X, shuffle=True, final_only=False)
        self.model_ = Baseline(
            num_features=len(self.feature_cols_),
            cnn_out_channels=self.cnn_out_channels,
            lstm_hidden_dim=self.lstm_hidden_dim,
            dropout_p=self.dropout_p,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        best_state = copy.deepcopy(self.model_.state_dict())
        best_loss = float("inf")
        stale = 0
        for epoch in range(self.epochs):
            self.model_.train()
            total_loss = 0.0
            total_items = 0
            for features, targets in loader:
                features = features.to(self.device)
                targets = targets.to(self.device).reshape(-1, 1)
                optimizer.zero_grad()
                loss = loss_fn(self.model_(features), targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(features)
                total_items += len(features)
            epoch_loss = total_loss / total_items
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
            if self.verbose:
                print(f"epoch={epoch + 1} train_loss={epoch_loss:.4f}")
        self.model_.load_state_dict(best_state)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict one RUL value for the final window of each UAV."""
        if not hasattr(self, "model_") or not hasattr(self, "feature_cols_"):
            raise RuntimeError("UAVWindowRegressor must be fitted before predict")
        dataset = UAVWindowDataset(
            X,
            window_size=self.window_size,
            feature_cols=self.feature_cols_,
            require_target=False,
            final_window_only=True,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model_.eval()
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for features, _ in loader:
                predictions.append(
                    self.model_(features.to(self.device)).cpu().numpy().ravel()
                )
        return np.concatenate(predictions)


class FinalWindowRegressor(BaseEstimator, RegressorMixin):
    """Diagnostic sklearn regressor using summary features from final windows."""

    def __init__(
        self,
        window_size: int = 30,
        random_state: int = 42,
        max_iter: int = 200,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 31,
        verbose: bool = True,
    ) -> None:
        self.window_size = window_size
        self.random_state = random_state
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.verbose = verbose

    def _features(
        self, X: pd.DataFrame, *, final_only: bool = False
    ) -> tuple[pd.DataFrame, list[float]]:
        columns = self.feature_cols_
        rows: list[dict[str, float]] = []
        targets: list[float] = []
        for _, group in X.groupby("uav_id", sort=False):
            ordered = group.sort_values("flight_cycle")
            if len(ordered) < self.window_size:
                raise ValueError(
                    f"UAV has {len(ordered)} rows; at least "
                    f"{self.window_size} are required"
                )
            starts = (
                [len(ordered) - self.window_size]
                if final_only
                else range(len(ordered) - self.window_size + 1)
            )
            for start in starts:
                endpoint = start + self.window_size - 1
                window = ordered.iloc[start : endpoint + 1]
                summary = {"flight_cycle": float(window["flight_cycle"].iloc[-1])}
                for column in columns:
                    values = window[column].to_numpy(dtype=float)
                    summary[f"{column}_last"] = values[-1]
                    summary[f"{column}_mean"] = float(values.mean())
                    summary[f"{column}_std"] = float(values.std())
                    summary[f"{column}_delta"] = float(values[-1] - values[0])
                rows.append(summary)
                if "RUL" in ordered:
                    targets.append(float(ordered["RUL"].iloc[endpoint]))
        return pd.DataFrame(rows), targets

    def fit(self, X: pd.DataFrame, y: Any = None) -> FinalWindowRegressor:
        """Fit a gradient-boosting regressor on one summary row per UAV."""
        X = _add_target(X, y)
        self.feature_cols_ = telemetry_columns(X)
        summary, targets = self._features(X)
        self.model_ = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=self.random_state,
            verbose=int(self.verbose),
        ).fit(summary, targets)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict one RUL value for each UAV's final observed window."""
        if not hasattr(self, "model_"):
            raise RuntimeError("FinalWindowRegressor must be fitted before predict")
        summary, _ = self._features(X, final_only=True)
        return self.model_.predict(summary)
