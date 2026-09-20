from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


def telemetry_columns(frame: pd.DataFrame) -> list[str]:
    """Return telemetry columns in a stable order."""
    columns = [column for column in frame.columns if "telemetry" in column]
    if not columns:
        raise ValueError("The frame contains no telemetry columns")
    return columns


def truncate_to_test_horizons(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Truncate training trajectories to the distribution of test horizons.

    Training and test UAV identifiers are different, so trajectories are paired
    by sorted length. This leaves endpoint labels above zero for partial
    trajectories and keeps the operation deterministic and opt-in.
    """
    train_groups = list(train.groupby("uav_id", sort=True))
    test_lengths = sorted(test.groupby("uav_id", sort=True).size().tolist())
    if len(train_groups) != len(test_lengths):
        raise ValueError("Training and test must contain the same number of UAVs")

    return truncate_to_horizons(train, test_lengths)


def truncate_to_horizons(frame: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Truncate trajectories to a supplied sorted list of horizons."""
    groups = list(frame.groupby("uav_id", sort=True))
    if len(groups) > len(horizons):
        raise ValueError("There must be at least one horizon per UAV")

    parts: list[pd.DataFrame] = []
    for (uav_id, group), horizon in zip(groups, horizons):
        ordered = group.sort_values("flight_cycle")
        parts.append(ordered.iloc[: min(horizon, len(ordered))])
    return pd.concat(parts, ignore_index=True)


class UAVWindowDataset(Dataset[tuple[Tensor, Any]]):
    """Fixed-length, within-trajectory windows for RUL regression."""

    def __init__(
        self,
        frame: pd.DataFrame,
        window_size: int = 30,
        stride: int = 1,
        feature_cols: list[str] | None = None,
        require_target: bool = True,
        final_window_only: bool = False,
    ) -> None:
        if window_size < 1 or stride < 1:
            raise ValueError("window_size and stride must be positive")
        if "uav_id" not in frame or "flight_cycle" not in frame:
            raise ValueError("Frame must contain uav_id and flight_cycle columns")
        if require_target and "RUL" not in frame:
            raise ValueError("Training windows require an RUL column")

        self.window_size = window_size
        self.feature_cols = feature_cols or telemetry_columns(frame)
        missing = [column for column in self.feature_cols if column not in frame]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        self.samples: list[tuple[np.ndarray, float | None, str, int]] = []

        for uav_id, group in frame.groupby("uav_id", sort=False):
            group = group.sort_values("flight_cycle")
            if len(group) < window_size:
                raise ValueError(
                    f"UAV {uav_id} has {len(group)} rows; "
                    f"at least {window_size} are required"
                )
            values = group[self.feature_cols].to_numpy(dtype=np.float32)
            targets = (
                group["RUL"].to_numpy(dtype=np.float32) if require_target else None
            )
            starts = (
                [len(group) - window_size]
                if final_window_only
                else range(0, len(group) - window_size + 1, stride)
            )
            cycles = group["flight_cycle"].to_numpy()
            for start in starts:
                endpoint = start + window_size - 1
                target = float(targets[endpoint]) if targets is not None else None
                self.samples.append(
                    (
                        values[start : endpoint + 1],
                        target,
                        str(uav_id),
                        int(cycles[endpoint]),
                    )
                )

        if not self.samples:
            raise ValueError("The frame produced no windows")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Any]:
        values, target, uav_id, endpoint_cycle = self.samples[index]
        features = torch.from_numpy(values.copy())
        if target is not None:
            return features, torch.tensor(target, dtype=torch.float32)
        return features, {"uav_id": uav_id, "flight_cycle": endpoint_cycle}
