import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from uav_rul_ebm.config import Config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_by_uav(
    frame: pd.DataFrame, validation_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=validation_size, random_state=seed
    )
    train_indices, validation_indices = next(
        splitter.split(frame, groups=frame["uav_id"])
    )
    return frame.iloc[train_indices].copy(), frame.iloc[validation_indices].copy()


def final_targets(frame: pd.DataFrame) -> np.ndarray:
    """Return one endpoint target per UAV in first-seen order."""
    return np.asarray(
        [
            group.sort_values("flight_cycle")["RUL"].iloc[-1]
            for _, group in frame.groupby("uav_id", sort=False)
        ]
    )


def save_pipeline(path: Path, pipeline: Pipeline, config: Config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(
            {
                "pipeline": pipeline,
                "model_kind": config.model_kind,
                "window_size": config.window_size,
            },
            file,
        )


def load_pipeline(path: Path, config: Config) -> Pipeline:
    with path.open("rb") as file:
        bundle = pickle.load(file)
    if bundle["model_kind"] != config.model_kind:
        raise ValueError("Saved artifact uses a different model_kind")
    if bundle["window_size"] != config.window_size:
        raise ValueError("Saved artifact uses a different window_size")
    return bundle["pipeline"]
