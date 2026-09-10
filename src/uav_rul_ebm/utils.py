from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REQUIRED_ID_COLUMNS = {"uav_id", "flight_cycle"}


def telemetry_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(column for column in frame.columns if "telemetry" in column)


def load_csv(path: str | Path, *, require_target: bool = False) -> pd.DataFrame:
    """Load and validate one UAV trajectory CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file does not exist: {path}")

    frame = pd.read_csv(path)
    missing = REQUIRED_ID_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if require_target and "RUL" not in frame.columns:
        raise ValueError("Training data is missing required target column: RUL")
    telemetry = [column for column in frame.columns if column.startswith("telemetry_")]
    if not telemetry:
        raise ValueError(f"{path} contains no telemetry columns")
    return frame


def downsample(df: pd.DataFrame, is_prototype=False) -> pd.DataFrame:
    if not is_prototype:
        return df
    # Fast stride sampling (takes every 100th row while preserving trajectory order)
    return df.iloc[::100]


def disp(is_prototype=False) -> None:
    if is_prototype:
        plt.show()


def load_data(
    train_path: Path, test_path: Path, is_prototype=False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = downsample(
        load_csv(train_path, require_target=True).assign(split="train"), is_prototype
    )
    test = downsample(load_csv(test_path).assign(split="test"), is_prototype)
    categories = ["uav_id", "split"]
    train[categories] = train[categories].astype("category")
    test[categories] = test[categories].astype("category")
    return pd.concat([train, test], ignore_index=True), train, test
