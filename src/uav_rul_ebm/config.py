from dataclasses import dataclass
from pathlib import Path

from torch.cuda import is_available as is_cuda_available


@dataclass
class Config:
    """Configure the grouped UAV RUL training and prediction workflow.

    The workflow fits row-preserving preprocessing steps followed by one grouped
    final estimator. Estimators receive complete UAV trajectories and return one
    prediction per UAV, using the final observed window for inference.

    Attributes:
        train_path: CSV containing training trajectories and ``RUL`` targets.
        test_path: CSV containing partially observed trajectories for prediction.
        artifact_path: Location of the cached fitted sklearn pipeline.
        prediction_path: Location of the final ``id,RUL`` prediction CSV.
        model_kind: Final estimator, either ``"tree"`` or ``"neural"``.
        window_size: Number of consecutive cycles in each final model window.
        stride: Step between neural training windows within a UAV trajectory.
        validation_size: Fraction of UAVs assigned to the validation split.
        batch_size: Number of windows processed in one neural training batch.
        epochs: Maximum number of neural training epochs.
        learning_rate: Adam learning rate for the neural estimator.
        patience: Neural early-stopping patience in epochs.
        tree_max_iter: Number of boosting iterations for the tree estimator.
        tree_learning_rate: Learning rate for the tree estimator.
        seed: Random seed for splitting and estimator training.
        truncate_to_test_horizons: Whether to shorten training trajectories to
            the sorted length distribution observed in the test data.
        reuse_artifact: Whether to load the cached pipeline when present.
        device: PyTorch device used by the neural estimator.
        verbose:
    """

    train_path: Path = Path("data/raw/train.csv")
    test_path: Path = Path("data/raw/test.csv")
    artifact_path: Path = Path("data/caches/rul_artifact.pkl")
    prediction_path: Path = Path("data/predictions/test_predictions.csv")
    model_kind: str = "tree"
    window_size: int = 30
    stride: int = 1
    validation_size: float = 0.2
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    patience: int = 5
    tree_max_iter: int = 200
    tree_learning_rate: float = 0.05
    seed: int = 42
    truncate_to_test_horizons: bool = True
    reuse_artifact: bool = False
    device: str = "cuda" if is_cuda_available() else "cpu"
    verbose: bool = False
