from dataclasses import dataclass, field
from pathlib import Path

from torch.cuda import is_available as is_cuda_available


@dataclass
class ModelConfig:
    """Configuration for one grouped model candidate."""

    name: str = "model"
    kind: str = "tree"
    window_size: int = 30
    stride: int = 1
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    patience: int = 5
    tree_max_iter: int = 200
    tree_learning_rate: float = 0.05
    seed: int = 42
    device: str = "cuda" if is_cuda_available() else "cpu"

    def __post_init__(self) -> None:
        if self.kind not in {"tree", "neural"}:
            raise ValueError("model kind must be 'tree' or 'neural'")
        if not self.name:
            raise ValueError("model candidate names must not be empty")
        for field_name in (
            "window_size",
            "stride",
            "batch_size",
            "epochs",
            "patience",
            "tree_max_iter",
        ):
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be positive")
        if self.learning_rate <= 0 or self.tree_learning_rate <= 0:
            raise ValueError("learning rates must be positive")


@dataclass
class PreprocessingConfig:
    """Enable and configure row-preserving preprocessing steps."""

    variance_threshold: bool = True
    variance_threshold_value: float = 1e-8
    point_outliers: bool = True
    point_outlier_window_size: int = 30
    point_outlier_n_sigma: float = 5.0
    trend_outliers: bool = True
    trend_outlier_window_size: int = 30
    trend_outlier_n_sigma: float = 5.0
    uav_pruning: bool = True
    uav_feat_ids: list[tuple[str, str]] = field(default_factory=list)
    imputer: bool = True
    imputer_max_iter: int = 10
    imputer_min_samples_leaf: int = 5
    scaler: bool = True
    telemetry_pruning: bool = True
    telemetry_cols: list[str] = field(
        default_factory=lambda: [
            "telemetry_01",
            "telemetry_02",
            "telemetry_04",
            "telemetry_05",
            "telemetry_07",
            "telemetry_09",
            "telemetry_10",
            "telemetry_16",
            "telemetry_18",
        ]
    )
    telemetry_merging: bool = True
    telemetry_mergers: list[list[str]] = field(
        default_factory=lambda: [
            ["telemetry_15", "telemetry_23"],
            ["telemetry_24", "telemetry_26"],
            ["telemetry_25", "telemetry_28"],
            ["telemetry_06", "telemetry_11", "telemetry_12"],
            ["telemetry_13", "telemetry_22"],
            ["telemetry_19", "telemetry_21"],
        ]
    )
    telemetry_smoothing: bool = True
    ema_alpha: float = 0.01

    def __post_init__(self) -> None:
        if self.variance_threshold_value < 0:
            raise ValueError("variance_threshold_value must be non-negative")
        for field_name in (
            "point_outlier_window_size",
            "trend_outlier_window_size",
            "imputer_max_iter",
            "imputer_min_samples_leaf",
        ):
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be positive")
        if self.ema_alpha <= 0 or self.ema_alpha > 1:
            raise ValueError("ema_alpha must be in (0, 1]")


@dataclass
class EnsembleConfig:
    """Optional candidate evaluation and prediction-blending settings."""

    enabled: bool = False
    candidates: tuple[ModelConfig, ...] = ()
    blend_weights: dict[str, float] = field(default_factory=dict)
    blend_name: str = "blend"

    def __post_init__(self) -> None:
        names = [candidate.name for candidate in self.candidates]
        if len(names) != len(set(names)):
            raise ValueError("ensemble candidate names must be unique")
        if self.blend_weights:
            if not all(name in names for name in self.blend_weights):
                raise ValueError("blend weights must refer to configured candidates")
            if any(weight < 0 for weight in self.blend_weights.values()):
                raise ValueError("blend weights must be non-negative")
            if sum(self.blend_weights.values()) <= 0:
                raise ValueError("blend weights must have a positive sum")
        if self.blend_name in names:
            raise ValueError("blend_name must differ from candidate names")


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
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    models: tuple[ModelConfig, ...] = ()

    def __post_init__(self) -> None:
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1")
        if not self.ensemble.candidates:
            candidate = ModelConfig(
                name=self.model_kind,
                kind=self.model_kind,
                window_size=self.window_size,
                stride=self.stride,
                batch_size=self.batch_size,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                patience=self.patience,
                tree_max_iter=self.tree_max_iter,
                tree_learning_rate=self.tree_learning_rate,
                seed=self.seed,
                device=self.device,
            )
            self.ensemble = EnsembleConfig(
                enabled=self.ensemble.enabled,
                candidates=(candidate,),
                blend_weights=self.ensemble.blend_weights,
                blend_name=self.ensemble.blend_name,
            )
