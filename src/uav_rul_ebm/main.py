from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

from uav_rul_ebm import preprocessors as p
from uav_rul_ebm.config import Config
from uav_rul_ebm.datasets import truncate_to_horizons, truncate_to_test_horizons
from uav_rul_ebm.estimators import FinalWindowRegressor, UAVWindowRegressor
from uav_rul_ebm.utils import (
    final_targets,
    load_pipeline,
    save_pipeline,
    seed_everything,
    split_by_uav,
)


def _make_model(config: Config):
    match config.model_kind:
        case "tree":
            return FinalWindowRegressor(
                window_size=config.window_size,
                random_state=config.seed,
                max_iter=config.tree_max_iter,
                learning_rate=config.tree_learning_rate,
                verbose=False,
            )
        case "neural":
            return UAVWindowRegressor(
                window_size=config.window_size,
                stride=config.stride,
                batch_size=config.batch_size,
                epochs=config.epochs,
                learning_rate=config.learning_rate,
                patience=config.patience,
                seed=config.seed,
                device=config.device,
            )
        case _:
            raise ValueError("model_kind must be either 'tree' or 'neural'")


def make_pipeline(config: Config) -> Pipeline:
    """Build the preprocessing and grouped-estimator pipeline."""
    return Pipeline(
        [
            (
                "variance threshold",
                p.VarianceThreshold(),
            ),
            (
                "outlier nullifier",
                p.VerticalHampelFilter(rolling_window_size=30, n_sigmas=5),
            ),
            (
                "defect nullifier",
                p.HorizontalHampelFilter(init_window_size=40, n_sigmas=5),
            ),
            (
                "imputer",
                p.HGBIImputer(),
            ),
            (
                "scaler",
                p.TelemetryStandardScaler(),
            ),
            (
                "model",
                _make_model(config),
            ),
        ],
        verbose=config.verbose,
    )


def run(config: Config | None = None) -> pd.DataFrame:
    """Fit, validate, cache, and run the configured grouped RUL pipeline."""
    config = Config() if config is None else config
    seed_everything(config.seed)
    train_df = pd.read_csv(config.train_path)
    test_df = pd.read_csv(config.test_path)

    if config.reuse_artifact and config.artifact_path.exists():
        pipeline = load_pipeline(config.artifact_path, config)
    else:
        if config.truncate_to_test_horizons:
            train_df = truncate_to_test_horizons(train_df, test_df)
        train_fold, validation_fold = split_by_uav(
            train_df, config.validation_size, config.seed
        )
        if not config.truncate_to_test_horizons:
            test_horizons = sorted(test_df.groupby("uav_id", sort=True).size().tolist())
            validation_fold = truncate_to_horizons(
                validation_fold, test_horizons[: validation_fold["uav_id"].nunique()]
            )
        validation_pipeline = make_pipeline(config)
        validation_pipeline.fit(train_fold)
        validation_predictions = np.asarray(
            validation_pipeline.predict(validation_fold)
        )
        validation_targets = final_targets(validation_fold)
        print(
            f"validation_r2={r2_score(validation_targets, validation_predictions):.4f} "
            f"target_std={validation_targets.std():.4f} "
            f"prediction_std={validation_predictions.std():.4f}"
        )

        pipeline = make_pipeline(config)
        pipeline.fit(train_df)
        save_pipeline(config.artifact_path, pipeline, config)

    predictions = np.maximum(np.asarray(pipeline.predict(test_df)), 0.0)
    uav_ids = test_df.groupby("uav_id", sort=False).size().index.tolist()
    output = pd.DataFrame({"id": uav_ids, "RUL": predictions})
    config.prediction_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(config.prediction_path, index=False)
    return output


if __name__ == "__main__":
    run(Config(verbose=True, model_kind="neural"))
