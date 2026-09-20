from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from uav_rul_ebm import estimators, feature_engineering, sanitisers
from uav_rul_ebm.config import Config, ModelConfig
from uav_rul_ebm.datasets import truncate_to_horizons, truncate_to_test_horizons
from uav_rul_ebm.ensemble import (
    EnsembleResult,
    predict_selected,
    refit_selected,
    validate_candidates,
)
from uav_rul_ebm.utils import (
    load_artifact,
    save_artifact,
    seed_everything,
    split_by_uav,
)


def make_model(model_config: ModelConfig, *, verbose: bool) -> object:
    if model_config.kind == "tree":
        return estimators.FinalWindowRegressor(
            window_size=model_config.window_size,
            random_state=model_config.seed,
            max_iter=model_config.tree_max_iter,
            learning_rate=model_config.tree_learning_rate,
            verbose=verbose,
        )
    return estimators.UAVWindowRegressor(
        window_size=model_config.window_size,
        stride=model_config.stride,
        batch_size=model_config.batch_size,
        epochs=model_config.epochs,
        learning_rate=model_config.learning_rate,
        patience=model_config.patience,
        seed=model_config.seed,
        device=model_config.device,
        verbose=verbose,
    )


def make_pipeline(config: Config, model_config: ModelConfig | None = None) -> Pipeline:
    model_config = (
        config.ensemble.candidates[0] if model_config is None else model_config
    )
    preprocessing = config.preprocessing
    steps: list[tuple[str, object]] = []
    if preprocessing.variance_threshold:
        steps.append(
            (
                "variance_threshold",
                sanitisers.FeatureVarianceThreshold(
                    threshold=preprocessing.variance_threshold_value
                ),
            )
        )
    if preprocessing.point_outliers:
        steps.append(
            (
                "point_outliers",
                sanitisers.PointOutlierFilter(
                    rolling_window_size=preprocessing.point_outlier_window_size,
                    n_sigma=preprocessing.point_outlier_n_sigma,
                ),
            )
        )
    if preprocessing.trend_outliers:
        steps.append(
            (
                "trend_outliers",
                sanitisers.TrendOutlierFilter(
                    init_window_size=preprocessing.trend_outlier_window_size,
                    n_sigma=preprocessing.trend_outlier_n_sigma,
                ),
            )
        )
    if preprocessing.uav_pruning:
        steps.append(("uav_pruning", sanitisers.UAVPruning(preprocessing.uav_feat_ids)))
    if preprocessing.imputer:
        steps.append(
            (
                "imputer",
                sanitisers.HGBIImputer(
                    max_iter=preprocessing.imputer_max_iter,
                    min_samples_leaf=preprocessing.imputer_min_samples_leaf,
                ),
            )
        )
    if preprocessing.scaler:
        steps.append(("scaler", sanitisers.FeatureScaler()))
    if preprocessing.telemetry_pruning:
        steps.append(
            (
                "telemetry_pruning",
                feature_engineering.TelemetryPruning(preprocessing.telemetry_cols),
            )
        )
    if preprocessing.telemetry_merging:
        steps.append(
            (
                "telemetry_merging",
                feature_engineering.TelemetryMerging(preprocessing.telemetry_mergers),
            )
        )
    if preprocessing.telemetry_smoothing:
        steps.append(
            (
                "telemetry_smoothing",
                feature_engineering.EMAFeatures(preprocessing.ema_alpha),
            )
        )
    steps.append(("model", make_model(model_config, verbose=config.verbose)))
    return Pipeline(steps, verbose=config.verbose)


def run(config: Config | None = None) -> pd.DataFrame:
    """Fit, validate, cache, and run the configured grouped RUL pipeline."""
    config = Config() if config is None else config
    seed_everything(config.seed)
    train_df = pd.read_csv(config.train_path)
    test_df = pd.read_csv(config.test_path)

    if config.reuse_artifact and config.artifact_path.exists():
        bundle = load_artifact(config.artifact_path, config)
        result = EnsembleResult(
            selected_name=bundle["selected_name"],
            scores=bundle["scores"],
            pipelines=bundle["pipelines"],
            candidates=config.ensemble.candidates,
            blend_weights=bundle["blend_weights"],
        )
        fitted_pipelines = bundle["pipelines"]
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
        result = validate_candidates(config, train_fold, validation_fold)
        for name, score in result.scores.items():
            print(f"{name}_validation_r2={score:.4f}")
        print(f"selected_model={result.selected_name}")
        fitted_pipelines = refit_selected(config, result, train_df)
        save_artifact(
            config.artifact_path,
            {
                "selected_name": result.selected_name,
                "scores": result.scores,
                "blend_weights": result.blend_weights,
                "pipelines": fitted_pipelines,
            },
            config,
        )

    predictions = np.maximum(
        predict_selected(config, result, fitted_pipelines, test_df), 0.0
    )
    uav_ids = test_df.groupby("uav_id", sort=False).size().index.tolist()
    output = pd.DataFrame({"id": uav_ids, "RUL": predictions})
    config.prediction_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(config.prediction_path, index=False)
    return output


if __name__ == "__main__":
    config = Config(epochs=30, verbose=True)
    run(config)
