from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

from uav_rul_ebm.config import Config, ModelConfig
from uav_rul_ebm.utils import final_targets


@dataclass
class EnsembleResult:
    """Validation result and fitted pipelines for an ensemble selection run."""

    selected_name: str
    scores: dict[str, float]
    pipelines: dict[str, Pipeline]
    candidates: tuple[ModelConfig, ...]
    blend_weights: dict[str, float]


def _uav_order(frame: pd.DataFrame) -> list[object]:
    return frame.groupby("uav_id", sort=False).size().index.tolist()


def _blend_weights(
    candidates: tuple[ModelConfig, ...], configured: dict[str, float]
) -> dict[str, float]:
    names = [candidate.name for candidate in candidates]
    if configured:
        weights = {name: configured.get(name, 0.0) for name in names}
    else:
        weights = {name: 1.0 for name in names}
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("ensemble blend weights must have a positive sum")
    return {name: weight / total for name, weight in weights.items() if weight > 0}


def _blend(predictions: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    selected = [predictions[name] for name in weights]
    if not selected or any(len(values) != len(selected[0]) for values in selected):
        raise ValueError("ensemble predictions must have equal lengths")
    return sum(weights[name] * predictions[name] for name in weights)


def validate_candidates(
    config: Config, train_fold: pd.DataFrame, validation_fold: pd.DataFrame
) -> EnsembleResult:
    """Fit candidates on one fold and select the highest validation R2."""
    candidates = config.ensemble.candidates
    if not candidates:
        raise ValueError("ensemble requires at least one candidate")
    validation_order = _uav_order(validation_fold)
    targets = final_targets(validation_fold)
    predictions: dict[str, np.ndarray] = {}
    pipelines: dict[str, Pipeline] = {}
    scores: dict[str, float] = {}

    for candidate in candidates:
        pipeline = _make_pipeline(config, candidate)
        pipeline.fit(train_fold)
        values = np.asarray(pipeline.predict(validation_fold), dtype=float)
        if len(values) != len(validation_order):
            raise ValueError(
                f"candidate {candidate.name!r} returned {len(values)} predictions "
                f"for {len(validation_order)} validation UAVs"
            )
        pipelines[candidate.name] = pipeline
        predictions[candidate.name] = values
        scores[candidate.name] = float(r2_score(targets, values))

    if config.ensemble.enabled and len(candidates) > 1:
        weights = _blend_weights(candidates, config.ensemble.blend_weights)
        predictions[config.ensemble.blend_name] = _blend(predictions, weights)
        scores[config.ensemble.blend_name] = float(
            r2_score(targets, predictions[config.ensemble.blend_name])
        )
    else:
        weights = {}

    selected_name = max(scores, key=scores.get)
    return EnsembleResult(
        selected_name=selected_name,
        scores=scores,
        pipelines=pipelines,
        candidates=candidates,
        blend_weights=weights,
    )


def _make_pipeline(config: Config, candidate: ModelConfig) -> Pipeline:
    # Import locally to avoid a master -> ensemble -> master import cycle.
    from uav_rul_ebm.master import make_pipeline

    return make_pipeline(config, candidate)


def refit_selected(
    config: Config, result: EnsembleResult, training_frame: pd.DataFrame
) -> dict[str, Pipeline]:
    """Fit only the selected candidate or selected blend components."""
    names = (
        list(result.blend_weights)
        if result.selected_name == config.ensemble.blend_name
        else [result.selected_name]
    )
    candidates = {candidate.name: candidate for candidate in result.candidates}
    fitted: dict[str, Pipeline] = {}
    for name in names:
        pipeline = _make_pipeline(config, candidates[name])
        pipeline.fit(training_frame)
        fitted[name] = pipeline
    return fitted


def predict_selected(
    config: Config,
    result: EnsembleResult,
    pipelines: dict[str, Pipeline],
    frame: pd.DataFrame,
) -> np.ndarray:
    """Predict with the selected fitted model or blend."""
    predictions = {
        name: np.asarray(pipeline.predict(frame), dtype=float)
        for name, pipeline in pipelines.items()
    }
    if result.selected_name == config.ensemble.blend_name:
        return _blend(predictions, result.blend_weights)
    return predictions[result.selected_name]
