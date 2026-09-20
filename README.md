# UAV RUL - Team EBM

A company operates a fleet of Unmanned Aerial Vehicles (UAVs). Our task is to build a model that predicts the Remaining Useful Life (RUL) of each UAV.

## Training workflow

Run the default workflow from the repository root with:

```text
python -m uav_rul_ebm.main
```

The workflow splits labeled data by `uav_id`, fits preprocessing only on the training fold, scales telemetry channels, evaluates one final-window prediction per validation UAV, then refits on all labeled data. It writes:

- `data/predictions/test_predictions.csv`: one `id,RUL` prediction per test UAV, in test-file order.
- `data/caches/rul_artifact.pkl`: fitted preprocessing pipeline, model weights, feature order, and model configuration.

By default, training trajectories are deterministically truncated to the sorted distribution of test trajectory lengths. This mimics the partial test observations, so training windows include endpoints with nonzero RUL instead of only terminal zero-RUL rows. Set `Config(truncate_to_test_horizons=False)` when this behavior is not wanted. The default 30-cycle window is configurable through `Config(window_size=...)`; trajectories shorter than the window fail with a clear error.

The final model can be selected with the compatibility fields on `Config`:

```python
from uav_rul_ebm.main import Config, run

run(Config(model_kind="tree"))    # diagnostic HistGradientBoosting baseline
run(Config(model_kind="neural"))  # CNN/LSTM window regressor
```

For model selection, enable an ensemble and declare independent candidates.
Candidates are fitted on the same UAV-held-out fold, scored with validation
`R2`, and the best individual model or weighted blend is refit on all training
data:

```python
from uav_rul_ebm.config import EnsembleConfig, ModelConfig
from uav_rul_ebm.master import Config, run

config = Config(
    ensemble=EnsembleConfig(
        enabled=True,
        candidates=(
            ModelConfig(name="tree", kind="tree", tree_max_iter=300),
            ModelConfig(name="neural_a", kind="neural", seed=7, epochs=30),
            ModelConfig(name="neural_b", kind="neural", seed=19, epochs=30),
        ),
        blend_weights={"tree": 1.0, "neural_a": 1.0, "neural_b": 1.0},
    )
)
run(config)
```

Use `PreprocessingConfig` through `Config(preprocessing=...)` to enable or
disable the existing sanitisers and feature-engineering steps and tune their
parameters. Ensemble artifacts include the selected fitted pipeline(s) and a
configuration fingerprint; changing model, preprocessing, or ensemble
settings requires retraining rather than reusing the artifact.

Both estimators implement the grouped sklearn contract: they accept complete
UAV trajectories and return one prediction per UAV, based on its final observed
window. The preprocessing steps preserve `uav_id`, `flight_cycle`, and `RUL`,
while `TelemetryStandardScaler` standardizes telemetry columns only.

Validation always reports target and prediction variation. When truncation is
disabled, held-out validation trajectories are evaluated at test-like horizons
so terminal `RUL=0` rows do not produce a misleading perfect score.

To reuse the saved pipeline and model without retraining:

```python
from uav_rul_ebm.main import Config, run

run(Config(reuse_artifact=True))
```
