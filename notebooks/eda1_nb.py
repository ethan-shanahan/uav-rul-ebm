import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    from pathlib import Path

    import matplotlib.pyplot as plt
    import seaborn as sb
    import pandas as pd

    PROJECT_ROOT = Path.cwd()
    TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
    TEST_PATH = PROJECT_ROOT / "data" / "raw" / "test.csv"
    PLOT_DIR = PROJECT_ROOT / "eda" / "plots"

    PROTOTYPING = False

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



@app.cell
def _():
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.compose import ColumnTransformer

    return (VarianceThreshold,)


@app.cell
def _():
    # Raw EDA
    raw, train_raw, test_raw = load_data(TRAIN_PATH, TEST_PATH, PROTOTYPING)
    base_telemetries = telemetry_columns(raw)
    return base_telemetries, raw


@app.cell
def _(raw):
    raw
    return


@app.cell(hide_code=True)
def _():
    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ('flatline_filter', VarianceThreshold(1e-6), base_telemetries)
    #     ],
    #     remainder='passthrough'
    # )
    return


@app.cell
def _(VarianceThreshold, base_telemetries, raw):
    # flatline filtration
    data = raw \
        .drop(columns=base_telemetries) \
        .join(VarianceThreshold(1e-6).set_output(transform="pandas").fit_transform(raw[base_telemetries]))
    return (data,)


@app.cell
def _(data):
    sb.displot(
        data=data, 
        x="telemetry_01",
    
        kind="kde",
        common_norm=True,
    
        # # hist options:
        # binwidth=2,
        # stat="density", 
        # element="step",
    
        # kde options:
        common_grid=True,
        fill=True,
    
        row="split",
        hue="uav_id", 
        legend=False,
        height=6, aspect=2
    )
    return


@app.cell
def _(data):
    # g = sb.displot(
    #     data=data,
    #     binwidth=2,
    #     stat="density", common_norm=False,
    #     x="telemetry_01",
    #     hue="split", element="step",
    #     height=6, aspect=2
    # )

    # ax = g.ax

    f = plt.figure(figsize=(12, 6))
    gs = f.add_gridspec(2, 1, hspace=0, height_ratios=[1, 0.1])

    ax = f.add_subplot(gs[0])
    sb.histplot(
        ax=ax,
        data=data,
        binwidth=2,
        stat="density", common_norm=False,
        x="telemetry_01",
        hue="split", element="step",
    )
    xlim = ax.get_xlim()
    ax = f.add_subplot(gs[1])
    sb.pointplot(
        ax=ax,
        data=data,
        x="telemetry_01",
        errorbar=("sd", 1),
        hue="split", legend=False
    )
    ax.set_xlim(left=xlim[0], right=xlim[1])

    # with sns.axes_style("darkgrid"):
    #     ax = f.add_subplot(gs[0, 0])
    #     sinplot(6)

    # with sns.axes_style("white"):
    #     ax = f.add_subplot(gs[0, 1])
    #     sinplot(6)

    # with sns.axes_style("ticks"):
    #     ax = f.add_subplot(gs[1, 0])
    #     sinplot(6)

    # with sns.axes_style("whitegrid"):
    #     ax = f.add_subplot(gs[1, 1])
    #     sinplot(6)

    f.tight_layout()

    # 2. Calculate mean and std per split
    # stats = data.groupby('split')['telemetry_01'].agg(['mean', 'std'])

    # 3. Position error bars near the top Y-axis limit
    # ylim = ax.get_ylim()
    # y_train = ylim[1] * 0.92
    # y_test = ylim[1] * 0.85

    # colors = {'train': sb.color_palette()[0], 'test': sb.color_palette()[1]}

    # # 4. Plot mean ± std error bars
    # ax.errorbar(
    #     x=stats.loc['train', 'mean'], y=0,
    #     xerr=stats.loc['train', 'std'],
    #     fmt='o', color=colors['train'], capsize=4, label='train mean ± 1 std'
    # )
    # ax.errorbar(
    #     x=stats.loc['test', 'mean'], y=0,
    #     xerr=stats.loc['test', 'std'],
    #     fmt='s', color=colors['test'], capsize=4, label='test mean ± 1 std'
    # )

    # g
    f
    return


@app.cell
def _(data):
    sb.relplot(
        data=data,
        kind="line",
        x="flight_cycle",
        y="telemetry_01",
        row="split",
        hue="uav_id",
        legend=False,
        alpha=0.33,
        height=6,
        aspect=2,
    )

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
