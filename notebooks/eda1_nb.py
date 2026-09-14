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
    return base_telemetries, raw, train_raw


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
    # drop flatlines
    data = raw \
        .drop(columns=base_telemetries) \
        .join(VarianceThreshold(1e-6).set_output(transform="pandas").fit_transform(raw[base_telemetries]))
    core_telemetries = telemetry_columns(data)
    return core_telemetries, data


@app.cell
def _():
    # data.groupby("uav_id", sort=False)[core_telemetries].rolling(window=5, min_periods=1).median()
    # data
    return


@app.cell
def _():
    # data_medians = data.groupby("uav_id")[core_telemetries].mean()
    return


@app.function
def clean_hampel(df, feature_cols, group_col='uav_id', window=5, n_sigmas=3):
    grouped = df.groupby(group_col)[feature_cols]
    
    # Calculate rolling median & MAD in parallel across all feature columns
    med = grouped.transform(lambda x: x.rolling(window, center=True, min_periods=1).median())
    mad = grouped.transform(
        lambda x: (x - x.rolling(window, center=True, min_periods=1).median())
        .abs()
        .rolling(window, center=True, min_periods=1)
        .median()
    )
    
    # 1.4826 converts MAD to an equivalent standard deviation
    threshold = n_sigmas * 1.4826 * mad
    is_spike = (df[feature_cols] - med).abs() > threshold
    
    return df[feature_cols].where(~is_spike, med)


@app.cell
def _(core_telemetries, data):
    # Pipeline usage:
    data_B = data \
        .drop(columns=core_telemetries) \
        .join(clean_hampel(data, core_telemetries, window=9, n_sigmas=2.5))
    return (data_B,)


@app.cell
def _(data_B):
    data_B
    return


@app.cell
def _():
    # fig, axs_spikes = plt.subplots(1,2, figsize=(12,6))
    # sb.lineplot(ax=axs_spikes[0], data=data, x="flight_cycle", y="telemetry_10", hue="uav_id", alpha=0.33, legend=False)
    # sb.lineplot(ax=axs_spikes[1], data=data_B, x="flight_cycle", y="telemetry_10", hue="uav_id", alpha=0.33, legend=False)
    # fig.tight_layout()
    # fig
    return


@app.cell
def _():
    # for t in telemetry_columns(data_B):
    #     print(f"Plotting: {t}.png")
    #     fig, axs_spikes =plt.subplots(2, 1, figsize=(12,12))
    #     sb.lineplot(ax=axs_spikes[0], data=data, x="flight_cycle", y=t, hue="uav_id", alpha=0.33, legend=False)
    #     sb.lineplot(ax=axs_spikes[1], data=data_B, x="flight_cycle", y=t, hue="uav_id", alpha=0.33, legend=False)
    #     fig.tight_layout()
    #     fig.savefig(PLOT_DIR / "telemetry_spike_removal" / f"{t}.png")
    #     plt.close()
    return


@app.cell
def _():
    from sklearn.ensemble import HistGradientBoostingRegressor

    def impute_corrupted_telemetries(dataframe, target_cols, group_col='uav_id', z_thresh=3.5, init_window=10):
        train_clean = dataframe[dataframe["split"] == "train"]
        test_clean = dataframe[dataframe["split"] == "test"]
        all_telemetries = [c for c in dataframe.columns if c.startswith('telemetry_')]

        for target_col in target_cols:
            # 1. MAD Baseline Detection
            def get_bad_uavs(df):
                starts = df.groupby(group_col)[target_col].apply(lambda x: x.iloc[:init_window].median())
                med = starts.median()
                scale = 1.4826 * (starts - med).abs().median()
                return starts[(starts - med).abs() / scale > z_thresh].index.tolist() if scale > 0 else []

            bad_train_uavs = get_bad_uavs(train_clean)
            bad_test_uavs = get_bad_uavs(test_clean)

            if not bad_train_uavs and not bad_test_uavs:
                continue

            # 2. Train Cross-Sensor Model using scikit-learn
            predictors = [c for c in all_telemetries if c != target_col]
            healthy_mask = ~train_clean[group_col].isin(bad_train_uavs)
        
            model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
            model.fit(train_clean.loc[healthy_mask, predictors], train_clean.loc[healthy_mask, target_col])

            # 3. Impute Corrupted Telemetry Signals
            if bad_train_uavs:
                tr_mask = train_clean[group_col].isin(bad_train_uavs)
                train_clean.loc[tr_mask, target_col] = model.predict(train_clean.loc[tr_mask, predictors])
            if bad_test_uavs:
                te_mask = test_clean[group_col].isin(bad_test_uavs)
                test_clean.loc[te_mask, target_col] = model.predict(test_clean.loc[te_mask, predictors])

        return pd.concat([train_clean, test_clean], ignore_index=True)

    return HistGradientBoostingRegressor, impute_corrupted_telemetries


@app.cell
def _(core_telemetries, data_B, impute_corrupted_telemetries):
    # Usage
    data_C = impute_corrupted_telemetries(
        data_B, 
        target_cols=core_telemetries#['telemetry_22', 'telemetry_26']
    )
    return


@app.cell
def _():
    # for t in telemetry_columns(data):
    #     print(f"Plotting: {t}.png")
    #     plot = sb.displot(
    #         data=data,
    #         x=t,
    #         kind="kde",
    #         common_norm=True,
    #         common_grid=True,
    #         fill=True,
    #         row="split",
    #         hue="uav_id",
    #         legend=False,
    #         height=6,
    #         aspect=2,
    #     )
    #     plot.savefig(PLOT_DIR / "telemetry_kdeplots" / f"{t}.png")
    #     plt.close()
    return


@app.cell
def _():
    # for t in telemetry_columns(data_B):
    #     print(f"Plotting: {t}.png")
    #     fig, axs_imputed =plt.subplots(2, 1, figsize=(12,12))
    #     sb.lineplot(ax=axs_imputed[0], data=data_B, x="flight_cycle", y=t, hue="uav_id", alpha=0.33, legend=False)
    #     sb.lineplot(ax=axs_imputed[1], data=data_C, x="flight_cycle", y=t, hue="uav_id", alpha=0.33, legend=False)
    #     fig.tight_layout()
    #     fig.savefig(PLOT_DIR / "telemetry_imputed_relplots" / f"{t}.png")
    #     plt.close()
    return


@app.cell
def _():
    # for t in telemetry_columns(data_C):
    #     print(f"Plotting: {t}.png")
    #     plot = sb.relplot(
    #         kind="line",
    #         data=data_C,
    #         x="flight_cycle",
    #         y=t,
    #         row="split",
    #         hue="uav_id",
    #         legend=False,
    #         alpha=0.33,
    #         height=6,
    #         aspect=2,
    #     )
    #     plot.savefig(PLOT_DIR / "telemetry_imputed_relplots" / f"{t}.png")
    #     plt.close()
    return


@app.cell
def _():
    # sb.relplot(
    #     data=data,
    #     kind="line",
    #     x="flight_cycle",
    #     y="telemetry_01",
    #     row="split",
    #     hue="uav_id",
    #     legend=False,
    #     alpha=0.33,
    #     height=6,
    #     aspect=2,
    # )
    return


@app.cell
def _():
    # from sklearn.feature_selection import mutual_info_regression
    # # from sklearn.ensemble import HistGradientBoostingRegressor

    # def rank_telemetry_features(df, target_col='RUL', sample_size=10000):
    #     telemetries = [c for c in df.columns if c.startswith('telemetry_')]
    #     sample_df = df.sample(min(sample_size, len(df)), random_state=42)
    
    #     # 1. Variance & Correlations
    #     std = df[telemetries].std()
    #     pearson = df[telemetries].apply(lambda x: x.corr(df[target_col])).abs()
    #     spearman = df[telemetries].apply(lambda x: x.corr(df[target_col], method='spearman')).abs()
    
    #     # 2. Mutual Information
    #     mi = mutual_info_regression(sample_df[telemetries], sample_df[target_col], random_state=42)
    
    #     # 3. Model Feature Importance
    #     model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
    #     model.fit(sample_df[telemetries], sample_df[target_col])
    
    #     rankings = pd.DataFrame({
    #         'std_dev': std,
    #         'pearson_abs': pearson,
    #         'spearman_abs': spearman,
    #         'mutual_info': mi
    #     }).sort_values(by='mutual_info', ascending=False)
    
    #     return rankings
    return


@app.cell
def _(HistGradientBoostingRegressor):
    # import pandas as pd
    # from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance

    def rank_telemetry_features(df, target_col='RUL', sample_size=10000):
        telemetries = [c for c in df.columns if c.startswith('telemetry_')]
        sample_df = df.sample(min(sample_size, len(df)), random_state=42)
    
        # 1. Fit GBDT Model
        model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
        model.fit(sample_df[telemetries], sample_df[target_col])
    
        # 2. Compute Permutation Importance
        perm = permutation_importance(
            model, 
            sample_df[telemetries], 
            sample_df[target_col], 
            n_repeats=5, 
            random_state=42
        )
    
        # 3. Assemble Full Ranking
        rankings = pd.DataFrame({
            'std_dev': df[telemetries].std(),
            'spearman_abs': df[telemetries].apply(lambda x: x.corr(df[target_col], method='spearman')).abs(),
            'gbdt_permutation_importance': perm.importances_mean
        }).sort_values(by='gbdt_permutation_importance', ascending=False)
    
        return rankings

    return (rank_telemetry_features,)


@app.cell
def _(rank_telemetry_features, train_raw):
    # Run ranking
    rank_telemetry_features(train_raw)
    return


@app.cell
def _(data_B, rank_telemetry_features):
    rank_telemetry_features(data_B[data_B["split"] == "train"])
    return


if __name__ == "__main__":
    app.run()
