import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sb
    import marimo as mo

    from sklearn.pipeline import Pipeline

    from uav_rul_ebm import new_trans as trans
    from scrap.utils_deprecated import telemetry_columns

    PROJECT_ROOT = Path.cwd()
    TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
    TEST_PATH = PROJECT_ROOT / "data" / "raw" / "test.csv"
    PLOT_DIR = PROJECT_ROOT / "eda" / "plots"


@app.cell
def _():
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    return test_raw, train_raw


@app.cell
def _():
    pp = Pipeline(
        [
            ("variance threshold", trans.VarianceThreshold(verbose=True)),
            (
                "outlier nullifier",
                trans.VerticalHampelFilter(
                    rolling_window_size=30, n_sigmas=5, verbose=False
                ),
            ),
            (
                "defect nullifier",
                trans.HorizontalHampelFilter(
                    init_window_size=40, n_sigmas=5, verbose=True
                ),
            ),
            ("imputer", trans.HGBIImputer()),
        ],
        verbose=True,
    )
    return (pp,)


@app.cell
def _(pp, test_raw, train_raw):
    train = pp.fit_transform(train_raw)
    test = pp.transform(test_raw)
    train.to_csv(PROJECT_ROOT / "data" / "processed" / "train_processed_13.csv")
    test.to_csv(PROJECT_ROOT / "data" / "processed" / "test_processed_13.csv")
    return test, train


@app.cell
def _():
    # t01 = set(['UAV_0001', 'UAV_0035', 'UAV_0068', 'UAV_0072', 'UAV_0076', 'UAV_0078', 'UAV_0085', 'UAV_0089'])
    # t18 = set(['UAV_0001', 'UAV_0002', 'UAV_0007', 'UAV_0009', 'UAV_0010', 'UAV_0011', 'UAV_0017', 'UAV_0018', 'UAV_0020', 'UAV_0021', 'UAV_0024', 'UAV_0027', 'UAV_0033', 'UAV_0034', 'UAV_0035', 'UAV_0037', 'UAV_0038', 'UAV_0039', 'UAV_0041', 'UAV_0042', 'UAV_0043', 'UAV_0045', 'UAV_0046', 'UAV_0049', 'UAV_0055', 'UAV_0060', 'UAV_0062', 'UAV_0068', 'UAV_0071', 'UAV_0072', 'UAV_0073', 'UAV_0075', 'UAV_0076', 'UAV_0077', 'UAV_0078', 'UAV_0081', 'UAV_0082', 'UAV_0084', 'UAV_0085', 'UAV_0088', 'UAV_0089', 'UAV_0094', 'UAV_0096', 'UAV_0097', 'UAV_0098'])

    # t01.intersection(t18)
    return


@app.cell
def _():
    # t01 = ['UAV_0001', 'UAV_0035', 'UAV_0068', 'UAV_0072', 'UAV_0076', 'UAV_0078', 'UAV_0085', 'UAV_0089']
    # t18 = ['UAV_0001', 'UAV_0002', 'UAV_0007', 'UAV_0009', 'UAV_0010', 'UAV_0011', 'UAV_0017', 'UAV_0018', 'UAV_0020', 'UAV_0021', 'UAV_0024', 'UAV_0027', 'UAV_0033', 'UAV_0034', 'UAV_0035', 'UAV_0037', 'UAV_0038', 'UAV_0039', 'UAV_0041', 'UAV_0042', 'UAV_0043', 'UAV_0045', 'UAV_0046', 'UAV_0049', 'UAV_0055', 'UAV_0060', 'UAV_0062', 'UAV_0068', 'UAV_0071', 'UAV_0072', 'UAV_0073', 'UAV_0075', 'UAV_0076', 'UAV_0077', 'UAV_0078', 'UAV_0081', 'UAV_0082', 'UAV_0084', 'UAV_0085', 'UAV_0088', 'UAV_0089', 'UAV_0094', 'UAV_0096', 'UAV_0097', 'UAV_0098']
    # t19 = ['UAV_0024', 'UAV_0042', 'UAV_0055', 'UAV_0071', 'UAV_0073']
    # t21 = ['UAV_0024', 'UAV_0042', 'UAV_0055', 'UAV_0071', 'UAV_0073']

    # ts = {'t01': t01, 't18': t18, 't19': t19, 't21': t21}

    # from itertools import chain
    # from collections import Counter

    # Counter(chain(*ts.values())).items()

    # [uav for uav, cnt in Counter(chain(*ts.values())).items() if cnt > 1]
    return


@app.cell
def _(test, test_raw, train, train_raw):
    df_train_raw = train_raw.assign(split="train")
    df_train_raw = df_train_raw.assign(state="raw")
    df_test_raw = test_raw.assign(split="test")
    df_test_raw = df_test_raw.assign(state="raw")
    df_train = train.assign(split="train")
    df_train = df_train.assign(state="processed")
    df_test = test.assign(split="test")
    df_test = df_test.assign(state="processed")
    data = pd.concat([df_train_raw, df_test_raw, df_train, df_test], ignore_index=True)
    return


@app.cell
def _():
    # tlm = "telemetry_05"
    # sb.relplot(
    #     kind = "line",
    #     data=data, x="flight_cycle", y=tlm,
    #     hue="uav_id", row="state", col="split",
    #     alpha=0.33, legend=False
    # )
    return


@app.cell
def _():
    # for t in telemetry_columns(data):
    #     print(f"Plotting: {t}.png")
    #     plot = sb.relplot(
    #         kind = "line",
    #         data=data, x="flight_cycle", y=t,
    #         hue="uav_id", row="state", col="split",
    #         alpha=0.33, legend=False
    #     )
    #     plot.savefig(PLOT_DIR / "4-step-pipeline" / f"{t}.png")
    #     plt.close()
    return


@app.cell
def _():
    # t = "telemetry_04"
    # is_train_outlier = pd.read_csv(f"./data/outliers/a/train/{t}.csv")
    # is_test_outlier = pd.read_csv(f"./data/outliers/a/test/{t}.csv")
    # pd.concat([is_train_outlier, is_test_outlier], ignore_index=True)

    # outliers = data \
    #     .loc[data["state"] == "raw"].reset_index() \
    #     .loc[
    #         pd.concat([is_train_outlier, is_test_outlier], ignore_index=True).any(axis=1),
    #         ["uav_id", "flight_cycle", "split", t]
    #     ]
    # sb.relplot(
    #     data=outliers, x="flight_cycle", y=t,
    #     hue="uav_id", col="split",
    #     alpha=0.33, legend=False,
    #     # height=6, aspect=1
    # )
    return


@app.cell
def _():
    # for t in telemetry_columns(data):
    #     print(f"Plotting: {t}.png")
    #     is_train_outlier = pd.read_csv(f"./data/outliers/a/train/{t}.csv")
    #     is_test_outlier = pd.read_csv(f"./data/outliers/a/test/{t}.csv")
    #     outliers = data \
    #         .loc[data["state"] == "raw"].reset_index() \
    #         .loc[
    #             pd.concat([is_train_outlier, is_test_outlier], ignore_index=True).any(axis=1),
    #             ["uav_id", "flight_cycle", "split", t]
    #         ]
    #     plot = sb.relplot(
    #         data=outliers, x="flight_cycle", y=t,
    #         hue="uav_id", col="split",
    #         alpha=0.33, legend=False,
    #         height=6, aspect=1
    #     )
    #     plot.savefig(PLOT_DIR / "4-step-pipeline" / "diff" / f"{t}.png")
    #     plt.close()
    return


@app.cell
def _():
    # print(f"Plotting: {t}.png")
    # fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12,12))
    # sb.lineplot(ax=axs[0,0], data=train_raw, x="flight_cycle", y=t, hue="uav_id", alpha=0.33, legend=False)
    # sb.lineplot(ax=axs[0,1], data=test_raw, x="flight_cycle", y=t, hue="uav_id", alpha=0.33, legend=False)
    # try:
    #     sb.lineplot(ax=axs[1], data=train, x="flight_cycle", y=t, hue="uav_id", alpha=0.33, legend=False)
    # except ValueError as e:
    #     axs[1].text(0.5, 0.5, "telemetry removed", size="xx-large", horizontalalignment="center", verticalalignment="center", transform=axs[1].transAxes)
    # fig.tight_layout()
    # plt.show()
    return


@app.cell
def _():
    # print(f"Plotting: {t}.png")
    # fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(12,12))
    # sb.lineplot(ax=axs[0], data=test_raw, x="flight_cycle", y=t, hue="uav_id", alpha=0.33, legend=False)
    # try:
    #     sb.lineplot(ax=axs[1], data=test, x="flight_cycle", y=t, hue="uav_id", alpha=0.33, legend=False)
    # except ValueError as e:
    #     axs[1].text(0.5, 0.5, "telemetry removed", size="xx-large", horizontalalignment="center", verticalalignment="center", transform=axs[1].transAxes)
    # fig.tight_layout()
    # plt.show()
    return


@app.cell
def _():
    # for t in telemetry_columns(test_raw):
    #     print(f"Plotting: {t}.png")
    #     fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(12,12))
    #     sb.lineplot(ax=axs[0], data=test_raw, x="flight_cycle", y=t, hue="uav_id", alpha=0.33, legend=False)
    #     try:
    #         sb.lineplot(ax=axs[1], data=test, x="flight_cycle", y=t, hue="uav_id", alpha=0.33, legend=False)
    #     except ValueError as e:
    #         axs[1].text(0.5, 0.5, "telemetry removed", size="xx-large", horizontalalignment="center", verticalalignment="center", transform=axs[1].transAxes)
    #     fig.tight_layout()
    #     fig.savefig(PLOT_DIR / "4-step-pipeline" / f"{t}.png")
    #     plt.close()
    return


if __name__ == "__main__":
    app.run()
