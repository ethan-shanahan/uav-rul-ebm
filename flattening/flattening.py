import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def run_flattening():
    print("\nStart flattening")
    base = os.path.dirname(__file__)
    input_csv = [os.path.join(base, "..", "data", "merged", "merged_train.csv"),
                 os.path.join(base, "..", "data", "merged", "merged_test.csv")]
    output_csv = [os.path.join(base, "..", "data", "flattened", "flattened_train.csv"),
                  os.path.join(base, "..", "data", "flattened", "flattened_test.csv")]
    output_png = [os.path.join(base, "flattened_train.png"),
                  os.path.join(base, "flattened_test.png")]

    for i in range(len(input_csv)):
        df = pd.read_csv(input_csv[i])

        telemetry_cols = [col for col in df.columns if col.startswith("merge_")]

        window = 5   # flattening window
        variant = 5  #choosing flattening variation 1-5

        df_smooth = df.copy()

        #variant 1 average
        if variant == 1:
            for col in telemetry_cols:
                for uav in df["uav_id"].unique():
                    mask = df["uav_id"] == uav
                    df_smooth.loc[mask, col] = df.loc[mask, col].rolling(window=window, center=True).mean()

        #variant 2 median
        if variant == 2:
            for col in telemetry_cols:
                for uav in df["uav_id"].unique():
                    mask = df["uav_id"] == uav
                    df_smooth.loc[mask, col] = df.loc[mask, col].rolling(window=window, center=True).median()

        #variant 3 Savitzky-Golay
        if variant == 3:
            from scipy.signal import savgol_filter
            for col in telemetry_cols:
                for uav in df["uav_id"].unique():
                    mask = df["uav_id"] == uav
                    series = df.loc[mask, col].values

                    df_smooth.loc[mask, col] = savgol_filter(series, window_length=7, polyorder=2)

        # lowess/lowpass
        if variant == 4:
            from scipy.signal import butter, filtfilt

            b, a = butter(N=3, Wn=0.05, btype='low', output='ba')  # Order 3 (higher = stronger, but maye oscillation),
            for col in telemetry_cols:                             # Cutoff 0.05 (lower = stronger)
                for uav in df["uav_id"].unique():
                    mask = df["uav_id"] == uav
                    series = df.loc[mask, col].values

                    df_smooth.loc[mask, col] = filtfilt(b, a, series)

        #ema
        if variant == 5:
            alpha = 0.05  # flattening factor, lower = stronger
            for col in telemetry_cols:
                for uav in df["uav_id"].unique():
                    mask = df["uav_id"] == uav
                    series = df.loc[mask, col]

                    df_smooth.loc[mask, col] = series.ewm(alpha=alpha).mean().values

        df_smooth.to_csv(output_csv[i], index=False)


        # === Plotten ===
        fig, axes = plt.subplots(2, 3, figsize=(22, 18))
        axes = axes.flatten()

        uavs = df["uav_id"].unique()
        num_uavs = len(uavs)

        colors = plt.cm.turbo(np.linspace(0, 1, num_uavs))
        color_map = {uav: colors[j] for j, uav in enumerate(uavs)}

        for idx, col in enumerate(telemetry_cols):
            ax = axes[idx]

            for uav in uavs:
                data_uav = df_smooth[df_smooth["uav_id"] == uav][col]

                ax.plot(
                    data_uav.values,
                    linewidth=0.3,
                    alpha=0.35,
                    color=color_map[uav]
                )

            ax.set_title(col, fontsize=10)
            ax.grid(True)
            ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(output_png[i], dpi=300, bbox_inches="tight")
        #plt.show()
        print("Flattening ", i+1, " of 2 complete")

if __name__ == "__main__":
    run_flattening()