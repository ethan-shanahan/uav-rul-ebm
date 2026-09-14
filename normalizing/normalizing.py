import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def run_normalize():
    print("Start normalisation")
    # === Parameter ===
    base = os.path.dirname(__file__)
    input_csv = [os.path.join(base, "..","data","denoised","train_denoised.csv"),
                 os.path.join(base, "..","data","denoised","test_denoised.csv")]
    output_csv = [os.path.join(base, "..", "data", "normalized", "normalized_train.csv"),
                  os.path.join(base, "..", "data", "normalized", "normalized_test.csv")]
    output_png = [os.path.join(base, "normalized_train.png"),
                  os.path.join(base, "normalized_test.png")]

    col_glob_min = []
    col_glob_max = []

    for i in range(len(input_csv)):
        # === CSV laden ===
        df = [pd.read_csv(input_csv[0]), pd.read_csv(input_csv[1])]

        # Spalten bestimmen
        telemetry_cols = [col for col in df[i].columns if col.startswith("telemetry_")]

        # copy for normalized (and if necessary mirrored) data
        df_norm = df[i].copy()


        #inverting all negative channels to positive
        channels_to_invert = ["telemetry_04", "telemetry_05", "telemetry_06", "telemetry_09", "telemetry_10", "telemetry_16",
                              "telemetry_18", "telemetry_23", "telemetry_24", "telemetry_25", "telemetry_26", "telemetry_28"]

        #telemetry_08, 14, 20 are negative but not in the list, because they are removed though uselessnes
        for col in channels_to_invert:
            df_norm[col] = -df_norm[col]

        #mirroring selected channels
        channels_to_flip = ["telemetry_11", "telemetry_18"]

        for col in channels_to_flip:
            col_min = df_norm[col].min()
            col_max = df_norm[col].max()
            df_norm[col] = (col_min + col_max) - df_norm[col]

        #fixing channels with trends in both directions
        flipped_uavs = set()
        channels_to_fix = ["telemetry_06", "telemetry_11", "telemetry_12", "telemetry_15", "telemetry_23", "telemetry_24", "telemetry_26"]

        for col in channels_to_fix:
            for uav in df_norm["uav_id"].unique():
                mask = df_norm["uav_id"] == uav
                series = df_norm.loc[mask, col]

                # calculating trend
                start = series.iloc[0]
                trend = series.iloc[-1] - start

                # mirror only UAVs which are decreasing
                if trend < 0:
                    flipped_uavs.add((uav, col))
                    df_norm.loc[mask, col] = 2 * start - series

        # === Normierung ===
        c = 0
        for col in telemetry_cols:
            if i == 0:
                col_glob_min.append(df_norm[col].min())
                col_glob_max.append(df_norm[col].max())
            df_norm[col] = (df_norm[col] - col_glob_min[c]) / (col_glob_max[c] - col_glob_min[c])
            c += 1


        # === Neue CSV speichern ===
        df_norm.to_csv(output_csv[i], index=False)

        # === Plotten ===
        #fig, axes = plt.subplots(7, 4, figsize=(22, 18))        #old scaling
        fig, axes = plt.subplots(6, 4, figsize=(22, 18))
        axes = axes.flatten()

        uavs = df[i]["uav_id"].unique()
        num_uavs = len(uavs)

        colors = plt.cm.turbo(np.linspace(0, 1, num_uavs))
        color_map = {uav: colors[j] for j, uav in enumerate(uavs)}

        for idx, col in enumerate(telemetry_cols):
            ax = axes[idx]

            for uav in uavs:
                data_uav = df_norm[df_norm["uav_id"] == uav][col]

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
        print("Normalisation ", i+1 ," of 2 complete\n")

if __name__ == "__main__":
    run_normalize()