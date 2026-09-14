import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def run_merge():
    print("\nStart merging")
    base = os.path.dirname(__file__)
    input_csv = [os.path.join(base, "..", "data", "normalized", "normalized_train.csv"),
                 os.path.join(base, "..", "data", "normalized", "normalized_test.csv")]
    output_csv = [os.path.join(base, "..", "data", "merged", "merged_train.csv"),
                  os.path.join(base, "..", "data", "merged", "merged_test.csv")]
    output_png = [os.path.join(base, "merged_train.png"),
                  os.path.join(base, "merged_test.png")]

    for i in range(len(input_csv)):
        df = pd.read_csv(input_csv[i])

        #columns to merge
        merge0 = ["telemetry_25", "telemetry_28"]
        merge1 = ["telemetry_06", "telemetry_11", "telemetry_12"]
        merge2 = ["telemetry_15", "telemetry_23"]
        merge3 = ["telemetry_24", "telemetry_26"]
        merge4 = ["telemetry_19", "telemetry_21"]
        merge5 = ["telemetry_13", "telemetry_22"]

        merges = [merge0, merge1, merge2, merge3, merge4, merge5]

        df_merged = df.copy()

        for j, merge_cols in enumerate(merges):
            new_col = f"merge_{j}"
            df_merged[new_col] = df[merge_cols].mean(axis=1)

        if i == 0:
            cols = ["uav_id", "flight_cycle"] + [f"merge_{j}" for j in range(len(merges))] + ["RUL"]
        else:
            cols = ["uav_id", "flight_cycle"] + [f"merge_{j}" for j in range(len(merges))]

        df_merged = df_merged[cols]
        df_merged.to_csv(output_csv[i], index=False)

        # === Plotten ===
        fig, axes = plt.subplots(2, 3, figsize=(22, 18))
        axes = axes.flatten()

        uavs = df["uav_id"].unique()
        num_uavs = len(uavs)

        colors = plt.cm.turbo(np.linspace(0, 1, num_uavs))
        color_map = {uav: colors[j] for j, uav in enumerate(uavs)}

        merge_cols = [f"merge_{j}" for j in range(len(merges))]

        for idx, col in enumerate(merge_cols):
            ax = axes[idx]

            for uav in uavs:
                data_uav = df_merged[df_merged["uav_id"] == uav][col]

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
        print("Merge ", i+1, " of 2 complete")

if __name__ == "__main__":
    run_merge()