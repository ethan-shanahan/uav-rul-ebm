import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

input_csv = "../data/normalized/normalized_train.csv"
output_csv = "../data/merged/merged_output.csv"

df = pd.read_csv(input_csv)

#columns to merge
merge0 = ["telemetry_25", "telemetry_28"]
merge1 = ["telemetry_06", "telemetry_11", "telemetry_12"]
merge2 = ["telemetry_15", "telemetry_23"]
merge3 = ["telemetry_24", "telemetry_26"]
merge4 = ["telemetry_19", "telemetry_21"]
merge5 = ["telemetry_13", "telemetry_22"]

merges = [merge0, merge1, merge2, merge3, merge4, merge5]

df_merged = df.copy()

for i, merge_cols in enumerate(merges):
    new_col = f"merge_{i}"
    df_merged[new_col] = df[merge_cols].mean(axis=1)

df_merged = df_merged[["uav_id", "flight_cycle", "RUL"] + [f"merge_{i}" for i in range(len(merges))]]

df_merged.to_csv(output_csv, index=False)

# === Plotten ===
fig, axes = plt.subplots(2, 3, figsize=(22, 18))
axes = axes.flatten()

uavs = df["uav_id"].unique()
num_uavs = len(uavs)

colors = plt.cm.turbo(np.linspace(0, 1, num_uavs))
color_map = {uav: colors[i] for i, uav in enumerate(uavs)}

merge_cols = [f"merge_{i}" for i in range(len(merges))]

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

plt.tight_layout()
plt.savefig("output.png", dpi=300, bbox_inches="tight")
plt.show()