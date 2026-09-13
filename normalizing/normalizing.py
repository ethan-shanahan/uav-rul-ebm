import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Parameter ===
input_csv = "../data/denoised/train_denoised.csv"
output_csv = "../data/normalized/normalized.csv"

# === CSV laden ===
df = pd.read_csv(input_csv)

# Spalten bestimmen
telemetry_cols = [col for col in df.columns if col.startswith("telemetry_")]

# copy for normalized (and if necessary mirrored) data
df_norm = df.copy()


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

#fixing channel 6, 11 and 12
flipped_uavs = set()
channels_to_fix = ["telemetry_06", "telemetry_11", "telemetry_12"]

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
for col in telemetry_cols:
    col_min = df_norm[col].min()
    col_max = df_norm[col].max()
    df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)


# === Neue CSV speichern ===
df_norm.to_csv(output_csv, index=False)

# === Plotten ===
#fig, axes = plt.subplots(7, 4, figsize=(22, 18))        #old scaling
fig, axes = plt.subplots(6, 4, figsize=(22, 18))
axes = axes.flatten()

uavs = df["uav_id"].unique()
num_uavs = len(uavs)

colors = plt.cm.turbo(np.linspace(0, 1, num_uavs))
color_map = {uav: colors[i] for i, uav in enumerate(uavs)}

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

plt.tight_layout()
plt.savefig("output.png", dpi=300, bbox_inches="tight")
#plt.show()