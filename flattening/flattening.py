import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

input_csv = "../data/normalized/normalized_train.csv"
output_csv = "../data/flattened/flattened_.csv"

df = pd.read_csv(input_csv)

telemetry_cols = [col for col in df.columns if col.startswith("telemetry_")]

window = 5   # flattening window
variant = 1  #choosing flattening variation

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

df_smooth.to_csv(output_csv, index=False)


# === Plotten ===
fig, axes = plt.subplots(6, 4, figsize=(22, 18))
axes = axes.flatten()

uavs = df["uav_id"].unique()
num_uavs = len(uavs)

colors = plt.cm.turbo(np.linspace(0, 1, num_uavs))
color_map = {uav: colors[i] for i, uav in enumerate(uavs)}

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

plt.tight_layout()
plt.savefig("output.png", dpi=300, bbox_inches="tight")
plt.show()