import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def run_gradient():
    print("\nRunning gradient calculation")
    base = os.path.dirname(__file__)
    input_csv = [os.path.join(base, "..", "data", "flattened", "flattened_train.csv"),
                 os.path.join(base, "..", "data", "flattened", "flattened_test.csv")]
    output_csv = [os.path.join(base, "..", "data", "gradient", "gradient_train.csv"),
                  os.path.join(base, "..", "data", "gradient", "gradient_test.csv")]
    output_pre = [os.path.join(base, "..", "data", "preprocessed", "gradient_train.csv"),
                  os.path.join(base, "..", "data", "preprocessed", "gradient_test.csv")]
    output_png = [os.path.join(base, "gradient_train.png"),
                  os.path.join(base, "gradient_test.png")]
    output_png_pre = [os.path.join(base, "..", "data", "preprocessed", "gradient_train.png"),
                      os.path.join(base, "..", "data", "preprocessed", "gradient_test.png")]

    for i in range(len(input_csv)):
        # CSV laden
        df = pd.read_csv(input_csv[i])

        # Telemetry-Spalten
        merge_cols = [f"merge_{j}" for j in range(6)]

        df_grad = df.copy()

        # Gradient berechnen
        for col in merge_cols:
            df_grad[f"{col}_grad"] = np.gradient(df_grad[col].values)

        df_grad.to_csv(output_csv[i], index=False)
        df_grad.to_csv(output_pre[i], index=False)


        # === Plotten ===
        fig, axes = plt.subplots(2, 3, figsize=(22, 18))
        axes = axes.flatten()

        uavs = df["uav_id"].unique()
        num_uavs = len(uavs)

        colors = plt.cm.turbo(np.linspace(0, 1, num_uavs))
        color_map = {uav: colors[j] for j, uav in enumerate(uavs)}

        for idx, col in enumerate(merge_cols):
            ax = axes[idx]

            for uav in uavs:
                data_uav = df_grad[df_grad["uav_id"] == uav][col]

                ax.plot(
                    data_uav.values,
                    linewidth=0.3,
                    alpha=0.35,
                    color=color_map[uav]
                )

            ax.set_title(col, fontsize=10)
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(output_png[i], dpi=300, bbox_inches="tight")
        plt.savefig(output_png_pre[i], dpi=300, bbox_inches="tight")
        #plt.show()
        print("Gradient ", i+1, " of 2 complete")

if __name__ == "__main__":
    run_gradient()