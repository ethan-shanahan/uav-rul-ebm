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

    mode = "no"  # "moderate" oder "ema" oder "no"
    use_fit = False  # Fit aktivieren oder deaktivieren

    def smooth_gradient(grad, mode):
        if mode == "moderate":
            return pd.Series(grad).rolling(
                window=7, center=True, min_periods=1
            ).mean().values

        elif mode == "ema":
            alpha = 0.1
            return pd.Series(grad).ewm(alpha=alpha, adjust=False).mean().values


        return grad

    for i in range(len(input_csv)):
        # CSV laden
        df = pd.read_csv(input_csv[i])

        # Telemetry-Spalten
        #merge_cols = [f"merge_{j}" for j in range(6)]       #number of merge colums, to see in merge.py
        merge_cols = [col for col in df.columns if col.startswith("merge_")]

        df_grad = df.copy()

        # Gradient berechnen

        for col in merge_cols:
            grads_all = []

            # UAVs einzeln verarbeiten
            for uav_id, df_uav in df.groupby("uav_id"):
                sig = df_uav[col].values  # Signal ist bereits glatt
                n = len(sig)

                # 1) Gradient berechnen
                grad = np.gradient(sig)

                # 2) Gradient glätten
                grad = smooth_gradient(grad, mode)

                # 3) Fit-Parameter
                k_fit = 50
                k_tail = 25

                if use_fit and n > k_fit and n > k_tail:
                    # Quadratischer Fit auf die letzten 50 Signalwerte
                    x_fit = np.arange(n - k_fit, n)
                    y_fit = sig[n - k_fit:n]

                    coeffs = np.polyfit(x_fit, y_fit, deg=2)
                    p = np.poly1d(coeffs)

                    # Ableitung des Fits für die letzten 25 Werte
                    p_deriv = np.polyder(p)
                    x_tail = np.arange(n - k_tail, n)
                    grad_tail = p_deriv(x_tail)

                    # Ersetzen der letzten 25 Gradient-Werte
                    grad[n - k_tail:n] = grad_tail

                grads_all.extend(grad)

            df_grad[col] = grads_all

        '''for col in merge_cols:
            grads_all = []

            # UAVs einzeln verarbeiten
            for uav_id, df_uav in df.groupby("uav_id"):
                sig = df_uav[col].values
                n = len(sig)

                # 1) einfache Ableitung
                grad = np.gradient(sig)

                # 2) Fit-Parameter
                k_fit = 5
                k_tail = 2

                # Falls UAV zu kurz ist → nichts ersetzen
                if n > k_fit and n > k_tail:
                    # Quadratischer Fit auf die letzten 50 Signalwerte dieses UAVs
                    x_fit = np.arange(n - k_fit, n)
                    y_fit = sig[n - k_fit:n]

                    coeffs = np.polyfit(x_fit, y_fit, deg=2)
                    p = np.poly1d(coeffs)

                    # Ableitung des Fits für die letzten 25 Werte
                    p_deriv = np.polyder(p)
                    x_tail = np.arange(n - k_tail, n)
                    grad_tail = p_deriv(x_tail)

                    # Ersetzen der letzten 25 Gradient-Werte
                    grad[n - k_tail:n] = grad_tail

                grads_all.extend(grad)

            df_grad[col] = grads_all'''

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