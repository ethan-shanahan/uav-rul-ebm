import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 0) Daten einlesen
# ---------------------------------------------------------
# Passe die Pfade/Dateinamen an deine Struktur an
merged_train = pd.read_csv("train-test-data/merged_train_final.csv")
merged_val = pd.read_csv("train-test-data/merged_val_final.csv")
merged_val_cut = pd.read_csv("train-test-data/merged_val_cut_final.csv")
merged_test = pd.read_csv("train-test-data/merged_test_final.csv")

# ---------------------------------------------------------
# 1) Feature-Spalten definieren
# ---------------------------------------------------------
feature_cols = (
    ["flight_cycle"] +
    [f"merge_{i}" for i in range(6)] +              #number of merged channels
    [f"grad_merge_{i}" for i in range(6)]           #number of merged gradient channels
)

timesteps = 100  # Anzahl vergangener flight_cycles pro Sample
feature_dim = len(feature_cols)

MODEL_TYPE = "den_bat"      # oder "cnn_gru" oder "small_gru"


# ---------------------------------------------------------
# 2) Funktion zum Erstellen von Sequenzen pro UAV
# ---------------------------------------------------------
def build_sequences(df, timesteps, feature_cols, include_target=True):
    X_list = []
    y_list = []
    id_list = []

    for uav_id in df["uav_id"].unique():
        df_uav = df[df["uav_id"] == uav_id].sort_values("flight_cycle")

        values = df_uav[feature_cols].values
        if include_target:
            rul_values = df_uav["RUL"].values

        # ---------------------------------------------------------
        # 1) UAV hat weniger Daten als timesteps → Padding
        # ---------------------------------------------------------
        if len(df_uav) < timesteps:
            pad_len = timesteps - len(df_uav)
            pad = np.zeros((pad_len, len(feature_cols)))
            seq = np.vstack([pad, values])  # vorne auffüllen

            X_list.append(seq)
            id_list.append(uav_id)

            if include_target:
                y_list.append(rul_values[-1])  # letzter RUL zählt
            continue

        # ---------------------------------------------------------
        # 2) Normale Sliding-Window-Sequenzen
        # ---------------------------------------------------------
        for i in range(len(df_uav) - timesteps + 1):
            seq = values[i:i+timesteps]
            X_list.append(seq)
            id_list.append(uav_id)

            if include_target:
                y_list.append(rul_values[i+timesteps-1])

    if include_target:
        return np.array(X_list), np.array(y_list), np.array(id_list)
    else:
        return np.array(X_list), np.array(id_list)

# ---------------------------------------------------------
# 3) Sequenzen bauen
# ---------------------------------------------------------
X_train, y_train, train_ids = build_sequences(merged_train, timesteps, feature_cols, include_target=True)
X_val, y_val, val_ids = build_sequences(merged_val, timesteps, feature_cols, include_target=True)
X_val_cut, y_val_cut, val_cut_ids = build_sequences(merged_val_cut, timesteps, feature_cols, include_target=True)
X_test, uav_test_ids = build_sequences(merged_test, timesteps, feature_cols, include_target=False)

# ---------------------------------------------------------
# 4) GRU-Modell definieren
# ---------------------------------------------------------
def build_model(model_type: str, timesteps: int, feature_dim: int):
    if model_type == "gru":
        # Dein verbessertes GRU-Modell
        model = models.Sequential([
            layers.Input(shape=(timesteps, feature_dim)),
            layers.GRU(256, return_sequences=True),
            layers.GRU(128, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(128, activation="relu"),
            layers.Dense(1)
        ])

    elif model_type == "small_gru":
        # Dein verbessertes GRU-Modell
        model = models.Sequential([
            layers.Input(shape=(timesteps, feature_dim)),
            layers.GRU(64, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(64, activation="relu"),
            layers.Dense(1)
        ])

    elif model_type == "cnn_gru":
        # CNN-GRU-Hybrid
        model = models.Sequential([
            layers.Input(shape=(timesteps, feature_dim)),

            # CNN über die Zeitachse (flight_cycles)
            layers.Conv1D(filters=64, kernel_size=5, padding="same", activation="relu"),
            layers.Conv1D(filters=64, kernel_size=5, padding="same", activation="relu"),
            layers.MaxPooling1D(pool_size=2),

            # GRU auf den gefilterten Sequenzen
            layers.GRU(128, return_sequences=True),
            layers.GRU(64, return_sequences=False),

            layers.Dropout(0.2),
            layers.Dense(64, activation="relu"),
            layers.Dense(1)
        ])

    else:
        raise ValueError(f"Unbekannter model_type: {model_type}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-1),
        loss="mse",
        metrics=["mae"]
    )
    return model
model = build_model(MODEL_TYPE, timesteps, feature_dim)
# ---------------------------------------------------------
# R² Funktion
# ---------------------------------------------------------
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res/ss_tot if ss_tot != 0 else 0

# ---------------------------------------------------------
# 5) Training mit R² auf val_cut
# ---------------------------------------------------------
history = {
    "train_loss": [],
    "val_loss": [],
    "val_cut_loss": [],
    "r2_val_cut": [],
    "r2_val": []
}

EPOCHS = 30
BATCH = 64

for epoch in range(EPOCHS):
    hist = model.fit(
        X_train, y_train,
        batch_size=BATCH,
        epochs=1,
        verbose=0
    )

    train_loss = hist.history["loss"][0]
    val_loss = model.evaluate(X_val, y_val, verbose=0)[0]
    val_cut_loss = model.evaluate(X_val_cut, y_val_cut, verbose=0)[0]

    y_val_cut_pred = model.predict(X_val_cut, verbose=0).flatten()
    # R² berechnen
    r2_val_cut = r2_score(y_val_cut, y_val_cut_pred)

    y_val_pred = model.predict(X_val, verbose=0).flatten()
    r2_val = r2_score(y_val, y_val_pred)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_cut_loss"].append(val_cut_loss)
    history["r2_val"].append(r2_val)
    history["r2_val_cut"].append(r2_val_cut)

    print(
        f"Epoch {epoch+1}/{EPOCHS}  "
        f"Train={train_loss:.4f}  "
        f"Val={val_loss:.4f}  "
        f"ValCut={val_cut_loss:.4f}  "
        f"R2_ValCut={r2_val_cut:.4f}"
    )

# ---------------------------------------------------------
# 6) R² Verlauf plotten
# ---------------------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(history["r2_val"])
plt.plot(history["r2_val_cut"])
plt.xlabel("Epoch")
plt.ylabel("R²")
plt.title("R² Verlauf (Val & Val-Cut)")
plt.show()

# ---------------------------------------------------------
# 7) Modell auf Test anwenden
# ---------------------------------------------------------
y_pred_test = model.predict(X_test).flatten()

# ---------------------------------------------------------
# 8) Test-RUL speichern
# ---------------------------------------------------------
df_test_pred = pd.DataFrame({
    "id": uav_test_ids,
    "RUL": y_pred_test
})

df_test_last = df_test_pred.groupby("id").tail(1)

df_test_last.to_csv("predicted_test_rul_gru.csv", index=False)
print("Test-RUL gespeichert in predicted_test_rul_gru.csv")