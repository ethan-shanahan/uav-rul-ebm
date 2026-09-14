import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1) CSVs laden
# ---------------------------------------------------------
train_df      = pd.read_csv("../data/preprocessed/preprocessed_train.csv")
test_df       = pd.read_csv("../data/preprocessed/preprocessed_test.csv")
grad_train_df = pd.read_csv("../data/preprocessed/gradient_train.csv")
grad_test_df  = pd.read_csv("../data/preprocessed/gradient_test.csv")

# ---------------------------------------------------------
# 2) UAV-Liste mischen
# ---------------------------------------------------------
uavs = train_df["uav_id"].unique().tolist()
np.random.shuffle(uavs)

# 80:20 UAV-Split
split_idx = int(len(uavs) * 0.8)
train_uavs = set(uavs[:split_idx])
val_uavs   = set(uavs[split_idx:])

print("Train UAVs:", len(train_uavs), train_uavs)
print("Val UAVs:", len(val_uavs), val_uavs)

# ---------------------------------------------------------
# 3) UAV-Level-Split anwenden
# ---------------------------------------------------------
train_split      = train_df[train_df["uav_id"].isin(train_uavs)].copy()
val_split        = train_df[train_df["uav_id"].isin(val_uavs)].copy()

grad_train_split = grad_train_df[grad_train_df["uav_id"].isin(train_uavs)].copy()
grad_val_split   = grad_train_df[grad_train_df["uav_id"].isin(val_uavs)].copy()

print("Train-Split:", train_split.shape)
print("Val-Split:", val_split.shape)
print("Grad-Train-Split:", grad_train_split.shape)
print("Grad-Val-Split:", grad_val_split.shape)

# ---------------------------------------------------------
# 4) Maximale Test-Länge bestimmen
# ---------------------------------------------------------
max_test_len = test_df.groupby("uav_id")["flight_cycle"].max().max()
min_test_len = test_df.groupby("uav_id")["flight_cycle"].max().min()
print("Maximale Test-Länge:", max_test_len)
print("Minimale Test-Länge:", min_test_len)

# ---------------------------------------------------------
# 5) Val-Cut erzeugen (nur für Val-UAVs)
# ---------------------------------------------------------
val_cut_list      = []
grad_val_cut_list = []

for uav in val_uavs:
    grp  = val_split[val_split["uav_id"] == uav].sort_values("flight_cycle")
    ggrp = grad_val_split[grad_val_split["uav_id"] == uav].sort_values("flight_cycle")

    # Zufällige Ziel-Länge (mind. Min_Test-Länge, max. Test-Länge)
    target_len = np.random.randint(min_test_len, max_test_len + 1)

    val_cut_list.append(grp.iloc[:target_len])
    grad_val_cut_list.append(ggrp.iloc[:target_len])

val_cut      = pd.concat(val_cut_list).reset_index(drop=True)
grad_val_cut = pd.concat(grad_val_cut_list).reset_index(drop=True)

print("Val-Cut:", val_cut.shape)
print("Grad-Val-Cut:", grad_val_cut.shape)

# ---------------------------------------------------------
# 6) Gradient-Spalten ersetzen (merge_x -> grad_merge_x)
# ---------------------------------------------------------
def replace_grad_columns(df):
    new_df = df.copy()
    for c in df.columns:
        if c.startswith("merge_"):
            new_df["grad_" + c] = new_df[c]
            new_df = new_df.drop(columns=[c])
    return new_df

grad_train_split = replace_grad_columns(grad_train_split)
grad_val_split   = replace_grad_columns(grad_val_split)
grad_val_cut     = replace_grad_columns(grad_val_cut)
grad_test_df     = replace_grad_columns(grad_test_df)

# ---------------------------------------------------------
# 7) Merge-Funktion: nur Gradienten-Spalten anhängen
# ---------------------------------------------------------
def merge_sets(df_main, df_grad):
    grad_cols = [c for c in df_grad.columns if c.startswith("grad_merge_")]
    return df_main.merge(
        df_grad[["uav_id", "flight_cycle"] + grad_cols],
        on=["uav_id", "flight_cycle"],
        how="inner"
    )

merged_train   = merge_sets(train_split, grad_train_split)
merged_val     = merge_sets(val_split, grad_val_split)
merged_val_cut = merge_sets(val_cut, grad_val_cut)
merged_test    = merge_sets(test_df, grad_test_df)

# ---------------------------------------------------------
# 8) RUL aus den Original-Splits wieder anhängen (falls verloren)
# ---------------------------------------------------------
def add_rul_back(merged_df, source_df):
    if "RUL" not in merged_df.columns and "RUL" in source_df.columns:
        return merged_df.merge(
            source_df[["uav_id", "flight_cycle", "RUL"]],
            on=["uav_id", "flight_cycle"],
            how="left"
        )
    return merged_df

merged_train   = add_rul_back(merged_train, train_split)
merged_val     = add_rul_back(merged_val, val_split)
merged_val_cut = add_rul_back(merged_val_cut, val_cut)
# merged_test hat kein RUL → bleibt wie es ist

# ---------------------------------------------------------
# 9) Spalten final ordnen
# ---------------------------------------------------------
def reorder_final(df):
    base   = ["uav_id", "flight_cycle"]
    merges = [c for c in df.columns if c.startswith("merge_")]
    grads  = [c for c in df.columns if c.startswith("grad_merge_")]

    if "RUL" in df.columns:
        return df[["RUL"] + base + merges + grads]
    else:
        return df[base + merges + grads]

merged_train   = reorder_final(merged_train)
merged_val     = reorder_final(merged_val)
merged_val_cut = reorder_final(merged_val_cut)
merged_test    = reorder_final(merged_test)

# ---------------------------------------------------------
# 10) Speichern
# ---------------------------------------------------------
merged_train.to_csv("train-test-data/merged_train_final.csv", index=False)
merged_val.to_csv("train-test-data/merged_val_final.csv", index=False)
merged_val_cut.to_csv("train-test-data/merged_val_cut_final.csv", index=False)
merged_test.to_csv("train-test-data/merged_test_final.csv", index=False)