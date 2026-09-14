#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UAV Latent Space + Clustering + Feature Engineering
TRAIN + TEST kompatibel

Neu:
------------------------------------------------------------
- Datensatz beim Start auswählen:
    * gefilterte Trainingsdaten
    * originale train.csv
    * test.csv
    * eigener CSV-Pfad

- Telemetrien frei auswählen:
    * 1
    * 1,5,8
    * 1-10
    * 1,5,10-15
    * all

- test.csv darf OHNE RUL-Spalte vorliegen.
  Dann werden alle RUL-Auswertungen automatisch deaktiviert.

- Bei Testdaten zwei Modi:
    1) Test separat analysieren und PCA/Clustering neu fitten
       -> nur Exploration, NICHT direkt mit Train-Features kombinieren
    2) gespeicherte Train-Modelle anwenden
       -> empfohlen für spätere Modellvorhersage

- Es wird gefragt, ob die erzeugten PCA-/Cluster-Features
  als CSV gespeichert werden sollen.

WICHTIG:
RUL wird niemals zum Clustering verwendet.
RUL dient nur zur Visualisierung und Diagnose.

Für ein finales RUL-Modell:
Scaler, PCA und Clustering auf TRAIN fitten und bei TEST
nur transform()/predict() verwenden.
"""

import os
import re
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


# ============================================================
# DATEIPFADE
# ============================================================

TRAIN_FILTERED_FILE = (
    "/Users/benjaminkoehl/Documents/Uni Stuttgart/Master/MLMech/"
    "Feature Engineering/Plot Telemetry Data/Plots/"
    "Outlier_Filter_Results/train_filtered_mad_rolling_median.csv"
)

TRAIN_FILE = (
    "/Users/benjaminkoehl/Documents/Uni Stuttgart/Master/"
    "MLMech/uav-remaining-useful-life/train.csv"
)

TEST_FILE = (
    "/Users/benjaminkoehl/Documents/Uni Stuttgart/Master/"
    "MLMech/uav-remaining-useful-life/test.csv"
)

OUTPUT_FOLDER = "Plots/Latent_Clustering_Analysis"

# Nur relevant, wenn Testdaten mit bereits trainierten
# PCA-/Cluster-Modellen verarbeitet werden sollen.
#
# Beispiel:
# TRAIN_MODEL_FOLDER = (
#     "Plots/Latent_Clustering_Analysis/"
#     "train_pca_kmeans_8features"
# )
TRAIN_MODEL_FOLDER = ""


# ============================================================
# STANDARD-EINSTELLUNGEN
# ============================================================

INTERACTIVE_SELECTION = True

DATASET_MODE = "train_filtered"
TEST_PROCESSING_MODE = "apply_train"

VISUALIZATION_METHOD = "pca"
COLOR_MODE = "rul_continuous"

CLUSTER_METHOD = "kmeans"
CLUSTER_NUMBER_MODE = "automatic"

N_CLUSTERS = 4
MAX_CLUSTERS_TO_TEST = 10

PCA_VARIANCE_TARGET = 0.95

TSNE_MAX_SAMPLES = 5000
TSNE_PERPLEXITY = 30.0
TSNE_ITERATIONS = 1000

RUL_BIN_SIZE = 20

RANDOM_STATE = 42
DPI = 300

SILHOUETTE_MAX_SAMPLES = 5000

SAVE_MODEL_FEATURE_CSV = True
SAVE_MODELS = True


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def choose_option(title, options):

    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

    keys = list(options.keys())

    for i, key in enumerate(keys, start=1):

        print(
            f"{i:2d} = {options[key]}"
        )

    while True:

        try:

            selection = int(
                input("\nAuswahl: ")
            )

            if 1 <= selection <= len(keys):

                return keys[
                    selection - 1
                ]

        except ValueError:
            pass

        print(
            "Ungültige Eingabe."
        )


def telemetry_sort_key(column):

    match = re.search(
        r"(\d+)$",
        column
    )

    if match:

        return int(
            match.group(1)
        )

    return column


def telemetry_number(column):

    match = re.search(
        r"(\d+)$",
        column
    )

    if match:

        return int(
            match.group(1)
        )

    return None


# ============================================================
# DATENSATZ AUSWÄHLEN
# ============================================================

if INTERACTIVE_SELECTION:

    DATASET_MODE = choose_option(
        "DATENSATZ AUSWÄHLEN",
        {
            "train_filtered":
                "Gefilterte Trainingsdaten",

            "train":
                "Originale train.csv",

            "test":
                "test.csv – darf ohne RUL sein",

            "custom":
                "Eigenen CSV-Pfad eingeben"
        }
    )


if DATASET_MODE == "train_filtered":

    CSV_FILE = TRAIN_FILTERED_FILE
    DATASET_NAME = "train_filtered"

elif DATASET_MODE == "train":

    CSV_FILE = TRAIN_FILE
    DATASET_NAME = "train"

elif DATASET_MODE == "test":

    CSV_FILE = TEST_FILE
    DATASET_NAME = "test"

elif DATASET_MODE == "custom":

    CSV_FILE = input(
        "\nVollständigen Pfad zur CSV-Datei eingeben: "
    ).strip()

    DATASET_NAME = os.path.splitext(
        os.path.basename(
            CSV_FILE
        )
    )[0]

else:

    raise ValueError(
        "Unbekannter DATASET_MODE."
    )


# ============================================================
# DATEN EINLESEN
# ============================================================

df_original = pd.read_csv(
    CSV_FILE
)


# ============================================================
# PFLICHTSPALTEN
# ============================================================

for required in [
    "uav_id",
    "flight_cycle"
]:

    if required not in df_original.columns:

        raise ValueError(
            f"Spalte '{required}' fehlt."
        )


HAS_RUL = (
    "RUL"
    in
    df_original.columns
)


print("\n" + "=" * 72)
print("DATENSATZ")
print("=" * 72)

print(
    f"Datei:              {CSV_FILE}"
)

print(
    f"Zeilen:             {len(df_original)}"
)

print(
    f"RUL vorhanden:      {HAS_RUL}"
)


if not HAS_RUL:

    print(
        "\nHINWEIS:"
        "\nDie Datei besitzt keine RUL-Spalte."
        "\nRUL-Farbcodierung, RUL-Bins und "
        "Cluster-RUL-Statistiken werden deaktiviert."
    )


# ============================================================
# TELEMETRIESPALTEN
# ============================================================

telemetry_columns = sorted(
    [
        column
        for column in df_original.columns
        if column.startswith(
            "telemetry_"
        )
    ],
    key=telemetry_sort_key
)


if len(
    telemetry_columns
) == 0:

    raise ValueError(
        "Keine telemetry_-Spalten gefunden."
    )


uav_ids = np.sort(
    df_original[
        "uav_id"
    ].unique()
)


print(
    f"UAVs:               {len(uav_ids)}"
)

print(
    f"Telemetrien:        {len(telemetry_columns)}"
)


# ============================================================
# TELEMETRIEN ANZEIGEN
# ============================================================

print("\n" + "=" * 72)
print("TELEMETRIEN")
print("=" * 72)


for telemetry in telemetry_columns:

    number = telemetry_number(
        telemetry
    )

    print(
        f"{number:2d} = {telemetry}"
    )


# ============================================================
# TELEMETRIEN AUSWÄHLEN
# ============================================================

def select_telemetries():

    """
    Auswahl erfolgt über die tatsächliche Telemetrienummer.

    Beispiele:
        1
        6,11,12
        1-8
        1,5,10-15
        all
    """

    number_to_column = {
        telemetry_number(column):
            column
        for column in telemetry_columns
        if telemetry_number(column) is not None
    }


    print("\nAuswahlmöglichkeiten:")
    print("  all       -> alle Telemetrien")
    print("  6         -> nur telemetry_06")
    print("  6,11,12   -> mehrere Telemetrien")
    print("  1-10      -> telemetry_01 bis telemetry_10")
    print("  1,5,10-15 -> Kombination")


    while True:

        text = input(
            "\nWelche Telemetrien sollen angepasst / "
            "analysiert werden? "
        ).strip().lower()


        if text == "all":

            return telemetry_columns.copy()


        selected_numbers = set()


        try:

            for part in text.split(","):

                part = part.strip()


                if "-" in part:

                    start, end = [
                        int(value)
                        for value in part.split("-")
                    ]


                    for number in range(
                        start,
                        end + 1
                    ):

                        selected_numbers.add(
                            number
                        )


                else:

                    selected_numbers.add(
                        int(part)
                    )


            missing = [
                number
                for number in selected_numbers
                if number not in number_to_column
            ]


            if missing:

                print(
                    "Nicht vorhandene Telemetrien:",
                    missing
                )

                continue


            if len(
                selected_numbers
            ) >= 1:

                return [
                    number_to_column[number]
                    for number in sorted(
                        selected_numbers
                    )
                ]


        except ValueError:
            pass


        print(
            "Ungültige Eingabe."
        )


selected_telemetries = (
    select_telemetries()
)


# ============================================================
# TEST-MODUS
# ============================================================

IS_TEST_STYLE_DATA = (
    DATASET_MODE == "test"
    or
    not HAS_RUL
)


if (
    INTERACTIVE_SELECTION
    and
    IS_TEST_STYLE_DATA
):

    TEST_PROCESSING_MODE = choose_option(
        "WIE SOLLEN TESTDATEN VERARBEITET WERDEN?",
        {
            "apply_train":
                "Gespeicherte Train-Modelle anwenden – empfohlen",

            "fit_test":
                "PCA/Clustering auf Test neu fitten – nur Exploration"
        }
    )


# ============================================================
# VISUALISIERUNG AUSWÄHLEN
# ============================================================

if INTERACTIVE_SELECTION:

    VISUALIZATION_METHOD = choose_option(
        "LATENT-SPACE VISUALISIERUNG",
        {
            "pca":
                "PCA – stabil und als Modellfeature geeignet",

            "tsne":
                "t-SNE – nur zur Visualisierung"
        }
    )


# ============================================================
# FARBGEBUNG
# ============================================================

if INTERACTIVE_SELECTION:

    if HAS_RUL:

        COLOR_MODE = choose_option(
            "FARBGEBUNG IM LATENT SPACE",
            {
                "rul_continuous":
                    "RUL kontinuierlich",

                "rul_bins":
                    "RUL in Bereichen / Klassen",

                "flight_cycle":
                    "flight_cycle kontinuierlich"
            }
        )

    else:

        COLOR_MODE = choose_option(
            "FARBGEBUNG IM LATENT SPACE",
            {
                "flight_cycle":
                    "flight_cycle kontinuierlich",

                "none":
                    "Keine kontinuierliche Farbcodierung"
            }
        )


if (
    COLOR_MODE == "rul_bins"
    and
    HAS_RUL
):

    while True:

        try:

            RUL_BIN_SIZE = int(
                input(
                    "\nRUL-Bin-Größe, z.B. 20: "
                )
            )

            if RUL_BIN_SIZE > 0:

                break

        except ValueError:
            pass


# ============================================================
# CLUSTER-EINSTELLUNGEN
# ============================================================

APPLY_TRAIN_MODELS = (
    IS_TEST_STYLE_DATA
    and
    TEST_PROCESSING_MODE
    ==
    "apply_train"
)


if APPLY_TRAIN_MODELS:

    print("\n" + "=" * 72)
    print("GESPEICHERTE TRAIN-MODELLE")
    print("=" * 72)


    if INTERACTIVE_SELECTION:

        entered_folder = input(
            "\nOrdner mit scaler.joblib, pca_model.joblib, "
            "Cluster-Modell und Selected_Telemetries.csv"
            "\n[Enter = TRAIN_MODEL_FOLDER aus dem Skript]: "
        ).strip()


        if entered_folder:

            TRAIN_MODEL_FOLDER = (
                entered_folder
            )


    if not TRAIN_MODEL_FOLDER:

        raise ValueError(
            "TRAIN_MODEL_FOLDER ist leer. "
            "Bitte den Trainings-Modellordner angeben."
        )


    if not os.path.isdir(
        TRAIN_MODEL_FOLDER
    ):

        raise FileNotFoundError(
            f"Train-Modellordner nicht gefunden:\n"
            f"{TRAIN_MODEL_FOLDER}"
        )


    # --------------------------------------------------------
    # Gespeicherte Telemetrien prüfen
    # --------------------------------------------------------

    selected_file = os.path.join(
        TRAIN_MODEL_FOLDER,
        "Selected_Telemetries.csv"
    )


    if not os.path.exists(
        selected_file
    ):

        raise FileNotFoundError(
            "Selected_Telemetries.csv fehlt im "
            "Train-Modellordner."
        )


    saved_telemetries = (
        pd.read_csv(
            selected_file
        )[
            "selected_telemetry"
        ]
        .dropna()
        .astype(str)
        .tolist()
    )


    if (
        selected_telemetries
        !=
        saved_telemetries
    ):

        print(
            "\nAUSGEWÄHLTE TELEMETRIEN:"
        )

        print(
            selected_telemetries
        )

        print(
            "\nTELEMETRIEN DES TRAIN-MODELLS:"
        )

        print(
            saved_telemetries
        )


        raise ValueError(
            "Die Test-Telemetrien müssen exakt "
            "den beim Training verwendeten "
            "Telemetrien und deren Reihenfolge entsprechen."
        )


    # --------------------------------------------------------
    # Cluster-Modell automatisch erkennen
    # --------------------------------------------------------

    kmeans_path = os.path.join(
        TRAIN_MODEL_FOLDER,
        "kmeans_model.joblib"
    )


    gmm_path = os.path.join(
        TRAIN_MODEL_FOLDER,
        "gmm_model.joblib"
    )


    if os.path.exists(
        kmeans_path
    ):

        CLUSTER_METHOD = "kmeans"
        cluster_model_path = (
            kmeans_path
        )

    elif os.path.exists(
        gmm_path
    ):

        CLUSTER_METHOD = "gmm"
        cluster_model_path = (
            gmm_path
        )

    else:

        raise FileNotFoundError(
            "Kein kmeans_model.joblib oder "
            "gmm_model.joblib gefunden."
        )


else:

    if INTERACTIVE_SELECTION:

        CLUSTER_METHOD = choose_option(
            "CLUSTER-METHODE",
            {
                "kmeans":
                    "KMeans",

                "gmm":
                    "Gaussian Mixture Model (GMM)"
            }
        )


        CLUSTER_NUMBER_MODE = choose_option(
            "CLUSTERZAHL",
            {
                "automatic":
                    "Automatisch bestimmen",

                "manual":
                    "Clusterzahl manuell vorgeben"
            }
        )


        if (
            CLUSTER_NUMBER_MODE
            ==
            "manual"
        ):

            while True:

                try:

                    N_CLUSTERS = int(
                        input(
                            "\nAnzahl Cluster: "
                        )
                    )

                    if N_CLUSTERS >= 2:

                        break

                except ValueError:
                    pass


        else:

            while True:

                try:

                    MAX_CLUSTERS_TO_TEST = int(
                        input(
                            "\nMaximale Clusterzahl "
                            "zum Testen, z.B. 10: "
                        )
                    )

                    if (
                        MAX_CLUSTERS_TO_TEST
                        >=
                        2
                    ):

                        break

                except ValueError:
                    pass


# ============================================================
# CSV SPEICHERN?
# ============================================================
#
# Dieses Programm berechnet keine lokalen Regressionen.
# Deshalb bezieht sich die CSV-Abfrage hier auf die neu
# erzeugten PCA-/Cluster-Features.
# ============================================================

if INTERACTIVE_SELECTION:

    save_option = choose_option(
        "NEUE PCA-/CLUSTER-FEATURES ALS CSV SPEICHERN?",
        {
            "yes":
                "Ja – Originaldaten + neue Features speichern",

            "no":
                "Nein – nur Plots / Analyse"
        }
    )


    SAVE_MODEL_FEATURE_CSV = (
        save_option
        ==
        "yes"
    )


# ============================================================
# EINSTELLUNGEN
# ============================================================

print("\n" + "=" * 72)
print("EINSTELLUNGEN")
print("=" * 72)

print(
    f"Datensatz:           {DATASET_NAME}"
)

print(
    f"RUL vorhanden:       {HAS_RUL}"
)

print(
    f"Telemetrien:         "
    f"{', '.join(selected_telemetries)}"
)

print(
    f"Visualisierung:      {VISUALIZATION_METHOD}"
)

print(
    f"Farbmodus:           {COLOR_MODE}"
)

print(
    f"Clustering:          {CLUSTER_METHOD}"
)

print(
    f"Train-Modelle laden: {APPLY_TRAIN_MODELS}"
)

print(
    f"CSV speichern:       {SAVE_MODEL_FEATURE_CSV}"
)


# ============================================================
# AUSGABEORDNER
# ============================================================

processing_label = (
    "apply_train_models"
    if APPLY_TRAIN_MODELS
    else
    "fit"
)


analysis_folder = os.path.join(
    OUTPUT_FOLDER,
    (
        f"{DATASET_NAME}_"
        f"{processing_label}_"
        f"{VISUALIZATION_METHOD}_"
        f"{CLUSTER_METHOD}_"
        f"{len(selected_telemetries)}features"
    )
)


os.makedirs(
    analysis_folder,
    exist_ok=True
)


# ============================================================
# DATEN VORBEREITEN
# ============================================================

prepared_df = (
    df_original.copy()
)


prepared_df = (
    prepared_df.sort_values(
        [
            "uav_id",
            "flight_cycle"
        ]
    )
)


for telemetry in selected_telemetries:

    prepared_df[
        telemetry
    ] = (
        prepared_df
        .groupby(
            "uav_id"
        )[
            telemetry
        ]
        .transform(
            lambda series:
            series.interpolate(
                method="linear",
                limit_direction="both"
            )
        )
    )


prepared_df = (
    prepared_df.sort_index()
)


valid_mask = (
    prepared_df[
        selected_telemetries
    ]
    .notna()
    .all(
        axis=1
    )
)


valid_indices = (
    prepared_df.index[
        valid_mask
    ]
)


metadata_columns = [
    "uav_id",
    "flight_cycle"
]


if HAS_RUL:

    metadata_columns.append(
        "RUL"
    )


analysis_df = (
    prepared_df.loc[
        valid_indices,
        metadata_columns
        +
        selected_telemetries
    ]
    .copy()
)


X = (
    analysis_df[
        selected_telemetries
    ]
    .to_numpy(
        dtype=float
    )
)


print(
    "\nGültige Datenpunkte:",
    len(
        analysis_df
    )
)


# ============================================================
# STANDARDISIERUNG + PCA
# ============================================================

if APPLY_TRAIN_MODELS:

    scaler = joblib.load(
        os.path.join(
            TRAIN_MODEL_FOLDER,
            "scaler.joblib"
        )
    )


    pca_model = joblib.load(
        os.path.join(
            TRAIN_MODEL_FOLDER,
            "pca_model.joblib"
        )
    )


    cluster_model = joblib.load(
        cluster_model_path
    )


    X_scaled = scaler.transform(
        X
    )


    X_pca_model = (
        pca_model.transform(
            X_scaled
        )
    )


    explained_variance = (
        pca_model.explained_variance_ratio_
    )


    cumulative_variance = (
        np.cumsum(
            explained_variance
        )
    )


    n_model_pcs = (
        X_pca_model.shape[1]
    )


else:

    scaler = StandardScaler()


    X_scaled = (
        scaler.fit_transform(
            X
        )
    )


    pca_full = PCA()


    X_pca_full = (
        pca_full.fit_transform(
            X_scaled
        )
    )


    explained_variance = (
        pca_full.explained_variance_ratio_
    )


    cumulative_variance = (
        np.cumsum(
            explained_variance
        )
    )


    n_model_pcs = int(
        np.searchsorted(
            cumulative_variance,
            PCA_VARIANCE_TARGET
        )
        +
        1
    )


    # Bei nur einer ausgewählten Telemetrie ist nur PC1 möglich.
    n_model_pcs = max(
        1,
        min(
            n_model_pcs,
            X_scaled.shape[1]
        )
    )


    pca_model = PCA(
        n_components=n_model_pcs
    )


    X_pca_model = (
        pca_model.fit_transform(
            X_scaled
        )
    )


# ============================================================
# PCA INFORMATION
# ============================================================

print("\n" + "=" * 72)
print("PCA")
print("=" * 72)


for i, variance in enumerate(
    explained_variance,
    start=1
):

    print(
        f"PC{i:02d}: "
        f"{variance * 100:6.2f} % | "
        f"kumulativ: "
        f"{cumulative_variance[i - 1] * 100:6.2f} %"
    )


print(
    f"\nVerwendete PCA-Komponenten: "
    f"{n_model_pcs}"
)


# ============================================================
# PCA EXPLAINED VARIANCE
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        10,
        6
    )
)


components = np.arange(
    1,
    len(
        explained_variance
    )
    +
    1
)


ax.bar(
    components,
    explained_variance * 100,
    alpha=0.7,
    label="Einzeln"
)


ax.plot(
    components,
    cumulative_variance * 100,
    marker="o",
    label="Kumulativ"
)


ax.axhline(
    PCA_VARIANCE_TARGET * 100,
    linestyle="--",
    linewidth=1.0,
    label=(
        f"{PCA_VARIANCE_TARGET * 100:.0f} % Ziel"
    )
)


ax.set_xlabel(
    "Principal Component"
)


ax.set_ylabel(
    "Explained Variance [%]"
)


ax.set_title(
    "PCA Explained Variance"
)


ax.grid(
    True,
    alpha=0.25
)


ax.legend()


plt.tight_layout()


plt.savefig(
    os.path.join(
        analysis_folder,
        "PCA_Explained_Variance.png"
    ),
    dpi=DPI,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# PCA LOADINGS
# ============================================================

loading_columns = [
    f"PC{i}"
    for i in range(
        1,
        n_model_pcs + 1
    )
]


loadings_df = pd.DataFrame(
    pca_model.components_.T,
    index=selected_telemetries,
    columns=loading_columns
)


loadings_df.to_csv(
    os.path.join(
        analysis_folder,
        "PCA_Loadings.csv"
    )
)


max_loading = np.max(
    np.abs(
        loadings_df.to_numpy()
    )
)


fig, ax = plt.subplots(
    figsize=(
        max(
            8,
            n_model_pcs * 1.4
        ),
        max(
            4,
            len(
                selected_telemetries
            )
            *
            0.55
        )
    )
)


image = ax.imshow(
    loadings_df.to_numpy(),
    cmap="coolwarm",
    vmin=-max_loading,
    vmax=max_loading,
    aspect="auto"
)


ax.set_xticks(
    np.arange(
        len(
            loading_columns
        )
    )
)


ax.set_xticklabels(
    loading_columns
)


ax.set_yticks(
    np.arange(
        len(
            selected_telemetries
        )
    )
)


ax.set_yticklabels(
    selected_telemetries
)


fig.colorbar(
    image,
    ax=ax
).set_label(
    "PCA Loading"
)


ax.set_title(
    "PCA Loadings"
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        analysis_folder,
        "PCA_Loadings.png"
    ),
    dpi=DPI,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# AUTOMATISCHE CLUSTERZAHL
# ============================================================

def choose_number_of_clusters(
    cluster_data
):

    max_k = min(
        MAX_CLUSTERS_TO_TEST,
        len(
            cluster_data
        )
        -
        1
    )


    if max_k < 2:

        raise ValueError(
            "Zu wenige Punkte für Clustering."
        )


    k_values = list(
        range(
            2,
            max_k + 1
        )
    )


    scores = []


    if CLUSTER_METHOD == "kmeans":

        print(
            "\nAutomatische Auswahl via "
            "Silhouette Score"
        )


        if (
            len(
                cluster_data
            )
            >
            SILHOUETTE_MAX_SAMPLES
        ):

            rng = np.random.RandomState(
                RANDOM_STATE
            )


            eval_idx = rng.choice(
                len(
                    cluster_data
                ),
                size=SILHOUETTE_MAX_SAMPLES,
                replace=False
            )

        else:

            eval_idx = np.arange(
                len(
                    cluster_data
                )
            )


        for k in k_values:

            model = KMeans(
                n_clusters=k,
                random_state=RANDOM_STATE,
                n_init=10
            )


            labels = (
                model.fit_predict(
                    cluster_data
                )
            )


            score = silhouette_score(
                cluster_data[
                    eval_idx
                ],
                labels[
                    eval_idx
                ]
            )


            scores.append(
                score
            )


            print(
                f"k={k:2d} | "
                f"Silhouette={score:.4f}"
            )


        best_k = k_values[
            int(
                np.argmax(
                    scores
                )
            )
        ]


        fig, ax = plt.subplots(
            figsize=(
                9,
                5
            )
        )


        ax.plot(
            k_values,
            scores,
            marker="o"
        )


        ax.axvline(
            best_k,
            linestyle="--",
            linewidth=1.0
        )


        ax.set_xlabel(
            "Anzahl Cluster k"
        )


        ax.set_ylabel(
            "Silhouette Score"
        )


        ax.set_title(
            f"Clusterwahl – bestes k = {best_k}"
        )


        ax.grid(
            True,
            alpha=0.25
        )


        plt.tight_layout()


        plt.savefig(
            os.path.join(
                analysis_folder,
                "Cluster_Number_Silhouette.png"
            ),
            dpi=DPI,
            bbox_inches="tight"
        )


        plt.close()


    elif CLUSTER_METHOD == "gmm":

        print(
            "\nAutomatische Auswahl via BIC"
        )


        for k in k_values:

            model = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=RANDOM_STATE
            )


            model.fit(
                cluster_data
            )


            bic = model.bic(
                cluster_data
            )


            scores.append(
                bic
            )


            print(
                f"k={k:2d} | "
                f"BIC={bic:.2f}"
            )


        best_k = k_values[
            int(
                np.argmin(
                    scores
                )
            )
        ]


        fig, ax = plt.subplots(
            figsize=(
                9,
                5
            )
        )


        ax.plot(
            k_values,
            scores,
            marker="o"
        )


        ax.axvline(
            best_k,
            linestyle="--",
            linewidth=1.0
        )


        ax.set_xlabel(
            "Anzahl Cluster k"
        )


        ax.set_ylabel(
            "BIC – kleiner ist besser"
        )


        ax.set_title(
            f"Clusterwahl – bestes k = {best_k}"
        )


        ax.grid(
            True,
            alpha=0.25
        )


        plt.tight_layout()


        plt.savefig(
            os.path.join(
                analysis_folder,
                "Cluster_Number_BIC.png"
            ),
            dpi=DPI,
            bbox_inches="tight"
        )


        plt.close()


    else:

        raise ValueError(
            "Unbekannte Cluster-Methode."
        )


    return best_k


# ============================================================
# CLUSTERING / CLUSTER-ZUWEISUNG
# ============================================================

cluster_space = (
    X_pca_model
)


if APPLY_TRAIN_MODELS:

    if CLUSTER_METHOD == "kmeans":

        cluster_labels = (
            cluster_model.predict(
                cluster_space
            )
        )


        cluster_distances = (
            cluster_model.transform(
                cluster_space
            )
        )


        N_CLUSTERS = (
            cluster_model.n_clusters
        )


        cluster_probabilities = None


    elif CLUSTER_METHOD == "gmm":

        cluster_labels = (
            cluster_model.predict(
                cluster_space
            )
        )


        cluster_probabilities = (
            cluster_model.predict_proba(
                cluster_space
            )
        )


        N_CLUSTERS = (
            cluster_model.n_components
        )


        means = (
            cluster_model.means_
        )


        cluster_distances = np.sqrt(
            np.sum(
                (
                    cluster_space[
                        :,
                        None,
                        :
                    ]
                    -
                    means[
                        None,
                        :,
                        :
                    ]
                )
                ** 2,
                axis=2
            )
        )


else:

    if (
        CLUSTER_NUMBER_MODE
        ==
        "automatic"
    ):

        N_CLUSTERS = (
            choose_number_of_clusters(
                cluster_space
            )
        )


    print(
        "\nVerwendete Clusterzahl:",
        N_CLUSTERS
    )


    if CLUSTER_METHOD == "kmeans":

        cluster_model = KMeans(
            n_clusters=N_CLUSTERS,
            random_state=RANDOM_STATE,
            n_init=10
        )


        cluster_labels = (
            cluster_model.fit_predict(
                cluster_space
            )
        )


        cluster_distances = (
            cluster_model.transform(
                cluster_space
            )
        )


        cluster_probabilities = None


    elif CLUSTER_METHOD == "gmm":

        cluster_model = GaussianMixture(
            n_components=N_CLUSTERS,
            covariance_type="full",
            random_state=RANDOM_STATE
        )


        cluster_model.fit(
            cluster_space
        )


        cluster_labels = (
            cluster_model.predict(
                cluster_space
            )
        )


        cluster_probabilities = (
            cluster_model.predict_proba(
                cluster_space
            )
        )


        means = (
            cluster_model.means_
        )


        cluster_distances = np.sqrt(
            np.sum(
                (
                    cluster_space[
                        :,
                        None,
                        :
                    ]
                    -
                    means[
                        None,
                        :,
                        :
                    ]
                )
                ** 2,
                axis=2
            )
        )


    else:

        raise ValueError(
            "Unbekannte Cluster-Methode."
        )


assigned_distance = (
    cluster_distances[
        np.arange(
            len(
                cluster_labels
            )
        ),
        cluster_labels
    ]
)


# ============================================================
# VISUALISIERUNGSKOORDINATEN
# ============================================================

if VISUALIZATION_METHOD == "pca":

    vis_positions = np.arange(
        len(
            analysis_df
        )
    )


    # Bei nur einer PCA-Komponente:
    # PC1 gegen 0 darstellen.
    if (
        X_pca_model.shape[1]
        >=
        2
    ):

        vis_coordinates = (
            X_pca_model[
                :,
                :2
            ]
        )

    else:

        vis_coordinates = np.column_stack(
            [
                X_pca_model[
                    :,
                    0
                ],
                np.zeros(
                    len(
                        X_pca_model
                    )
                )
            ]
        )


    vis_name = "PCA"


elif VISUALIZATION_METHOD == "tsne":

    vis_name = "t-SNE"


    if (
        len(
            X_scaled
        )
        >
        TSNE_MAX_SAMPLES
    ):

        rng = np.random.RandomState(
            RANDOM_STATE
        )


        vis_positions = rng.choice(
            len(
                X_scaled
            ),
            size=TSNE_MAX_SAMPLES,
            replace=False
        )

    else:

        vis_positions = np.arange(
            len(
                X_scaled
            )
        )


    X_tsne_input = (
        X_scaled[
            vis_positions
        ]
    )


    if len(
        X_tsne_input
    ) < 3:

        raise ValueError(
            "Zu wenige Punkte für t-SNE."
        )


    perplexity = min(
        TSNE_PERPLEXITY,
        float(
            len(
                X_tsne_input
            )
            -
            1
        )
    )


    tsne_parameters = {
        "n_components":
            2,

        "perplexity":
            perplexity,

        "learning_rate":
            200.0,

        "init":
            "pca",

        "random_state":
            RANDOM_STATE
    }


    tsne_signature = inspect.signature(
        TSNE
    )


    if (
        "max_iter"
        in
        tsne_signature.parameters
    ):

        tsne_parameters[
            "max_iter"
        ] = TSNE_ITERATIONS

    else:

        tsne_parameters[
            "n_iter"
        ] = TSNE_ITERATIONS


    tsne = TSNE(
        **tsne_parameters
    )


    vis_coordinates = (
        tsne.fit_transform(
            X_tsne_input
        )
    )


else:

    raise ValueError(
        "Unbekannte Visualisierung."
    )


vis_df = (
    analysis_df.iloc[
        vis_positions
    ]
    .copy()
)


vis_cluster_labels = (
    cluster_labels[
        vis_positions
    ]
)


# ============================================================
# PLOT: RUL / FLIGHT CYCLE / NONE
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        10,
        7
    )
)


if (
    COLOR_MODE
    ==
    "rul_continuous"
    and
    HAS_RUL
):

    scatter = ax.scatter(
        vis_coordinates[
            :,
            0
        ],
        vis_coordinates[
            :,
            1
        ],
        c=vis_df[
            "RUL"
        ],
        cmap="viridis",
        s=10,
        alpha=0.6
    )


    fig.colorbar(
        scatter,
        ax=ax
    ).set_label(
        "RUL"
    )


elif (
    COLOR_MODE
    ==
    "flight_cycle"
):

    scatter = ax.scatter(
        vis_coordinates[
            :,
            0
        ],
        vis_coordinates[
            :,
            1
        ],
        c=vis_df[
            "flight_cycle"
        ],
        cmap="viridis",
        s=10,
        alpha=0.6
    )


    fig.colorbar(
        scatter,
        ax=ax
    ).set_label(
        "flight_cycle"
    )


elif (
    COLOR_MODE
    ==
    "rul_bins"
    and
    HAS_RUL
):

    rul_values = (
        vis_df[
            "RUL"
        ]
        .to_numpy()
    )


    max_rul = (
        np.nanmax(
            rul_values
        )
    )


    bins = np.arange(
        0,
        max_rul
        +
        RUL_BIN_SIZE
        +
        1,
        RUL_BIN_SIZE
    )


    rul_bin_ids = (
        np.digitize(
            rul_values,
            bins,
            right=False
        )
        -
        1
    )


    n_bins = max(
        len(
            bins
        )
        -
        1,
        2
    )


    cmap = plt.get_cmap(
        "viridis",
        n_bins
    )


    ax.scatter(
        vis_coordinates[
            :,
            0
        ],
        vis_coordinates[
            :,
            1
        ],
        c=rul_bin_ids,
        cmap=cmap,
        s=10,
        alpha=0.6
    )


else:

    ax.scatter(
        vis_coordinates[
            :,
            0
        ],
        vis_coordinates[
            :,
            1
        ],
        s=10,
        alpha=0.6
    )


ax.set_xlabel(
    f"{vis_name} 1"
)


ax.set_ylabel(
    f"{vis_name} 2"
)


ax.set_title(
    f"{vis_name} Latent Space – "
    f"{COLOR_MODE}"
)


ax.grid(
    True,
    alpha=0.2
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        analysis_folder,
        f"{vis_name}_Colored_{COLOR_MODE}.png"
    ),
    dpi=DPI,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# PLOT: CLUSTER
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        10,
        7
    )
)


cluster_cmap = plt.get_cmap(
    "tab10",
    max(
        N_CLUSTERS,
        2
    )
)


for cluster_id in range(
    N_CLUSTERS
):

    mask = (
        vis_cluster_labels
        ==
        cluster_id
    )


    ax.scatter(
        vis_coordinates[
            mask,
            0
        ],
        vis_coordinates[
            mask,
            1
        ],
        s=11,
        alpha=0.6,
        color=cluster_cmap(
            cluster_id
        ),
        label=(
            f"Cluster {cluster_id}"
        )
    )


ax.set_xlabel(
    f"{vis_name} 1"
)


ax.set_ylabel(
    f"{vis_name} 2"
)


ax.set_title(
    f"{vis_name} Latent Space – "
    f"{CLUSTER_METHOD} Cluster"
)


ax.grid(
    True,
    alpha=0.2
)


ax.legend(
    fontsize=8
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        analysis_folder,
        f"{vis_name}_Clusters.png"
    ),
    dpi=DPI,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# RUL-DIAGNOSE NUR WENN RUL EXISTIERT
# ============================================================

if HAS_RUL:

    cluster_analysis_df = (
        analysis_df[
            [
                "uav_id",
                "flight_cycle",
                "RUL"
            ]
        ]
        .copy()
    )


    cluster_analysis_df[
        "cluster_id"
    ] = cluster_labels


    cluster_rul_summary = (
        cluster_analysis_df
        .groupby(
            "cluster_id"
        )
        .agg(
            n_points=(
                "RUL",
                "size"
            ),
            RUL_mean=(
                "RUL",
                "mean"
            ),
            RUL_median=(
                "RUL",
                "median"
            ),
            RUL_std=(
                "RUL",
                "std"
            ),
            RUL_min=(
                "RUL",
                "min"
            ),
            RUL_max=(
                "RUL",
                "max"
            ),
            flight_cycle_mean=(
                "flight_cycle",
                "mean"
            )
        )
        .reset_index()
    )


    if SAVE_MODEL_FEATURE_CSV:

        cluster_rul_summary.to_csv(
            os.path.join(
                analysis_folder,
                "Cluster_RUL_Summary.csv"
            ),
            index=False
        )


    print(
        "\nCLUSTER-RUL-ZUSAMMENFASSUNG"
    )


    print(
        cluster_rul_summary.to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # RUL Boxplot
    # --------------------------------------------------------

    box_values = []

    box_labels = []


    for cluster_id in range(
        N_CLUSTERS
    ):

        values = (
            cluster_analysis_df.loc[
                cluster_analysis_df[
                    "cluster_id"
                ]
                ==
                cluster_id,
                "RUL"
            ]
            .dropna()
            .to_numpy()
        )


        if len(
            values
        ) > 0:

            box_values.append(
                values
            )

            box_labels.append(
                f"C{cluster_id}"
            )


    if box_values:

        fig, ax = plt.subplots(
            figsize=(
                max(
                    8,
                    N_CLUSTERS
                    *
                    1.2
                ),
                6
            )
        )


        ax.boxplot(
            box_values,
            labels=box_labels
        )


        ax.set_xlabel(
            "Cluster"
        )


        ax.set_ylabel(
            "RUL"
        )


        ax.set_title(
            "RUL-Verteilung innerhalb der Cluster"
        )


        ax.grid(
            True,
            axis="y",
            alpha=0.25
        )


        plt.tight_layout()


        plt.savefig(
            os.path.join(
                analysis_folder,
                "Cluster_RUL_Boxplot.png"
            ),
            dpi=DPI,
            bbox_inches="tight"
        )


        plt.close()


else:

    print(
        "\nKeine RUL-Spalte vorhanden:"
        "\nCluster-RUL-Summary und RUL-Boxplot "
        "werden übersprungen."
    )


# ============================================================
# NEUE MODELLFEATURES
# ============================================================

feature_df = pd.DataFrame(
    index=valid_indices
)


# PCA-Komponenten
for component_index in range(
    n_model_pcs
):

    feature_df[
        f"latent_PC{component_index + 1}"
    ] = X_pca_model[
        :,
        component_index
    ]


# Cluster-ID
feature_df[
    "latent_cluster_id"
] = cluster_labels


# One-Hot Cluster
for cluster_id in range(
    N_CLUSTERS
):

    feature_df[
        f"latent_cluster_{cluster_id}"
    ] = (
        cluster_labels
        ==
        cluster_id
    ).astype(
        int
    )


# Distanzen zu Clusterzentren
for cluster_id in range(
    N_CLUSTERS
):

    feature_df[
        f"latent_distance_cluster_{cluster_id}"
    ] = cluster_distances[
        :,
        cluster_id
    ]


# Distanz zum zugewiesenen Cluster
feature_df[
    "latent_distance_assigned"
] = assigned_distance


# GMM Wahrscheinlichkeiten
if (
    CLUSTER_METHOD == "gmm"
    and
    cluster_probabilities is not None
):

    for cluster_id in range(
        N_CLUSTERS
    ):

        feature_df[
            f"latent_cluster_probability_{cluster_id}"
        ] = cluster_probabilities[
            :,
            cluster_id
        ]


# ============================================================
# CSV EXPORT
# ============================================================

if SAVE_MODEL_FEATURE_CSV:

    output_df = (
        df_original.copy()
    )


    for column in feature_df.columns:

        output_df[
            column
        ] = np.nan


        output_df.loc[
            feature_df.index,
            column
        ] = feature_df[
            column
        ]


    output_csv = os.path.join(
        analysis_folder,
        (
            f"{DATASET_NAME}_with_"
            f"Latent_Cluster_Features_"
            f"{CLUSTER_METHOD}_"
            f"k{N_CLUSTERS}.csv"
        )
    )


    output_df.to_csv(
        output_csv,
        index=False
    )


    print(
        "\nNeue Feature-CSV:"
    )

    print(
        output_csv
    )


    # --------------------------------------------------------
    # Nur neue Features separat
    # --------------------------------------------------------

    export_metadata = [
        "uav_id",
        "flight_cycle"
    ]


    if HAS_RUL:

        export_metadata.append(
            "RUL"
        )


    feature_export = (
        df_original.loc[
            valid_indices,
            export_metadata
        ]
        .copy()
    )


    feature_export = pd.concat(
        [
            feature_export,
            feature_df
        ],
        axis=1
    )


    feature_export.to_csv(
        os.path.join(
            analysis_folder,
            (
                f"{DATASET_NAME}_"
                f"Latent_Cluster_Features_Only.csv"
            )
        ),
        index=False
    )


# ============================================================
# MODELLE SPEICHERN
# ============================================================
#
# Nur sinnvoll, wenn in diesem Lauf FIT durchgeführt wurde.
# Bei apply_train_models werden die bestehenden Modelle nicht
# erneut gespeichert.
# ============================================================

if (
    SAVE_MODELS
    and
    not APPLY_TRAIN_MODELS
):

    joblib.dump(
        scaler,
        os.path.join(
            analysis_folder,
            "scaler.joblib"
        )
    )


    joblib.dump(
        pca_model,
        os.path.join(
            analysis_folder,
            "pca_model.joblib"
        )
    )


    joblib.dump(
        cluster_model,
        os.path.join(
            analysis_folder,
            f"{CLUSTER_METHOD}_model.joblib"
        )
    )


    pd.DataFrame(
        {
            "selected_telemetry":
                selected_telemetries
        }
    ).to_csv(
        os.path.join(
            analysis_folder,
            "Selected_Telemetries.csv"
        ),
        index=False
    )


# ============================================================
# FERTIG
# ============================================================

print("\n" + "=" * 72)
print("FERTIG")
print("=" * 72)

print(
    f"Clusterzahl: {N_CLUSTERS}"
)

print(
    f"PCA-Komponenten als Features: "
    f"{n_model_pcs}"
)


print(
    "\nNeue Modellfeatures:"
)


for column in feature_df.columns:

    print(
        f" - {column}"
    )


print(
    "\nErgebnisse:"
)

print(
    os.path.abspath(
        analysis_folder
    )
)


if not HAS_RUL:

    print(
        "\nTEST OHNE RUL WURDE ERFOLGREICH VERARBEITET."
    )


if APPLY_TRAIN_MODELS:

    print(
        "\nDie Testdaten wurden mit den bereits "
        "auf Train gefitteten Modellen transformiert."
    )

else:

    print(
        "\nHINWEIS:"
        "\nFür eine faire finale RUL-Auswertung "
        "Scaler/PCA/Clustering nur auf Train fitten "
        "und auf Test anschließend nur anwenden."
    )
