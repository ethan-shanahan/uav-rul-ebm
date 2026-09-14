#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kombiniertes UAV Feature Engineering:
Transformation -> lokale Regression

Ablauf
------------------------------------------------------------
1. train / gefilterte train / test / eigene CSV auswählen
2. Telemetrien über Nummer 1-28 auswählen
3. X-Achse auswählen: RUL oder flight_cycle
4. Transformation auswählen
5. Transformation pro UAV chronologisch berechnen
6. Lokale Regression auf den TRANSFORMIERTEN Daten pro UAV
7. Regression und/oder Gradient plotten
8. Subplots oder Einzelplots
9. Optional transformierte CSV speichern
10. Optional CSV mit lokalen Regressionsfeatures speichern
11. test.csv ohne RUL wird automatisch über flight_cycle verarbeitet

Wichtig:
- Transformationen wie difference/rolling_* werden immer nach
  flight_cycle innerhalb jeder UAV berechnet.
- Die Regression kann anschließend gegen RUL oder flight_cycle
  durchgeführt werden.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

OUTPUT_FOLDER = "Plots/Transformation_Local_Regression"


# ============================================================
# STANDARDWERTE
# ============================================================

INTERACTIVE_SELECTION = True
DATASET_MODE = "train_filtered"  # train_filtered / train / test / custom

X_AXIS = "RUL"
TRANSFORM_MODE = "raw"
ROLLING_WINDOW = 5

REGRESSION_TYPE = "linear"
REGRESSION_WINDOW_SIZE = 10

SHOW_MEASUREMENT_POINTS = True
PLOT_LAYOUT = "subplots"
RESULT_MODE = "both"

UAVS_PER_FIGURE = 10
NCOLS = 4
DPI = 300
SHOW_LEGEND = False

USE_INCOMPLETE_LAST_WINDOW = True
SAVE_TRANSFORMED_CSV = True
SAVE_REGRESSION_FEATURES = True

# "center" = Gradient in Fenstermitte
# "end"    = Gradient am Fensterende (für ML oft sinnvoller)
REFERENCE_MODE = "center"


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def choose_option(title, options):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    keys = list(options.keys())
    for i, key in enumerate(keys, start=1):
        print(f"{i:2d} = {options[key]}")

    while True:
        try:
            selection = int(input("\nAuswahl: "))
            if 1 <= selection <= len(keys):
                return keys[selection - 1]
        except ValueError:
            pass
        print("Ungültige Eingabe.")


def telemetry_sort_key(column):
    match = re.search(r"(\d+)$", column)
    return int(match.group(1)) if match else column


def telemetry_number(column):
    match = re.search(r"(\d+)$", column)
    return int(match.group(1)) if match else None


def select_telemetries(telemetry_columns):
    """Auswahl über die echte Telemetrienummer, z.B. 1-28."""

    number_to_column = {
        telemetry_number(column): column
        for column in telemetry_columns
        if telemetry_number(column) is not None
    }

    print("\nAuswahlmöglichkeiten:")
    print("  all       -> alle Telemetrien")
    print("  3         -> nur telemetry_03")
    print("  1,5,8     -> mehrere Telemetrien")
    print("  1-10      -> telemetry_01 bis telemetry_10")
    print("  1,5,10-15 -> Kombination")

    while True:
        user_input = input(
            "\nWelche Telemetrien sollen transformiert "
            "und regressiert werden? "
        ).strip().lower()

        if user_input == "all":
            return telemetry_columns.copy()

        selected_numbers = set()

        try:
            for part in user_input.split(","):
                part = part.strip()

                if "-" in part:
                    start, end = [int(v) for v in part.split("-")]
                    if start > end:
                        start, end = end, start
                    for number in range(start, end + 1):
                        selected_numbers.add(number)
                else:
                    selected_numbers.add(int(part))

            missing = [
                n for n in selected_numbers
                if n not in number_to_column
            ]

            if missing:
                print(f"Nicht vorhandene Telemetrien: {missing}")
                continue

            if selected_numbers:
                return [
                    number_to_column[n]
                    for n in sorted(selected_numbers)
                ]

        except ValueError:
            pass

        print("Ungültige Eingabe.")


def calculate_r_squared(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot


# ============================================================
# DATENSATZ AUSWÄHLEN UND EINLESEN
# ============================================================

if INTERACTIVE_SELECTION:
    DATASET_MODE = choose_option(
        "DATENSATZ AUSWÄHLEN",
        {
            "train_filtered": "Gefilterte Trainingsdaten",
            "train": "Originale train.csv",
            "test": "test.csv – ohne RUL möglich",
            "custom": "Eigenen CSV-Pfad eingeben"
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
    CSV_FILE = input("\nVollständigen CSV-Pfad eingeben: ").strip()
    DATASET_NAME = os.path.splitext(os.path.basename(CSV_FILE))[0]
else:
    raise ValueError("Unbekannter DATASET_MODE.")

df_original = pd.read_csv(CSV_FILE)

for required_column in ["uav_id", "flight_cycle"]:
    if required_column not in df_original.columns:
        raise ValueError(f"Spalte '{required_column}' fehlt.")

HAS_RUL = "RUL" in df_original.columns

telemetry_columns = sorted(
    [c for c in df_original.columns if c.startswith("telemetry_")],
    key=telemetry_sort_key
)

if not telemetry_columns:
    raise ValueError("Keine telemetry_-Spalten gefunden.")

uav_ids = np.sort(df_original["uav_id"].unique())

print("\n" + "=" * 70)
print("DATENSATZ")
print("=" * 70)
print(f"Datei:             {CSV_FILE}")
print(f"Datensatz:         {DATASET_NAME}")
print(f"RUL vorhanden:     {HAS_RUL}")
print(f"UAVs:              {len(uav_ids)}")
print(f"Telemetrien:       {len(telemetry_columns)}")
print(f"Datenpunkte:       {len(df_original)}")

if not HAS_RUL:
    print(
        "\nHINWEIS: Diese Datei enthält keine RUL-Spalte. "
        "Die lokale Regression wird deshalb automatisch gegen "
        "flight_cycle durchgeführt."
    )

print("\n" + "=" * 70)
print("TELEMETRIEN")
print("=" * 70)
for telemetry in telemetry_columns:
    number = telemetry_number(telemetry)
    print(f"{number:2d} = {telemetry}")

selected_telemetries = select_telemetries(telemetry_columns)


# ============================================================
# INTERAKTIVE EINSTELLUNGEN
# ============================================================

if INTERACTIVE_SELECTION:

    if HAS_RUL:
        X_AXIS = choose_option(
            "ABHÄNGIGKEIT / X-ACHSE",
            {
                "RUL": "Telemetrie in Abhängigkeit von RUL",
                "flight_cycle": "Telemetrie in Abhängigkeit vom Flugzyklus"
            }
        )
    else:
        X_AXIS = "flight_cycle"
        print(
            "\nKeine RUL-Spalte vorhanden -> "
            "X-Achse automatisch: flight_cycle"
        )

    TRANSFORM_MODE = choose_option(
        "FEATURE-TRANSFORMATION",
        {
            "raw": "Originaldaten ohne Transformation",
            "abs": "Betrag |x|",
            "square": "Quadratisch x²",
            "sqrt_abs": "Wurzel des Betrags sqrt(|x|)",
            "log_abs": "log(1 + |x|)",
            "signed_log": "sign(x) * log(1 + |x|)",
            "difference": "Differenz x(t) - x(t-1)",
            "abs_difference": "Betrag der Differenz",
            "percent_change": "Relative Änderung zum vorherigen Wert",
            "zscore": "Z-Score je UAV",
            "minmax": "Min-Max-Normierung je UAV",
            "rolling_mean": "Gleitender Mittelwert",
            "rolling_std": "Gleitende Standardabweichung"
        }
    )

    if TRANSFORM_MODE in ["rolling_mean", "rolling_std"]:
        print("\n" + "=" * 70)
        print("ROLLING WINDOW DER TRANSFORMATION")
        print("=" * 70)
        while True:
            try:
                ROLLING_WINDOW = int(input("\nFenstergröße der Transformation: "))
                if ROLLING_WINDOW > 1:
                    break
            except ValueError:
                pass
            print("Fenstergröße muss größer als 1 sein.")

    REGRESSION_TYPE = choose_option(
        "REGRESSIONSART",
        {
            "linear": "Lineare Regression / Gerade",
            "quadratic": "Quadratische Regression / Kurve 2. Ordnung",
            "cubic": "Kubische Regression / Kurve 3. Ordnung"
        }
    )


# ============================================================
# POLYNOMGRAD
# ============================================================

if REGRESSION_TYPE == "linear":
    POLY_DEGREE = 1
elif REGRESSION_TYPE == "quadratic":
    POLY_DEGREE = 2
elif REGRESSION_TYPE == "cubic":
    POLY_DEGREE = 3
else:
    raise ValueError("Unbekannter Regressionstyp.")


if INTERACTIVE_SELECTION:

    print("\n" + "=" * 70)
    print("FENSTERGRÖSSE DER LOKALEN REGRESSION")
    print("=" * 70)

    while True:
        try:
            REGRESSION_WINDOW_SIZE = int(
                input("\nAnzahl Messpunkte pro Regression [z.B. 10]: ")
            )
            if REGRESSION_WINDOW_SIZE > POLY_DEGREE:
                break
        except ValueError:
            pass
        print("Fenstergröße ist für den Polynomgrad zu klein.")

    SHOW_MEASUREMENT_POINTS = (
        choose_option(
            "MESSPUNKTE DARSTELLEN?",
            {
                "yes": "Regression MIT transformierten Messpunkten",
                "no": "Nur Regressionsgeraden / Regressionskurven"
            }
        ) == "yes"
    )

    REFERENCE_MODE = choose_option(
        "WO SOLL DER GRADIENT AUSGEWERTET WERDEN?",
        {
            "center": "In der Mitte des lokalen Fensters",
            "end": "Am Ende des Fensters – sinnvoll für ML"
        }
    )

    print("\n" + "=" * 70)
    print("UAVs PRO ABBILDUNG")
    print("=" * 70)
    print("0 -> alle UAVs gemeinsam")

    while True:
        try:
            UAVS_PER_FIGURE = int(input("\nAnzahl UAVs pro Bild: "))
            if UAVS_PER_FIGURE >= 0:
                break
        except ValueError:
            pass
        print("Bitte 0 oder eine positive ganze Zahl eingeben.")

    PLOT_LAYOUT = choose_option(
        "PLOT-AUFTEILUNG",
        {
            "subplots": "Alle ausgewählten Telemetrien als Subplots",
            "individual": "Jede Telemetrie als einzelner Plot"
        }
    )

    RESULT_MODE = choose_option(
        "ERGEBNIS",
        {
            "regression": "Nur lokale Regressionen",
            "gradient": "Nur lokale Gradienten",
            "both": "Regression UND Gradient"
        }
    )

    SAVE_TRANSFORMED_CSV = (
        choose_option(
            "TRANSFORMIERTE DATEN ALS CSV SPEICHERN?",
            {"yes": "Ja", "no": "Nein"}
        ) == "yes"
    )

    SAVE_REGRESSION_FEATURES = (
        choose_option(
            "REGRESSIONSERGEBNISSE ALS CSV SPEICHERN?",
            {
                "yes": "Ja – lokale Regressionen + Zusammenfassung speichern",
                "no": "Nein – nur Plots erzeugen"
            }
        ) == "yes"
    )


# ============================================================
# EINSTELLUNGEN AUSGEBEN
# ============================================================

print("\n" + "=" * 70)
print("AUSGEWÄHLTE EINSTELLUNGEN")
print("=" * 70)
print(f"Datensatz:                   {DATASET_NAME}")
print(f"RUL vorhanden:               {HAS_RUL}")
print(f"Telemetrien:                 {', '.join(selected_telemetries)}")
print(f"X-Achse:                     {X_AXIS}")
print(f"Transformation:              {TRANSFORM_MODE}")
if TRANSFORM_MODE in ["rolling_mean", "rolling_std"]:
    print(f"Transformationsfenster:      {ROLLING_WINDOW}")
print(f"Regression:                  {REGRESSION_TYPE}")
print(f"Polynomgrad:                 {POLY_DEGREE}")
print(f"Regressionsfenster:          {REGRESSION_WINDOW_SIZE}")
print(f"Messpunkte anzeigen:         {SHOW_MEASUREMENT_POINTS}")
print(f"Gradient-Referenz:           {REFERENCE_MODE}")
print(f"UAVs pro Abbildung:          {UAVS_PER_FIGURE}")
print(f"Plot Layout:                 {PLOT_LAYOUT}")
print(f"Ergebnis:                    {RESULT_MODE}")
print(f"Regression CSV speichern:    {SAVE_REGRESSION_FEATURES}")


# ============================================================
# 1. TRANSFORMATION
# ============================================================
# Nur ausgewählte Telemetrien werden transformiert.
# Alle anderen Spalten bleiben erhalten.
# ============================================================

def transform_uav_data(uav_data):
    uav_data = uav_data.copy().sort_values("flight_cycle")
    telemetry_data = uav_data[selected_telemetries].copy()

    if TRANSFORM_MODE == "raw":
        transformed = telemetry_data

    elif TRANSFORM_MODE == "abs":
        transformed = telemetry_data.abs()

    elif TRANSFORM_MODE == "square":
        transformed = telemetry_data ** 2

    elif TRANSFORM_MODE == "sqrt_abs":
        transformed = np.sqrt(telemetry_data.abs())

    elif TRANSFORM_MODE == "log_abs":
        transformed = np.log1p(telemetry_data.abs())

    elif TRANSFORM_MODE == "signed_log":
        transformed = np.sign(telemetry_data) * np.log1p(telemetry_data.abs())

    elif TRANSFORM_MODE == "difference":
        transformed = telemetry_data.diff()

    elif TRANSFORM_MODE == "abs_difference":
        transformed = telemetry_data.diff().abs()

    elif TRANSFORM_MODE == "percent_change":
        transformed = telemetry_data.pct_change(fill_method=None)
        transformed = transformed.replace([np.inf, -np.inf], np.nan)

    elif TRANSFORM_MODE == "zscore":
        mean = telemetry_data.mean()
        std = telemetry_data.std().replace(0, np.nan)
        transformed = (telemetry_data - mean) / std

    elif TRANSFORM_MODE == "minmax":
        minimum = telemetry_data.min()
        maximum = telemetry_data.max()
        value_range = (maximum - minimum).replace(0, 1)
        transformed = (telemetry_data - minimum) / value_range

    elif TRANSFORM_MODE == "rolling_mean":
        transformed = telemetry_data.rolling(
            window=ROLLING_WINDOW,
            min_periods=1
        ).mean()

    elif TRANSFORM_MODE == "rolling_std":
        transformed = telemetry_data.rolling(
            window=ROLLING_WINDOW,
            min_periods=2
        ).std()

    else:
        raise ValueError(f"Unbekannte Transformation: {TRANSFORM_MODE}")

    uav_data[selected_telemetries] = transformed
    return uav_data


print("\n" + "=" * 70)
print("1. TRANSFORMATION")
print("=" * 70)

transformed_parts = []

for index, uav_id in enumerate(uav_ids, start=1):
    uav_data = df_original[df_original["uav_id"] == uav_id].copy()
    transformed_parts.append(transform_uav_data(uav_data))

    if index % 10 == 0 or index == len(uav_ids):
        print(f"[{index}/{len(uav_ids)}] UAVs transformiert")


df_transformed = pd.concat(
    transformed_parts,
    ignore_index=True
)


# ============================================================
# AUSGABEORDNER
# ============================================================

analysis_name = (
    f"{DATASET_NAME}_{TRANSFORM_MODE}_{X_AXIS}_{REGRESSION_TYPE}_"
    f"regwindow_{REGRESSION_WINDOW_SIZE}"
)

if TRANSFORM_MODE in ["rolling_mean", "rolling_std"]:
    analysis_name += f"_transformwindow_{ROLLING_WINDOW}"

analysis_folder = os.path.join(OUTPUT_FOLDER, analysis_name)
regression_folder = os.path.join(analysis_folder, "Regression")
gradient_folder = os.path.join(analysis_folder, "Gradient")
data_folder = os.path.join(analysis_folder, "Data")

for folder in [analysis_folder, regression_folder, gradient_folder, data_folder]:
    os.makedirs(folder, exist_ok=True)


# ============================================================
# TRANSFORMIERTE CSV SPEICHERN
# ============================================================

transformed_csv_filepath = None

if SAVE_TRANSFORMED_CSV:
    transformed_csv_filename = f"{DATASET_NAME}_transformed_{TRANSFORM_MODE}.csv"
    transformed_csv_filepath = os.path.join(
        data_folder,
        transformed_csv_filename
    )
    df_transformed.to_csv(transformed_csv_filepath, index=False)
    print("\nTransformierter Datensatz gespeichert:")
    print(transformed_csv_filepath)


# ============================================================
# UAV-GRUPPEN
# ============================================================

def create_uav_groups():
    if UAVS_PER_FIGURE == 0:
        return [uav_ids]

    return [
        uav_ids[start:start + UAVS_PER_FIGURE]
        for start in range(0, len(uav_ids), UAVS_PER_FIGURE)
    ]


uav_groups = create_uav_groups()


# ============================================================
# 2. LOKALE REGRESSION AUF TRANSFORMIERTEN DATEN
# ============================================================

print("\n" + "=" * 70)
print("2. LOKALE REGRESSION AUF TRANSFORMIERTEN DATEN")
print("=" * 70)

regression_results = []

for uav_number, uav_id in enumerate(uav_ids, start=1):

    uav_data = df_transformed[
        df_transformed["uav_id"] == uav_id
    ].copy().sort_values("flight_cycle")

    for telemetry in selected_telemetries:
        window_number = 0

        for start in range(
            0,
            len(uav_data),
            REGRESSION_WINDOW_SIZE
        ):
            end = min(
                start + REGRESSION_WINDOW_SIZE,
                len(uav_data)
            )

            segment = uav_data.iloc[start:end].copy()

            if (
                len(segment) < REGRESSION_WINDOW_SIZE
                and not USE_INCOMPLETE_LAST_WINDOW
            ):
                continue

            segment = segment.replace([np.inf, -np.inf], np.nan)
            segment = segment.dropna(subset=[X_AXIS, telemetry])

            if len(segment) < POLY_DEGREE + 1:
                continue

            x = segment[X_AXIS].to_numpy(dtype=float)
            y = segment[telemetry].to_numpy(dtype=float)

            if len(np.unique(x)) <= POLY_DEGREE:
                continue

            window_number += 1

            coefficients = np.polyfit(x, y, POLY_DEGREE)
            polynomial = np.poly1d(coefficients)
            y_pred = polynomial(x)
            r_squared = calculate_r_squared(y, y_pred)

            if REFERENCE_MODE == "center":
                x_reference = float(np.mean(x))
                flight_cycle_reference = float(segment["flight_cycle"].mean())
                rul_reference = (float(segment["RUL"].mean()) if HAS_RUL else np.nan)

            elif REFERENCE_MODE == "end":
                x_reference = float(segment[X_AXIS].iloc[-1])
                flight_cycle_reference = float(segment["flight_cycle"].iloc[-1])
                rul_reference = (float(segment["RUL"].iloc[-1]) if HAS_RUL else np.nan)

            else:
                raise ValueError(
                    "REFERENCE_MODE muss 'center' oder 'end' sein."
                )

            y_reference = float(polynomial(x_reference))

            first_derivative = polynomial.deriv(1)
            gradient = float(first_derivative(x_reference))

            if POLY_DEGREE >= 2:
                second_derivative = polynomial.deriv(2)
                curvature = float(second_derivative(x_reference))
            else:
                curvature = 0.0

            regression_results.append(
                {
                    "uav_id": uav_id,
                    "telemetry": telemetry,
                    "transform_mode": TRANSFORM_MODE,
                    "regression_type": REGRESSION_TYPE,
                    "polynomial_degree": POLY_DEGREE,
                    "regression_window_size": REGRESSION_WINDOW_SIZE,
                    "window": window_number,
                    "start_index": start,
                    "end_index": end - 1,
                    "n_points": len(segment),
                    "flight_cycle_start": segment["flight_cycle"].iloc[0],
                    "flight_cycle_end": segment["flight_cycle"].iloc[-1],
                    "RUL_start": (segment["RUL"].iloc[0] if HAS_RUL else np.nan),
                    "RUL_end": (segment["RUL"].iloc[-1] if HAS_RUL else np.nan),
                    "x_axis": X_AXIS,
                    "reference_mode": REFERENCE_MODE,
                    "x_reference": x_reference,
                    "flight_cycle_reference": flight_cycle_reference,
                    "RUL_reference": rul_reference,
                    "y_reference": y_reference,
                    "gradient": gradient,
                    "abs_gradient": abs(gradient),
                    "curvature": curvature,
                    "r_squared": r_squared,
                    "coefficients": coefficients.tolist()
                }
            )

    print(f"[{uav_number}/{len(uav_ids)}] {uav_id}")


results_df = pd.DataFrame(regression_results)

if results_df.empty:
    raise ValueError(
        "Keine lokalen Regressionen konnten berechnet werden."
    )


# ============================================================
# REGRESSIONSFEATURES CSV
# ============================================================

regression_csv_filepath = None

if SAVE_REGRESSION_FEATURES:
    regression_csv_filename = (
        f"{DATASET_NAME}_local_regression_features_{TRANSFORM_MODE}_{X_AXIS}_"
        f"{REGRESSION_TYPE}_window_{REGRESSION_WINDOW_SIZE}.csv"
    )

    regression_csv_filepath = os.path.join(
        data_folder,
        regression_csv_filename
    )

    results_df.to_csv(
        regression_csv_filepath,
        index=False
    )

    print("\nLokale Regressionsfeatures gespeichert:")
    print(regression_csv_filepath)


# ============================================================
# PLOTFUNKTIONEN
# ============================================================

def get_colors(current_uavs):
    cmap = plt.get_cmap("turbo", max(len(current_uavs), 2))
    return {
        uav_id: cmap(i)
        for i, uav_id in enumerate(current_uavs)
    }


def plot_regression_on_axis(ax, telemetry, current_uavs):
    colors = get_colors(current_uavs)

    for uav_id in current_uavs:

        uav_data = df_transformed[
            df_transformed["uav_id"] == uav_id
        ].copy().sort_values("flight_cycle")

        if SHOW_MEASUREMENT_POINTS:
            valid_data = (
                uav_data
                .replace([np.inf, -np.inf], np.nan)
                .dropna(subset=[X_AXIS, telemetry])
            )

            ax.scatter(
                valid_data[X_AXIS],
                valid_data[telemetry],
                color=colors[uav_id],
                s=8,
                alpha=0.25
            )

        for start in range(
            0,
            len(uav_data),
            REGRESSION_WINDOW_SIZE
        ):
            end = min(
                start + REGRESSION_WINDOW_SIZE,
                len(uav_data)
            )

            segment = uav_data.iloc[start:end].copy()

            if (
                len(segment) < REGRESSION_WINDOW_SIZE
                and not USE_INCOMPLETE_LAST_WINDOW
            ):
                continue

            segment = segment.replace([np.inf, -np.inf], np.nan)
            segment = segment.dropna(subset=[X_AXIS, telemetry])

            if len(segment) < POLY_DEGREE + 1:
                continue

            x = segment[X_AXIS].to_numpy(dtype=float)
            y = segment[telemetry].to_numpy(dtype=float)

            if len(np.unique(x)) <= POLY_DEGREE:
                continue

            coefficients = np.polyfit(x, y, POLY_DEGREE)
            polynomial = np.poly1d(coefficients)

            x_fit = np.linspace(np.min(x), np.max(x), 100)
            y_fit = polynomial(x_fit)

            ax.plot(
                x_fit,
                y_fit,
                color=colors[uav_id],
                linewidth=1.6,
                alpha=0.9
            )

    ax.set_title(telemetry, fontsize=10)
    ax.set_xlabel(X_AXIS, fontsize=8)
    ax.set_ylabel(f"{TRANSFORM_MODE}({telemetry})", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7)


def plot_gradient_on_axis(ax, telemetry, current_uavs):
    colors = get_colors(current_uavs)

    for uav_id in current_uavs:
        data = results_df[
            (results_df["uav_id"] == uav_id)
            &
            (results_df["telemetry"] == telemetry)
        ].copy()

        if data.empty:
            continue

        data = data.sort_values("x_reference")

        ax.plot(
            data["x_reference"].to_numpy(),
            data["gradient"].to_numpy(),
            color=colors[uav_id],
            linewidth=1.0,
            marker="o",
            markersize=3,
            alpha=0.8,
            label=uav_id
        )

    ax.axhline(0, linewidth=0.8, alpha=0.5)
    ax.set_title(telemetry, fontsize=10)
    ax.set_xlabel(X_AXIS, fontsize=8)
    ax.set_ylabel(
        f"d[{TRANSFORM_MODE}({telemetry})] / d({X_AXIS})",
        fontsize=8
    )
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7)


def create_subplot_figures(plot_function, folder, prefix, title):
    for group_index, current_uavs in enumerate(uav_groups, start=1):

        n = len(selected_telemetries)
        nrows = int(np.ceil(n / NCOLS))

        fig, axes = plt.subplots(
            nrows,
            NCOLS,
            figsize=(20, 4.3 * nrows)
        )

        axes = np.atleast_1d(axes).flatten()

        for i, telemetry in enumerate(selected_telemetries):
            plot_function(
                axes[i],
                telemetry,
                current_uavs
            )

        for i in range(n, len(axes)):
            axes[i].remove()

        if SHOW_LEGEND:
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=min(len(current_uavs), 10),
                fontsize=8
            )

        first_uav = current_uavs[0]
        last_uav = current_uavs[-1]

        fig.suptitle(
            f"{title}\n"
            f"{first_uav} – {last_uav} | "
            f"Transformation = {TRANSFORM_MODE} | "
            f"Reg.Window = {REGRESSION_WINDOW_SIZE} | "
            f"{REGRESSION_TYPE}",
            fontsize=15
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        filename = (
            f"{prefix}_{TRANSFORM_MODE}_"
            f"group_{group_index:02d}_"
            f"{first_uav}_to_{last_uav}.png"
        )

        filepath = os.path.join(folder, filename)

        plt.savefig(
            filepath,
            dpi=DPI,
            bbox_inches="tight"
        )

        plt.close(fig)
        print(f"Gespeichert: {filename}")


def create_individual_figures(plot_function, folder, prefix, title):
    for telemetry in selected_telemetries:
        for group_index, current_uavs in enumerate(uav_groups, start=1):

            fig, ax = plt.subplots(figsize=(11, 6))

            plot_function(
                ax,
                telemetry,
                current_uavs
            )

            first_uav = current_uavs[0]
            last_uav = current_uavs[-1]

            ax.set_title(
                f"{title}\n"
                f"{telemetry} | "
                f"{first_uav} – {last_uav} | "
                f"{TRANSFORM_MODE}"
            )

            if SHOW_LEGEND:
                ax.legend(fontsize=7)

            plt.tight_layout()

            filename = (
                f"{prefix}_{TRANSFORM_MODE}_{telemetry}_"
                f"group_{group_index:02d}.png"
            )

            filepath = os.path.join(folder, filename)

            plt.savefig(
                filepath,
                dpi=DPI,
                bbox_inches="tight"
            )

            plt.close(fig)
            print(f"Gespeichert: {filename}")


# ============================================================
# 3. PLOTS ERSTELLEN
# ============================================================

if RESULT_MODE in ["regression", "both"]:
    print("\n" + "=" * 70)
    print("3. REGRESSIONSPLOTS")
    print("=" * 70)

    if PLOT_LAYOUT == "subplots":
        create_subplot_figures(
            plot_regression_on_axis,
            regression_folder,
            "Regression",
            f"Lokale Regression der transformierten Daten vs {X_AXIS}"
        )
    else:
        create_individual_figures(
            plot_regression_on_axis,
            regression_folder,
            "Regression",
            f"Lokale Regression der transformierten Daten vs {X_AXIS}"
        )


if RESULT_MODE in ["gradient", "both"]:
    print("\n" + "=" * 70)
    print("4. GRADIENTENPLOTS")
    print("=" * 70)

    if PLOT_LAYOUT == "subplots":
        create_subplot_figures(
            plot_gradient_on_axis,
            gradient_folder,
            "Gradient",
            f"Lokaler Gradient nach Transformation vs {X_AXIS}"
        )
    else:
        create_individual_figures(
            plot_gradient_on_axis,
            gradient_folder,
            "Gradient",
            f"Lokaler Gradient nach Transformation vs {X_AXIS}"
        )


# ============================================================
# REGRESSIONSQUALITÄT + SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("REGRESSIONSQUALITÄT")
print("=" * 70)

mean_r2 = results_df["r_squared"].mean()
median_r2 = results_df["r_squared"].median()

print(f"Mittleres R²: {mean_r2:.4f}")
print(f"Median R²:    {median_r2:.4f}")

summary = (
    results_df
    .groupby("telemetry")
    .agg(
        mean_gradient=("gradient", "mean"),
        median_gradient=("gradient", "median"),
        mean_abs_gradient=("abs_gradient", "mean"),
        mean_r_squared=("r_squared", "mean"),
        n_windows=("window", "count")
    )
    .reset_index()
)

summary_filepath = None

if SAVE_REGRESSION_FEATURES:
    summary_filepath = os.path.join(
        data_folder,
        f"{DATASET_NAME}_regression_summary_by_telemetry.csv"
    )

    summary.to_csv(
        summary_filepath,
        index=False
    )


# ============================================================
# FERTIG
# ============================================================

print("\n" + "=" * 70)
print("FERTIG")
print("=" * 70)

print("\nAnalyseordner:")
print(os.path.abspath(analysis_folder))

if transformed_csv_filepath is not None:
    print("\nTransformierte Daten:")
    print(os.path.abspath(transformed_csv_filepath))

if regression_csv_filepath is not None:
    print("\nRegressionsfeatures:")
    print(os.path.abspath(regression_csv_filepath))

if summary_filepath is not None:
    print("\nZusammenfassung:")
    print(os.path.abspath(summary_filepath))

if RESULT_MODE in ["regression", "both"]:
    print("\nRegressionsplots:")
    print(os.path.abspath(regression_folder))

if RESULT_MODE in ["gradient", "both"]:
    print("\nGradientenplots:")
    print(os.path.abspath(gradient_folder))
