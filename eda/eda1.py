from pathlib import Path

import seaborn as sb
from sklearn.feature_selection import VarianceThreshold

from uav_rul_ebm.utils import *

PROJECT_ROOT = Path.cwd()
TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "raw" / "test.csv"
PLOT_DIR = PROJECT_ROOT / "eda" / "plots"

PROTOTYPING = True

# Raw
raw, train_raw, test_raw = load_data(TRAIN_PATH, TEST_PATH, PROTOTYPING)
base_telemetries = telemetry_columns(raw)
# print(base_telemetries)
# Flatline Filtration
data = raw.drop(columns=base_telemetries).join(
    VarianceThreshold(1e-6)
    .set_output(transform="pandas")
    .fit_transform(raw[base_telemetries])
)
print("plotting...")
sb.displot(data=data, x="telemetry_01", stat="density", common_norm=True, row="split")
# sb.displot(
#     data=data,
#     x="telemetry_01",
#     kind="kde",
#     common_norm=True,
#     common_grid=True,
#     fill=True,
#     row="split",
#     hue="uav_id",
#     legend=False,
#     height=6,
#     aspect=2,
# )
print("showing...")
plt.show()
# plotting
# for t in telemetry_columns(data):
#     print(f"Plotting: {t}.png")
#     plot = sb.displot(
#         data=data,
#         x=t,
#         kind="kde",
#         common_norm=True,
#         common_grid=True,
#         fill=True,
#         row="split",
#         hue="uav_id",
#         legend=False,
#         height=6,
#         aspect=2,
#     )
#     plt.show()
#     print("Finished: plotting")
#     plot.savefig(PLOT_DIR / "telemetry_kdeplots" / f"{t}.png")
#     print("Finished: saving")
#     plt.close()
#     print("Finished: closing")
