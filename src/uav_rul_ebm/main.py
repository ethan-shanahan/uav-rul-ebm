import pandas as pd
from sklearn.pipeline import Pipeline

import uav_rul_ebm.preprocessors as pp

TRAIN_PATH = "./data/raw/train.csv"
TEST_PATH = "./data/raw/test.csv"
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

pl = Pipeline(
    [
        (
            "variance threshold",
            pp.VarianceThreshold(verbose=True),
        ),
        (
            "outlier nullifier",
            pp.VerticalHampelFilter(rolling_window_size=30, n_sigmas=5, verbose=False),
        ),
        (
            "defect nullifier",
            pp.HorizontalHampelFilter(init_window_size=40, n_sigmas=5, verbose=False),
        ),
        (
            "imputer",
            pp.HGBIImputer(),
        ),
    ],
    verbose=True,
)

train = pl.fit_transform(train_df)
test = pl.transform(test_df)
