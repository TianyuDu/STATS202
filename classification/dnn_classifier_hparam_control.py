# The controlling for hyper-parameter searching
import sys
import warnings
import itertools
import numpy as np
import pandas as pd

from sklearn import model_selection

sys.path.append("../")

from util import data_proc
from util import features
from util import grid_search_util

import DNNClassifier

SCOPE = {
    "EPOCHS": 600,
    "BATCH_SIZE": [32, 512],
    "LR": [1e-3, 3*1e-3, 1e-4, 3*1e-4],
    "NEURONS": [
        [512] * 4,
        [1024] * 4,
        [2048] * 4,
        [512] * 5,
        [1024] * 5,
        [2048] * 5,
        [1024] * 6
    ]
}

# This is a smaller hyper-parameter searching scope for quick debugging.
# SCOPE = {
#     "EPOCHS": 50,
#     "BATCH_SIZE": 1024,
#     "LR": [1e-5, 1e-5*3, 1e-4],
#     "NEURONS": [
#         [32, 64],
#         [32, 64, 128],
#     ],
# }


def provide_data(
        X_train, y_train, X_test,
        to_dir: str = None,
        dev_ratio: float = 0.2,
):
    """
    NOTE: test set will never be shuffled.
    Provides training data to model or saves them to local files.
    Args:
        
    """
    X, X_dev, y, y_dev = model_selection.train_test_split(
        X_train, y_train, test_size=dev_ratio, random_state=None, shuffle=True)

    # Convert to np.float32, and extract features.
    X, X_dev, y, y_dev, X_test = map(
        lambda z: z.values.astype(np.float32),
        [X[FEATURE], X_dev[FEATURE], y, y_dev, X_test[FEATURE]]
    )
    # Convert labels (None,) -> (None, 1)
    y, y_dev = map(lambda z: z.reshape(-1, 1), [y, y_dev])
    return X, X_dev, y, y_dev, X_test


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    df_train = data_proc.load_whole(path="../data/")
    print(df_train.shape)
    df_test = pd.read_csv("../data/Study_E.csv", header=0)
    print(df_test.shape)
    # Reduced countries
    major_countries = ["USA", "Russia", "Ukraine", "China", "Japan"]
    df_train = features.reduce_countries(df_train, major_countries)
    df_test = features.reduce_countries(df_test, major_countries)
    X_train, y_train, FEATURE, PANSS = data_proc.gen_slp_assessment(df_train)
    X_test = data_proc.parse_test_set(X_train, df_test)
    print("X_train: {}, X_test: {}".format(X_train.shape, X_test.shape))

    # Feature Engerineering
    poly_degree = 2
    X_train, CROSS = features.polynomial_standardized(X_train, PANSS, poly_degree)
    X_test, _ = features.polynomial_standardized(X_test, PANSS, poly_degree)
    FEATURE += CROSS

    # Grid Search
    LOG_DIR = input("Dir to store the hparam tuning log: ")
    grid_search_util.grid_search(
        scope=SCOPE,
        data_feed=lambda: provide_data(X_train, y_train, X_test),
        train_main=DNNClassifier.main,
        log_dir=LOG_DIR
    )
