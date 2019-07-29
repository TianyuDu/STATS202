# The controlling for hyper-parameter searching
import warnings
import itertools
import numpy as np
import pandas as pd

from sklearn import model_selection

import data_proc
import features
import grid_search_util

import DNNClassifier

SCOPE = {
    "EPOCHS": [100, 200, 300],
    "BATCH_SIZE": 1024,
    "LR": [1e-5, 1e-5*3, 1e-4, 1e-4*3, 1e-3],
    "NEURONS": [
        [256, 512], [512, 512],
        [512, 1024], [1024, 1024]],
}


def provide_data(
        X_train, y_train, X_test,
        to_dir: str = None
):
    """
    NOTE: test set will never be shuffled.
    Provides training data to model or saves them to local files.
    Args:
        
    """
    X, X_dev, y, y_dev = model_selection.train_test_split(
        X_train, y_train, test_size=0.2, random_state=None, shuffle=True)

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
    df_train = data_proc.load_whole(path="./data/")
    print(df_train.shape)
    df_test = pd.read_csv("./data/Study_E.csv", header=0)
    print(df_test.shape)
    # Reduced countries
    major_countries = ["USA", "Russia", "Ukraine", "China", "Japan"]
    df_train = features.reduce_countries(df_train, major_countries)
    df_test = features.reduce_countries(df_test, major_countries)
    X_train, y_train, FEATURE, PANSS = data_proc.gen_slp_assessment(df_train)
    X_test = data_proc.parse_test_set(X_train, df_test)
    print(f"Design_train: {X_train.shape}, Design_test: {X_test.shape}")

    # Feature Engerineering
    poly_degree = 3
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
