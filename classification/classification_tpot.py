"""
TPOT for the classification task.
"""
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from tpot import TPOTClassifier
import dask

sys.path.append("../")
import util.data_proc
import util.features


if __name__ == "__main__":
    client = Client(n_workers=4, threads_per_worker=1)
    df_train = util.data_proc.load_whole(path="../data/")
    print(df_train.shape)
    df_test = pd.read_csv("../data/Study_E.csv", header=0)
    print(df_test.shape)
    # Reduced countries
    major_countries = ["USA", "Russia", "Ukraine", "China", "Japan"]
    df_train = util.features.reduce_countries(df_train, major_countries)
    df_test = util.features.reduce_countries(df_test, major_countries)
    X_train, y_train, FEATURE, PANSS = util.data_proc.gen_slp_assessment(df_train)
    X_test = util.data_proc.parse_test_set(X_train, df_test)
    print(f"Design_train: {X_train.shape}, Design_test: {X_test.shape}")

    date = datetime.strftime(
        datetime.now(), "%Y_%m_%d_%H_%M")

    optimizer = TPOTClassifier(
        generations=5,
        population_size=20,
        cv=5,
        random_state=42,
        verbosity=2,
    )
    optimizer.fit(X_train.values, y_train.values)
    optimizer.export(f"tpot_{date}.py")
    print(f"Pipeline file exported to: tpot_{date}.py")

