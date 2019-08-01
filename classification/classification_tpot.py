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
    print("Design_train: {}, Design_test: {}".format(X_train.shape, X_test.shape))

    date = datetime.strftime(
        datetime.now(), "%Y_%m_%d_%H_%M")

    GEN = int(input("Number of Generations: "))
    POP = int(input("Population size: "))

    optimizer = TPOTClassifier(
        generations=GEN,
        population_size=POP,
        cv=5,
        random_state=42,
        verbosity=2,
    )
    optimizer.fit(X_train.values, y_train.values)
    optimizer.export("tpot_{}.py".format(date))
    print("Pipeline file exported to: tpot_{}.py".format(date))

