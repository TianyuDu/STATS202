"""
Main script.
"""
import warnings
from typing import Union, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection

import data_proc
import features

import DNNClassifier

def plot():
    fig1 = sns.catplot(x="Country", kind="count", data=df_train)
    fig1.set_xticklabels(rotation=30)
    fig2 = sns.catplot(x="Country", kind="count", data=df_test)


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

    # Feature engerineering
    poly_degree = 3
    X_train, CROSS = features.polynomial_standardized(X_train, PANSS, poly_degree)
    X_test, _ = features.polynomial_standardized(X_test, PANSS, poly_degree)
    FEATURE += CROSS
    print(f"Design_train: {X_train.shape}, Design_test: {X_test.shape}")
    pred = DNNClassifier.main(
        lambda: provide_data(X_train, y_train, X_test),
        EPOCHS=100, PERIOD=5, forecast=True)
    holder = pd.read_csv("./data/sample_submission_status.csv", header=0)
    sub_name = input("File name to store submission: ")
    holder["LeadStatus"] = pred
    holder.to_csv(f"./submissions/{sub_name}.csv", index=False)