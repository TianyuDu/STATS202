"""
Main scripts for the classification task.
Including data loading utilities and methods
to generate forecastings.
"""
import numpy as np
import pandas as pd
import sys

sys.path.append("../")
from util import data_proc
from util import features


def get_data():
    """
    
    """
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
    # Use polynomial degree 1 means no polynomial's generated.
    poly_degree = 1
    X_train, CROSS = features.polynomial_standardized(X_train, PANSS, poly_degree)
    X_test, _ = features.polynomial_standardized(X_test, PANSS, poly_degree)
    FEATURE += CROSS
    return X_train, y_train, X_test


def classification_write_to_file(pred: np.ndarray, path: str) -> None:
    """
    Writes the classification result to local file for submission.
    """
    holder = pd.read_csv("../data/sample_submission_status.csv", header=0)
    assert len(holder) == len(pred)
    holder["LeadStatus"] = pred
    holder.to_csv(path, index=False)
