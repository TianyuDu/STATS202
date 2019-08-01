"""
Main script.
"""
import warnings
import sys
from typing import Union, Optional, Tuple
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import model_selection

sys.path.append("../")
import util.data_proc
import util.features

import classification.DNNClassifier

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

    # Feature engerineering
    poly_degree = 1
    X_train, CROSS = util.features.polynomial_standardized(X_train, PANSS, poly_degree)
    X_test, _ = util.features.polynomial_standardized(X_test, PANSS, poly_degree)
    FEATURE += CROSS
    print("Design_train: {}, Design_test: {}".format(X_train.shape, X_test.shape))
    pred = classification.DNNClassifier.main(
        lambda: provide_data(X_train, y_train, X_test),
        EPOCHS=100, PERIOD=5, BATCH_SIZE=256,
        LR=1e-5, NEURONS=[256, 512],
        forecast=True, tuning=False)
    # Make predictions.
    holder = pd.read_csv("./data/sample_submission_status.csv", header=0)
    sub_name = input("File name to store submission: ")
    holder["LeadStatus"] = pred
    holder.to_csv("./submissions/{}.csv".format(sub_name), index=False)
