"""
The main file for random forest
"""
import warnings

import numpy as np
import pandas as pd

from typing import Union

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

import classification.classification_util as utils


def classify(path: Union[str, None] = None) -> None:
    """
    Generates the classification result for the given dataset.
    If a path of destination is given, this methods will write
    classification prediction to the destination file.
    """
    # Reading the training data.
    X_train, y_train, X_test = utils.get_data()
    # **** Modify model here ****
    PARAMS = {
        "n_estimators": 500,
        "max_depth": 100,
        "criterion": "gini",
    }
    model = RandomForestClassifier(
        **PARAMS,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    # **** End modification ****
    # Phase 1: fit the model, and estimate the loss measure on testing set
    # using a development set.
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train, y_train, test_size=0.5, random_state=42
    )
    print("Phase 1: fitting model on selected subset of training set...")
    model.fit(X_train.values, y_train.values)
    print("Estimating the test loss using a dev set...")
    pred_dev = model.predict_proba(X_dev)
    # Estimate loss on dev set:
    print("Log loss on dev set : {}".format(
        log_loss(y_true=y_dev, y_pred=pred_dev[:, 1])
    ))
    print("Phase 2: fitting model on the entire training set ...")
    # Re-fit using the entire traininig set.
    X_train, y_train, X_test = utils.get_data()
    model.fit(X_train.values, y_train.values)
    print("Predicting on the test set ...")
    pred_test = model.predict_proba(X_test)
    # Write to file.
    # gen_sub = bool(int(input("Generate submission file? >>> ")))
    if path is not None:
        print("Write submission file to {}".format(path))
        utils.generate_submission(pred_test[:, 1], path)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    classify()
