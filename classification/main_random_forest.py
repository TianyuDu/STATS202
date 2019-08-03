"""
The main file for random forest
"""
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from main_classification import get_data
from main_classification import classification_write_to_file


def main():
    X_train, y_train, X_test = get_data()
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=100,
        random_state=0,
        criterion="gini",
        n_jobs=-1,
        verbose=1
    )
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    print("Fitting Model...")
    model.fit(X_train.values, y_train.values)
    pred_dev = model.predict_proba(X_dev)
    # Estimate loss on dev set:
    print("Log loss: {}".format(log_loss(y_true=y_dev, y_pred=pred_dev[:, 1])))
    # Re-fit using the entire traininig set.
    X_train, y_train, X_test = get_data()
    model.fit(X_train.values, y_train.values)
    pred_test = model.predict_proba(X_test)
    # Write to file.
    gen_sub = bool(int(input("Generate submission file? >>> ")))
    if gen_sub:
        path = input("Path to write predcition: ")
        classification_write_to_file(pred_test[:, 1], path)



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
