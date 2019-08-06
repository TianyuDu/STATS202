"""
The main file for Support Vector Classifier
"""
import warnings
from typing import Union
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import log_loss

sys.path.append("../")
import classification.classification_utility as utils


# **** Modify model here ****
PARAMS = {'degree': 3, 'kernel': 'poly', 'C': 4, 'gamma': 0.0001, "probability": True}

# **** add configuration here ****
# Scope for rbf.
# PARAM_SCOPE = {
#     "kernel": ["rbf"],
#     "gamma": ["auto"] + [10 ** (-x) for x in range(1, 10)],
#     "C": [2 ** x for x in range(1, 10)],
#     "probability": [True],
# }

# Scope for poly kernel
PARAM_SCOPE = {
    "kernel": ["poly"],
    "gamma": ["auto"],
    "C": [1.0],
    "degree": [3],
    "probability": [True],
}

SCORE = "neg_log_loss"
# **** end ****


def predict(path: Union[str, None] = None) -> None:
    """
    Generates the classification result for the given dataset.
    If a path of destination is given, this methods will write
    classification prediction to the destination file.
    """
    # Reading the training data.
    X_train, y_train, X_test = utils.get_data()
    model = SVC(
        **PARAMS,
        random_state=42,
        verbose=1
    )
    print("Parameter used: {}".format(PARAMS))
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
    print("Loss on dev set : {}".format(
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


def grid_search(path: Union[str, None] = None) -> None:
    X_train, y_train, X_test = utils.get_data()
    # X_train, X_dev, y_train, y_dev = train_test_split(
    #     X_train, y_train, test_size=0.3, random_state=0)

    print("# Tuning hyper-parameters for {}\n".format(SCORE))

    model = GridSearchCV(
        SVC(),
        PARAM_SCOPE,
        cv=3,
        scoring=SCORE,
        error_score=np.nan,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)

    print("Best parameters set found on development set:\n")
    print(model.best_params_)
    if path is not None:
        with open(path, "w") as f:
            f.write(str(model.best_params_))
    # Save CV result to local file
    if path is not None:
        pd.DataFrame(model.cv_results_).to_csv(
            path[:-3] + "csv", index=False)

    print("Grid scores on development set:\n")
    means = model.cv_results_["mean_test_score"]
    stds = model.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, model.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    # print("\nDetailed classification report:\n")
    # print("The model is trained on the full development set.")
    # print("The scores are computed on the full evaluation set.")
    # y_true, y_pred = y_dev, model.predict(X_dev)
    # print(metrics.classification_report(y_true, y_pred))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--task", type=str, default=None)
    parser.add_argument(
        "--logdir", default=None, type=str)
    args = parser.parse_args()
    if args.task == "predict":
        print("Execute task: {}".format(args.task))
        if args.logdir is None:
            print("No log directory is provided, no submission file will be generated.")
        predict(path=args.logdir)
    elif args.task == "grid":
        print("Execute task: {}".format(args.task))
        if args.logdir is None:
            print("No log directory is provided, best model chosen will only be printed.")
        grid_search(path=args.logdir)
    else:
        raise SyntaxError("{} task provided is unavaiable.".format(args.task))
