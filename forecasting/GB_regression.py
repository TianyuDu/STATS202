"""
The main file for gradient boosting regressor.
"""
import warnings
from typing import Union
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

sys.path.append("../")
import forecasting.forecasting_utility as utils

# **** Modify model here ****
PARAMS = {
    "n_estimators": 1700,
    "max_depth": 200,
    "learning_rate": 0.1,
    "loss": "ls",
    "subsample": 1.0,
    "criterion": "friedman_mse",
    "max_features": "auto",
}

PARAMS = {'criterion': 'friedman_mse', 'learning_rate': 0.01,
          'max_depth': 3, 'max_features': 'auto', 'n_estimators': 500}

# **** add configuration here ****
PARAM_SCOPE = {
    # "max_depth": [2 ** x for x in range(5, 14)],
    "max_depth": [3, 6, 9, 12, 15, 18],
    "learning_rate": [0.01, 0.03, 0.1, 0.3],
    "n_estimators": [100 * x for x in range(1, 40, 2)],
    "max_features": ["auto", "sqrt", "log2"],
    "criterion": ["friedman_mse"],
}

SCORE = "neg_mean_squared_error"
# **** end ****


def predict(path: Union[str, None] = None) -> None:
    """
    Generates the classification result for the given dataset.
    If a path of destination is given, this methods will write
    classification prediction to the destination file.
    """
    # Reading the training data.
    X_train, y_train, X_test = utils.get_data()
    # **** Extract ndarray ****
    X_train = X_train.values
    y_train = y_train.values.reshape(-1,)
    X_test = X_test.values

    model = GradientBoostingRegressor(
        **PARAMS,
        random_state=42,
        verbose=1
    )
    # **** End modification ****
    # Phase 1: fit the model, and estimate the loss measure on testing set
    # using a development set.
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train, y_train, test_size=0.5, random_state=42
    )
    print("Phase 1: fitting model on selected subset of training set...")
    model.fit(X_train, y_train)
    print("Estimating the test loss using a dev set...")
    pred_dev = model.predict(X_dev)
    # Estimate loss on dev set:
    print("Loss on dev set : {}".format(
        metrics.mean_squared_error(y_true=y_dev, y_pred=pred_dev)
    ))
    print("Phase 2: fitting model on the entire training set ...")
    # Re-fit using the entire traininig set.
    X_train, y_train, X_test = utils.get_data()
    model.fit(X_train, y_train)
    print("Predicting on the test set ...")
    pred_test = model.predict(X_test)
    # Write to file.
    if path is not None:
        print("Write submission file to {}".format(path))
        utils.generate_submission(pred_test, path)


def grid_search(path: Union[str, None] = None) -> None:
    X_train, y_train, X_test = utils.get_data()
    # **** Extract ndarray ****
    X_train = X_train.values
    y_train = y_train.values.reshape(-1,)
    X_test = X_test.values

    # X_train, X_dev, y_train, y_dev = train_test_split(
    #     X_train, y_train, test_size=0.1, random_state=0)

    print("# Tuning hyper-parameters for {}\n".format(SCORE))

    model = GridSearchCV(
        GradientBoostingRegressor(),
        PARAM_SCOPE,
        cv=5,
        scoring=SCORE,
        error_score=np.nan,
        n_jobs=-1,
        verbose=1,
    )
    model.fit(X_train, y_train)

    print("Best parameters set found on development set:\n")
    print(model.best_params_)
    if path is not None:
        with open(path, "w") as f:
            f.write(str(model.best_params_))

    print("Grid scores on development set:\n")
    means = model.cv_results_["mean_test_score"]
    stds = model.cv_results_["std_test_score"]
    # Save CV result to local file
    if path is not None:
        pd.DataFrame(model.cv_results_).to_csv(
            path[:-3] + "csv", index=False)
    for mean, std, params in zip(means, stds, model.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    # print("\nDetailed classification report:\n")
    # print("The model is trained on the full development set.")
    # print("The scores are computed on the full evaluation set.")
    # y_true, y_pred = y_dev, model.predict(X_dev)
    # print("MSE: {}".format(
    #     metrics.mean_squared_error(y_true, y_pred)
    # ))


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
