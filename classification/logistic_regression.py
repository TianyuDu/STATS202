import warnings
import argparse
import sys

from typing import Union

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
sys.path.append("../")
import classification.classification_utility as utils


# **** Modify model here ****
# PARAMS = {'max_features': 'log2', 'criterion': 'gini',
#           'n_estimators': 1900, 'max_depth': 64}

# **** add configuration here ****
# Elastic Net
PARAM_SCOPE = {
    "penalty": ["elasticnet"],
    "C": [2 ** x for x in range(-10, 10)],
    "l1_ratio": [0.02 * x for x in range(51)],
    "solver": ["saga"]
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
    model = LogisticRegression(
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
    # Save CV result to local file
    if path is not None:
        pd.DataFrame(model.cv_results_).to_csv(
            path[:-3] + "csv", index=False)


def grid_search(path: Union[str, None] = None) -> None:
    X_train, y_train, X_test = utils.get_data()
    # X_train, X_dev, y_train, y_dev = train_test_split(
    #     X_train, y_train, test_size=0.1, random_state=0)

    print("# Tuning hyper-parameters for {}\n".format(SCORE))

    model = GridSearchCV(
        LogisticRegression(),
        PARAM_SCOPE,
        cv=5,
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
        "--logdir", default=None, type=str,
        help="The directory to store submission file.")
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