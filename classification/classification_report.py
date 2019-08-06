import sys
import argparse

import numpy as np
import pandas as pd
from datetime import datetime

from sklearn import metrics
from sklearn import model_selection
from sklearn import ensemble

sys.path.append("../")
import classification.classification_utility as utils
from classification.best_models import best_models


def model_cv_test(model, X, y, pred_fn: str, n_fold: int = 5):
    kf = model_selection.KFold(n_splits=n_fold, shuffle=True)
    cv_loss = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Fit the model
        model.fit(X_train, y_train)
        # Test the model
        if pred_fn == "predict":
            pred_test = model.predict(X_test)
        elif pred_fn == "predict_proba":
            pred_test = model.predict_proba(X_test)
        else:
            raise ValueError("Invalid predict_fn.")
        # Some model predicts both prob y=0 and prob y=1.
        # Extract prob y=1 only.
        if pred_test.shape[1] == 2:
            pred_test = pred_test[:, 1]
        loss = metrics.log_loss(y_test, pred_test)
        print("Log loss: {}".format(loss))
        cv_loss.append(loss)
    assert len(cv_loss) == n_fold
    return cv_loss


def record_cv_loss(name: str, cv_loss: list) -> pd.DataFrame:
    result = pd.DataFrame({
        "model": [name],
        "min": [np.min(cv_loss)],
        "max": [np.max(cv_loss)],
        "mean": [np.mean(cv_loss)]
    })
    return result


if __name__ == "__main__":
    # ==== Destination File ====
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", "--logdir", default=None, type=str)
    args = parser.parse_args()
    print(args.logdir)
    # ==== Load Data ====
    X_train, y_train, X_test = utils.get_data()
    start_time_overall = datetime.now()
    record = list()

    for (model, name, params, pred_fn) in best_models:
        print("Running CV for: {}".format(name))
        start_time = datetime.now()
        constructed_model = model(**params)
        cv_loss = model_cv_test(
            constructed_model,
            X_train.values, y_train.values,
            pred_fn=pred_fn
        )
        record.append(record_cv_loss(name, cv_loss))
        end_time = datetime.now()
        print("Time taken: {}".format(str(end_time - start_time)))
    record = pd.concat(record)
    record.to_csv(args.logdir, index=False)

    # # ==== Gradient Boosting Small LR ====
    # start_time = datetime.now()
    # PARAMS = {
    #     'max_features': 'sqrt', 'n_estimators': 700,
    #     'criterion': 'friedman_mse', 'max_depth': 6,
    #     'learning_rate': 0.003}
    # gbc_1 = ensemble.GradientBoostingClassifier(
    #     **PARAMS, random_state=42, verbose=0
    # )
    # cv_loss = model_cv_test(gbc_1, X_train.values, y_train.values, pred_fn="predict_proba")
    # record.append(record_cv_loss("GB_small_lr", cv_loss))
    # end_time = datetime.now()
    # print("Time taken: {}".format(str(end_time - start_time)))

    # # ==== Gradient Boosting with large LR ====
    # del(PARAMS)
    # PARAMS = {
    #     'max_depth': 6, 'n_estimators': 300, 'max_features': 'sqrt',
    #     'criterion': 'friedman_mse', 'learning_rate': 0.01}
    # gbc_2 = ensemble.GradientBoostingClassifier(
    #     **PARAMS, random_state=42, verbose=0
    # )
    # cv_loss = model_cv_test(
    #     gbc_2, X_train.values,
    #     y_train.values, pred_fn="predict_proba"
    # )
    # record.append(record_cv_loss("GB_small_lr", cv_loss))
    # end_time = datetime.now()
    # print("Time taken: {}".format(str(end_time - start_time)))
