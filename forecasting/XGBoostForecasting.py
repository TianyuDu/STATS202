"""
XGBoost for the forecasting/regression task.
"""
import numpy as np
import pandas as pd
import xgboost as xgb

import sys
sys.path.append("../")

from forecasting.main_forecasting import get_data


def set_params() -> dict:
    p = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
    p['nthread'] = 4
    p['eval_metric'] = 'auc'
    p['eval_metric'] = ['auc', 'ams@0']
    return p


def main():
    df_train, df_test, X_train, y_train, X_test = get_data()
    dtrain = xgb.DMatrix(X_train.values, label=y_train.values)
    params = set_params()
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = 10
    bst = xgb.train(params, dtrain, num_round)


if __name__ == "__main__":
    main()
