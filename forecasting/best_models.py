"""
This file stores the best model selected from grid search.
"""
from sklearn import ensemble
from sklearn import svm

best_models = [
    (
        ensemble.RandomForestRegressor,
        "RF",
        {"criterion": "mse", "max_depth": 4096,
            "max_features": "auto", "n_estimators": 500}
    ),
    (
        svm.SVR,
        "SVR_POLY",
        {"degree": 3, "kernel": "poly", "C": 4, "gamma": 0.0001}
    ),
    (
        svm.SVR,
        "SVR_RBF",
        {"C": 128, "gamma": 1e-05, "kernel": "rbf"}
    ),
    (
        ensemble.GradientBoostingRegressor,
        "GB",
        {"criterion": "mse", "max_depth": 4096,
            "max_features": "auto", "n_estimators": 500}
    )
]
