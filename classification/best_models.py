"""
This file stores the best model selected from grid search.
"""
from sklearn import ensemble
from sklearn import svm
from sklearn import linear_model

best_models = [
    (
        linear_model.LogisticRegression,
        "Logisitc",
        {"solver": "saga", "C": 0.00390625,
            "l1_ratio": 0.98, "penalty": "elasticnet"},
        "predict_proba"
    ),
    (  # RF: best
        ensemble.RandomForestClassifier,
        "RF",
        {"max_features": "log2", "criterion": "gini",
            "n_estimators": 1900, "max_depth": 64,
            "n_jobs": -1},
        "predict_proba"
    ),
    (  # GB
        ensemble.GradientBoostingClassifier,
        "GB",
        {"max_features": "sqrt", "n_estimators": 700,
            "criterion": "friedman_mse", "max_depth": 6,
            "learning_rate": 0.003},
        "predict_proba"
    ),
    (  # SVC
        svm.SVC,
        "SVC",
        {"kernel": "rbf", "gamma": 0.0001, "probability": True, "C": 512},
        "predict_proba"
    )
]
