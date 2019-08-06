from sklearn import ensemble

best_models = [
    (  # RF: best
        ensemble.RandomForestClassifier,
        "RF",
        {'max_features': 'log2', 'criterion': 'gini',
            'n_estimators': 1900, 'max_depth': 64,
            "n_jobs": -1},
        "predict_proba"
    ),
    # (  # GB
    #     ensemble.GradientBoostingClassifier,
    #     "GB",
    #     {'max_features': 'sqrt', 'n_estimators': 700,
    #         'criterion': 'friedman_mse', 'max_depth': 6,
    #         'learning_rate': 0.003},
    #     "predict_proba"
    # ),
]
