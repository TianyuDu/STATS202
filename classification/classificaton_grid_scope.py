# ******** Random Forest ********
# RF_grid_result.txt
# Fitting 5 folds for each of 420 candidates, totalling 2100 fits
PARAM_SCOPE = {
    "max_depth": [None] + [2 ** x for x in range(5, 11)],
    "n_estimators": [100 * x for x in range(1, 20, 2)],
    "criterion": ["entropy", "gini"],
    "max_features": ["auto", "sqrt", "log2"],
}

SCORE = "neg_log_loss"
# ******** End ********

# ******** Large Learning Rate Gradient Boosting ******** 
PARAM_SCOPE = {
    # "max_depth": [2 ** x for x in range(5, 14)],
    "max_depth": [3, 6, 9, 12],
    "learning_rate": [0.01, 0.03, 0.1, 0.3],
    "n_estimators": [100 * x for x in range(1, 20, 2)],
    "max_features": ["auto", "sqrt", "log2"],
    "criterion": ["friedman_mse"],
}

SCORE = "neg_log_loss"
# ******** End ********

# ******** Small Learning Rate Gradient Boosting ********
PARAM_SCOPE = {
    # "max_depth": [2 ** x for x in range(5, 14)],
    "max_depth": [3, 6, 9, 12],
    "learning_rate": [0.001, 0.003],
    "n_estimators": [100 * x for x in range(1, 20, 2)],
    "max_features": ["auto", "sqrt", "log2"],
    "criterion": ["friedman_mse"],
}

SCORE = "neg_log_loss"
# ******** End ********