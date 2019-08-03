"""
Grid Search for random forest classification
"""
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from main_classification import get_data
from main_classification import classification_write_to_file


def main(X: pd. DataFrame, y: pd.DataFrame) -> None:
    X_train, X_dev, y_train, y_dev = train_test_split(
        X, y, test_size=0.3, random_state=0)

    param_scope = {
        "max_depth": [None, 20, 30, 50, 100, 200, 300, 500],
        "n_estimators": [100 * x for x in range(1, 20, 2)],
        "criterion": ["entropy", "gini"],
    }

    # Chooose score here.
    score = "neg_log_loss"

    print("# Tuning hyper-parameters for {}\n".format(score))

    model = GridSearchCV(
        RandomForestClassifier(),
        param_scope,
        cv=5,
        scoring=score,
        error_score=np.nan,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)

    print("Best parameters set found on development set:\n")
    print(model.best_params_)
    now = datetime.strftime(datetime.now(), "%Y_%m_%d_%H_%M")
    with open("./RF_grid_search_{}.txt".format(now), "w") as f:
        f.write(str(model.best_params_))

    print("Grid scores on development set:\n")
    means = model.cv_results_["mean_test_score"]
    stds = model.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, model.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print("\nDetailed classification report:\n")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    y_true, y_pred = y_dev, model.predict(X_dev)
    print(metrics.classification_report(y_true, y_pred))

if __name__ == "__main__":
    X_train, y_train, X_test = get_data()
    main(X_train.values, y_train.values.reshape(-1,))
