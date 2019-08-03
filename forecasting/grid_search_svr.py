"""
Grid Search for SVR
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVR

from main_forecasting import read_from_disk


def main(X: pd. DataFrame, y: pd.DataFrame) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0)

    param_scope = [
        {"kernel": ["rbf"], "gamma": [1e-3, 1e-4],
            "C": [1, 10, 100, 1000]},
        {"kernel": ["linear"], "C": [1, 10, 100, 1000]}]

    # Chooose score here.
    score = "neg_mean_squared_error"

    print("# Tuning hyper-parameters for {}\n".format(score))

    model = GridSearchCV(
        SVR(),
        param_scope,
        cv=5,
        scoring=score,
        error_score=np.nan,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("Best parameters set found on development set:\n")
    print(model.best_params_)
    print("Grid scores on development set:\n")
    means = model.cv_results_["mean_test_score"]
    stds = model.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, model.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print("\nDetailed classification report:\n")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    y_true, y_pred = y_test, model.predict(X_test)
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    X, y, X_test = read_from_disk()
    main(X.values, y.values.reshape(-1,))
