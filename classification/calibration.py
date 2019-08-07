import sys
import matplotlib.pyplot as plt
import argparse

import numpy as np

from typing import Union, Optional

from sklearn import calibration
from sklearn import ensemble
from sklearn import model_selection

sys.path.append("../")
import classification.classification_utility as utils


def calibrate_model(
    model: "Estimator",
    X, y,
    path: str = None,
    X_test: Union[np.ndarray, None] = None
) -> Optional[np.array]:
    # Evaluate baseline model.
    print("Evaluating the uncalibrated model...")
    X_train, X_dev, y_train, y_dev = model_selection.train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model.fit(X_train, y_train)
    pred_dev = model.predict_proba(X_dev)
    fop, mpv = calibration.calibration_curve(y_dev, pred_dev[:, 1], n_bins=100)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(mpv, fop, marker=".")
    if path is None:
        plt.show()
    else:
        plt.savefig(path + "calibration_curve_raw_model.png", dpi=300)
    # Evaluate the calibrated model.
    print("Evaluating the calibrated model...")
    calibrator = calibration.CalibratedClassifierCV(
        model, method="isotonic", cv=10)
    calibrator.fit(X_train, y_train)
    pred_dev = calibrator.predict_proba(X_dev)

    fop, mpv = calibration.calibration_curve(y_dev, pred_dev[:, 1], n_bins=100)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(mpv, fop, marker=".")
    if path is None:
        plt.show()
    else:
        plt.savefig(path + "calibration_curve_calibrated_model.png", dpi=300)
    # Predicting on the test set
    if X_test is not None:
        print("Predicting test set results...")
        # Train using the whole training set.
        calibrator.fit(X, y)
        pred_test = calibrator.predict_proba(X_test)
        return pred_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default=None)
    args = parser.parse_args()
    X_train, y_train, X_test = utils.get_data()
    PARAMS = {
        'max_features': 'log2', 'criterion': 'gini',
        'n_estimators': 1900, 'max_depth': 64,
    }
    model = ensemble.RandomForestClassifier(
        **PARAMS,
        random_state=42,
        n_jobs=-1,
    )
    calibrate_model(model, X_train.values, y_train.values, path=args.logdir)
