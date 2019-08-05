import matplotlib.pyplot as plt

from sklearn import calibration


if __name__ == "__main__":
    fop, mpv = calibration.calibration_curve(y_dev, pred_dev[:, 1], n_bins=100)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(mpv, fop, marker=".")
    plt.show()

    calibrator = calibration.CalibratedClassifierCV(model, method="isotonic", cv=5)
    calibrator.fit(X_train, y_train)
    pred_dev = calibrator.predict_proba(X_dev)
