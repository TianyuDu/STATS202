import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import model_selection
from sklearn import metrics

import sys
sys.path.append("../")
from forecasting.main_forecasting import get_data, read_from_disk


if __name__ == "__main__":
    X_train, y_train, X_test = read_from_disk()
    # X_train, X_dev, y_train, y_dev = model_selection.train_test_split(
    #     X_train, y_train, test_size=0.2, random_state=42)
    records = list()
    for gamma in [10 ** x for x in np.linspace(-2, 1, 5)]:
        for epsilon in [10 ** x for x in np.linspace(-3, 0, 5)]:
            clf = svm.SVR(kernel='rbf', C=100, gamma=gamma, epsilon=epsilon)
            scores = model_selection.cross_val_score(clf, X_train.values, y_train.values.reshape(-1,), cv=5)
            mean_score = np.mean(scores)
            print(mean_score)
            records.append({"gamma": gamma, "epsilon": epsilon, "mean_score": mean_score})
    records.sort(key=lambda x: - x["mean_score"])
    print("Best: ")
    print(records[0])


clf = svm.SVR(kernel='rbf', C=100, gamma=0.01, epsilon=0.001)
clf.fit(X_train.values, y_train.values)
dev_pred = clf.predict(X_dev.values)
dev_mse = metrics.mean_squared_error(y_dev.values, dev_pred)
print("Dev set MSE: {}".format(dev_mse))
X_train, y_train, X_test = read_from_disk()
clf.fit(X_train.values, y_train.values)
test_pred = clf.predict(X_test.fillna(0.0).values)