import warnings
from typing import Union
import sys
import argparse
from datetime import datetime

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import log_loss

sys.path.append("../")
import classification.classification_utility as utils

if __name__ == "__main__":
    model = ...
    importances = model.feature_importances_
    std = np.std([
        tree.feature_importances_
        for tree in model.estimators_
    ], axis=0)
    indices = np.argsort(importances)[::-1]

    print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, X_train.columns[indices[f]], importances[indices[f]]))
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center",
        alpha=0.6)
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation="vertical")
plt.xlim([-1, X_train.shape[1]])
plt.show()


for var, imp in zip(X_train.columns, importances):
    print("{}: {}".format(var, imp))