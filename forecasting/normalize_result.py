import numpy as np
import pandas as pd


def bootstrap(n: int, iter: int, y_train: pd.DataFrame) -> float:
    result = []
    for _ in range(iter):
        sample = y_train.sample(n)
        result.append(sample.std().values[0])
    return np.mean(result)


mean = np.mean(pred_test)
demean_pred_test = pred_test - mean
norm_pred_test = demean_pred_test / pred_test.std() * bootstrap(len(pred_test), 1000, y_train)
reform = norm_pred_test + mean

sns.distplot(y_train.sample(len(pred_test)))
sns.distplot(reform)
plt.show()

sns.distplot(pred_test)
sns.distplot(reform)
plt.show()
