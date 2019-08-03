import sys
import numpy as np
import pandas as pd
from typing import List

sys.path.append("../")
from util import data_proc
from util import data_proc_reg
from util import features
from util import grid_search_util


def get_data() -> List[pd.DataFrame]:
    df_train, df_test = data_proc_reg.prepare_data()
    X_train, y_train = data_proc_reg.gen_slp_patient(
        df_train, include_label=True, min_visit=5)
    # Only consider patients in the test set.
    sample_submission = pd.read_csv(
        "../data/sample_submission_PANSS.csv", header=0)
    valid_patients = list(sample_submission["PatientID"])
    select = [x in valid_patients for x in df_test["PatientID"]]
    df_test = df_test[select]
    X_test = data_proc_reg.gen_slp_patient(
        df_test, include_label=False)
    X_test.fillna(0.0, inplace=True)
    print("X_train @{}, y_train @{}, X_test @{}".format(
        X_train.shape, y_train.shape, X_test.shape))
    return X_train, y_train, X_test


def read_from_disk():
    X_train = pd.read_csv("./X_train.csv", header=0)
    y_train = pd.read_csv("./y_train.csv", header=0)
    X_test = pd.read_csv("./X_test.csv", header=0)
    return X_train, y_train, X_test


def generate_submission(forecast: np.ndarray, path: str) -> None:
    """
    Writes the forecasting result to local file for submission.
    """
    holder = pd.read_csv("../data/sample_submission_PANSS.csv", header=0)
    assert len(holder) == len(forecast)
    holder["PANSS_Total"] = forecast
    holder.to_csv(path, index=False)
    print("File written to: {}".format(path))
