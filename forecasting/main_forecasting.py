"""
The main file for forecasting task.
Aug. 1 2019.
"""
import sys
import numpy as np
import pandas as pd
from typing import List

sys.path.append("../")
from util import data_proc
from util import data_proc_reg
from util import features
from util import grid_search_util
from forecasting import DNNRegressor


SCOPE = {
    "EPOCHS": 1000,
    "BATCH_SIZE": 32,
    "LR": [0.03, 0.01, 0.003, 0.01],
    "NEURONS": [
        [64, 128],
        [128, 256],
        [64, 128, 256],
        [128, 256, 512],
    ],
}

SCOPE = {
    "EPOCHS": 10,
    "BATCH_SIZE": 32,
    "LR": [0.03, 0.01],
    "NEURONS": [
        [64, 128],
    ],
}

PARAMS = dict()


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
    return df_train, df_test, X_train, y_train, X_test


def read_from_disk():
    X_train = pd.read_csv("./X_train.csv", header=0)
    y_train = pd.read_csv("./y_train.csv", header=0)
    X_test = pd.read_csv("./X_test.csv", header=0)
    return X_train, y_train, X_test


def forecasting_write_to_file(forecast: np.ndarray, path: str) -> None:
    """
    Writes the forecasting result to local file for submission.
    """
    holder = pd.read_csv("../data/sample_submission_PANSS.csv", header=0)
    assert len(holder) == len(forecast)
    holder["PANSS_Total"] = forecast
    holder.to_csv(path, index=False)
    print("File written to: {}".format(path))


if __name__ == "__main__":
    X_train, y_train, X_test = read_from_disk()

    # hist = DNNRegressor.main(
    #     X_train=X_train, y_train=y_train, X_test=X_test,
    #     EPOCHS=30, LR=0.01, NEURONS=[128, 256]
    # )

    do_grid_search = bool(int(input(
        "Task to perform on DNNClassifier: [0] Generate Prediction, [1] Perform Grid Search >>> "
    )))
    if do_grid_search:
        print("Perform grid search on hyper-parameters.")
        # Grid Search
        LOG_DIR = input("Dir to store the hparam tuning log: ")
        grid_search_util.grid_search(
            scope=SCOPE,
            data_feed=lambda: (X_train, y_train, X_test),
            train_main=DNNRegressor.main,
            log_dir=LOG_DIR
        )
    else:
        print("Generate prediction on test set...")
        pred = DNNRegressor.main(
            lambda: (X_train, y_train, X_test),
            forecast=True, tuning=False,
            **PARAMS
        )
        # Save predictions.
        holder = pd.read_csv("../data/sample_submission_status.csv", header=0)
        submission_name = input("File name to store submission: ")
        holder["LeadStatus"] = pred
        holder.to_csv("../submissions/{}".format(submission_name), index=False)
