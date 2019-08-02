"""
The main file for forecasting task.
Aug. 1 2019.
"""
import sys
import numpy as np
import pandas as pd

sys.path.append("../")
from util import data_proc
from util import data_proc_reg
from util import features


if __name__ == "__main__":
    df_train, df_test = data_proc_reg.prepare_data()
    X_train, y_train = data_proc_reg.gen_slp_patient(
        df_train, include_label=True, min_visit=5)
    # Only consider patients in the test set.
    sample_submission = pd.read_csv("../data/sample_submission_PANSS.csv", header=0)
    valid_patients = list(sample_submission["PatientID"])
    select = [x in valid_patients for x in df_test["PatientID"]]
    df_test = df_test[select]
    X_test = data_proc_reg.gen_slp_patient(
        df_test, include_label=False)
    print("X_train @{}, y_train @{}, X_test @{}".format(X_train.shape, y_train.shape, X_test.shape))
