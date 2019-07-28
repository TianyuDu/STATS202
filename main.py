"""
Main script.
"""
import numpy as np
import pandas as pd
import data_proc
from data_proc import gen_slp_assessment

if __name__ == "__main__":
    df_train = data_proc.load_whole(path="./data/")
    print(df_train.shape)
    df_test = pd.read_csv("./data/Study_E.csv", header=0)
    print(df_test.shape)
    X_train, y_train, FEATURE, PANSS = gen_slp_assessment(df_train)
    X_test = data_proc.parse_test_set(X_train, df_test)
