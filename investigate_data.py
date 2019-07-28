"""
Data investigate script
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
    K = set(df_train["Country"])
    print(K)
    np.mean([
        val in K
        for val in df_test["Country"]
        ])
    X, y, FEATURE, PANSS = gen_slp_assessment(df_train)
    C = parse_test_set_countries(X, df_test, FEATURE)
