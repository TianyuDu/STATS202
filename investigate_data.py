"""
Data investigate script
"""
import numpy as np
import pandas as pd
import data_proc

if __name__ == "__main__":
    df_train = data_proc.load_whole(path="./data/")
    print(df_train.shape)
    df_test = pd.read_csv("./data/Study_E.csv", header=0)
    print(df_test.shape)
    K = set(df_train["VisitDay"])
    print(K)
    np.mean([
        val in K
        for val in df_test["VisitDay"]
        ])
