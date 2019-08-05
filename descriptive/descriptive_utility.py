"""
The data management utility functions for the first part of problem.
"""
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("../")
from util.data_proc import load_whole


if __name__ == "__main__":
    PATH = "../report/figures/"
    df = load_whole(path="../data/")
    print(df["TxGroup"].value_counts())
    # sns.catplot(x="TxGroup", kind="count", palette="ch:.25", data=df)
    # Clean data
    df["Treatment"] = (df["TxGroup"] == "Treatment").astype(int)
    # plt.show()
    # Generate summary statistic
    df["P_Total"] = df[["P{}".format(x) for x in range(1, 8)]].sum(axis=1)
    df["N_Total"] = df[["N{}".format(x) for x in range(1, 8)]].sum(axis=1)
    df["G_Total"] = df[["G{}".format(x) for x in range(1, 17)]].sum(axis=1)
    # Save the loaded data to local file
    df.to_csv(path_or_buf="./Study_A_to_E_all.csv", index=False)
    # Select treatment group
    TREATMENT = df["TxGroup"] == "Treatment"
    sns.lmplot(
        x="VisitDay", y="PANSS_Total", hue="TxGroup",
        data=df, ci=0.95,
        lowess=True, markers=["o", "x"],
        scatter_kws={"alpha": 0.3},
        line_kws={"alpha": 0.8}
    )
    plt.savefig("{}lwlm_te.png".format(PATH), dpi=300)
    # Initial Distribution of scores
    INITIAL = df["VisitDay"] == 0
    f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
    # d_map = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (0, 1)}
    for i, var in enumerate(["PANSS_Total", "P_Total", "N_Total", "G_Total"]):
        # loc = (i // 2, i % 2)
        sns.distplot(
            df[INITIAL & TREATMENT][var],
            color="red", ax=axes[i // 2, i % 2],
            kde_kws={"alpha": 0.3},
            hist_kws={"alpha": 0.5}
        )
        sns.distplot(
            df[INITIAL & ~ TREATMENT][var],
            color="skyblue", ax=axes[i // 2, i % 2],
            kde_kws={"alpha": 0.3},
            hist_kws={"alpha": 0.5}
        )
    f.legend(labels=['Treatment', 'Control'])
    plt.savefig("{}dist_initial_scores.png".format(PATH), dpi=300)
