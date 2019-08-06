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
    emperical_ratio = []
    visit_days = []
    for t in set(df.VisitDay):
        subset = df[df.VisitDay == t]
        emperical_ratio.append(subset.Alert.mean())
        visit_days.append(t)
    df2 = pd.DataFrame({"Ratio": emperical_ratio, "VisitDay": visit_days})
    sns.lmplot(
        x="VisitDay", y="Ratio",
        size=4, aspect=2,
        data=df2, lowess=True)
    plt.savefig(PATH + "alert_ratio_days.png", dpi=300)
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
    # Select treatment group
    TREATMENT = df["TxGroup"] == "Treatment"
    sns.distplot(df["VisitDay"], kde=False)
    plt.savefig(PATH + "dist_visit_day_all.png", dpi=300)
    LESS95 = df["VisitDay"] <= df["VisitDay"].quantile(0.95)
    # Drop the top 5% observations.
    df = df[LESS95]
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
    for target in ["PANSS_Total", "P_Total", "N_Total", "G_Total"]:
        sns.lmplot(
            x="VisitDay", y=target, hue="TxGroup",
            data=df, ci=0.95,
            lowess=True, markers=["o", "x"],
            scatter_kws={"alpha": 0.3},
            line_kws={"alpha": 0.8}
        )
        plt.savefig("{}lwlm_te_{}.png".format(PATH, target), dpi=300)

    # Plot country distribution
    plt.close()
    plt.figure(figsize=(20, 10))
    g = sns.catplot(x="Country", kind="count", height=4, aspect=2, data=df)
    g.set_xticklabels(rotation=70)
    plt.savefig(PATH + "dist_country.png", dpi=300)
    df.to_csv(path_or_buf="./Study_A_to_E_95.csv", index=False)
