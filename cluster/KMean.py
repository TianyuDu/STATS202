import sys
import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import cluster
from sklearn import preprocessing

sys.path.append("../")
import cluster.clustering_utility as utils


def k_mean(n_clusters: int):
    df = utils.get_data()  # This dataset 
    scaler = preprocessing.StandardScaler()
    standardized = scaler.fit_transform(
        df.drop(columns=["PANSS_Total"]).values)
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(standardized)
    index = kmeans.predict(standardized)
    df["P_Total"] = df[["P" + str(i) for i in range(1, 8)]].sum(axis=1)
    df["N_Total"] = df[["N" + str(i) for i in range(1, 8)]].sum(axis=1)
    df["G_Total"] = df[["G" + str(i) for i in range(1, 17)]].sum(axis=1)
    fig = plt.figure()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    var1, var2, var3 = "P_Total", "N_Total", "G_Total"
    for g in set(index):
        ax.scatter(
            df[index == g][var1],
            df[index == g][var2],
            df[index == g][var3],
            alpha=0.6
        )
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.set_zlabel(var3)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=2)
    args = parser.parse_args()
    k_mean(args.n)
