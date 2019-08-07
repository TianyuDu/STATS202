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


def k_mean(n_clusters: int, path: str = None):
    df = utils.get_data()  # This dataset contains 30 PANSS sub-scores and PANSS_Total
    # K-Mean is sensitive to scale of data.
    # Eventhough the prior ranges of all sub-scores are the same, but they have different
    # empericial ranges and variance.
    scaler = preprocessing.StandardScaler()
    standardized = scaler.fit_transform(
        df.drop(columns=["PANSS_Total"]).values)
    # Create K mean.
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=0)
    # Fit KMeans using 30 standardized sub-scores only.
    kmeans.fit(standardized)
    index = kmeans.predict(standardized)
    # Create summary metrics for better visualization.
    df["P_Total"] = df[["P" + str(i) for i in range(1, 8)]].sum(axis=1)
    df["N_Total"] = df[["N" + str(i) for i in range(1, 8)]].sum(axis=1)
    df["G_Total"] = df[["G" + str(i) for i in range(1, 17)]].sum(axis=1)
    # Plot out the clustering result
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
    if path is None:
        plt.show()
    else:
        dest = path + "{}_means.png".format(n_clusters)
        plt.savefig(dest, dpi=300)
        print("Clustering visualization saved to: {}".format(dest))


def gen_2d_plots(n_clusters: int, path: str = None):
    df = utils.get_data()  # This dataset contains 30 PANSS sub-scores and PANSS_Total
    # K-Mean is sensitive to scale of data.
    # Eventhough the prior ranges of all sub-scores are the same, but they have different
    # empericial ranges and variance.
    scaler = preprocessing.StandardScaler()
    standardized = scaler.fit_transform(
        df.drop(columns=["PANSS_Total"]).values)
    # Create K mean.
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=0)
    # Fit KMeans using 30 standardized sub-scores only.
    kmeans.fit(standardized)
    index = kmeans.predict(standardized)
    # Create summary metrics for better visualization.
    df["P_Total"] = df[["P" + str(i) for i in range(1, 8)]].sum(axis=1)
    df["N_Total"] = df[["N" + str(i) for i in range(1, 8)]].sum(axis=1)
    df["G_Total"] = df[["G" + str(i) for i in range(1, 17)]].sum(axis=1)
    # Plot out the clustering result
    for var1, var2 in [("P_Total", "N_Total"), ("P_Total", "G_Total"), ("N_Total", "G_Total")]:
        fig = plt.figure()
        for g in set(index):
            plt.scatter(
                df[index == g][var1],
                df[index == g][var2],
                alpha=0.3
            )
        dest = path + "{}_means_{}_{}.png".format(n_clusters, var1, var2)
        plt.savefig(dest, dpi=300)
        print("Clustering visualization saved to: {}".format(dest))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--reduced", type=bool, default=None)
    args = parser.parse_args()
    if args.reduced is None:
        k_mean(args.n, args.logdir)
    else:
        gen_2d_plots(args.n, args.logdir)
