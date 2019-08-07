import sys
import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import cluster
from sklearn import preprocessing
from sklearn import decomposition

sys.path.append("../")
import cluster.clustering_utility as utils


def color_pca(n_clusters: int, n_components: int, path: str = None):
    """
    This method plots the clustering result from K-mean to the 
    """
    print("Clustering into {} groups...".format(n_clusters))
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
    # index = cluster.AgglomerativeClustering(
    #     n_clusters=n_clusters, linkage="average").fit_predict(standardized)
    print("Computing PCA...")
    df = utils.get_data()
    df.drop(columns=["PANSS_Total"], inplace=True)
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(df.values)
    reduced = pca.transform(df.values)

    fig = plt.figure()
    if n_components == 2:
        for g in set(index):
            plt.scatter(
                reduced[index == g][:, 0],
                reduced[index == g][:, 1],
                alpha=0.6
            )
            plt.xlabel("First Principle Component")
            plt.ylabel("Second Principle Component")
    elif n_components == 3:
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        for g in set(index):
            ax.scatter(
                reduced[index == g][:, 0],
                reduced[index == g][:, 1],
                reduced[index == g][:, 2],
                alpha=0.6
            )
        ax.set_xlabel("First Principle Component")
        ax.set_ylabel("Second Principle Component")
        ax.set_zlabel("Third Principle Component")
    if path is None:
        plt.show()
    else:
        dest = path + "{}_PCA_{}_Clusters.png".format(n_components, n_clusters)
        plt.savefig(dest, dpi=300)
        print("Clustering visualization saved to: {}".format(dest))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", type=int, default=2)
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--logdir", type=str, default=None)
    args = parser.parse_args()
    color_pca(args.clusters, args.components, args.logdir)
