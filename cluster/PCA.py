"""
Use PCA to identify the number of clusters to use.
"""
import sys
import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition

sys.path.append("../")
import cluster.clustering_utility as utils


def pca_reduction(n_components: int, path: str = None):
    df = utils.get_data()
    df.drop(columns=["PANSS_Total"], inplace=True)
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(df.values)
    transformed = pca.transform(df.values)

    fig = plt.figure()
    if n_components == 2:
        plt.scatter(transformed[:, 0], transformed[:, 1], alpha=0.6)
        plt.xlabel("First Principle Component")
        plt.ylabel("Second Principle Component")
    elif n_components == 3:
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        ax.scatter(
            transformed[:, 0],
            transformed[:, 1],
            transformed[:, 2],
            alpha=0.6
        )
        ax.set_xlabel("First Principle Component")
        ax.set_ylabel("Second Principle Component")
        ax.set_zlabel("Third Principle Component")
    if path is None:
        plt.show()
    else:
        dest = path + "{}_PCA.png".format(n_components)
        plt.savefig(dest, dpi=300)
        print("PCA visualization saved to: {}".format(dest))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--logdir", type=str, default=None)
    args = parser.parse_args()
    pca_reduction(args.n, args.logdir)
