import os
import time
import argparse
import warnings

import torch
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from data.load_data import load_data, split_X_y, get_folds

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import seaborn as sns

warnings.filterwarnings("ignore")

PROJECT_DIR = "."
os.chdir(PROJECT_DIR)

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="Iris")

args = parser.parse_args()


def plot_prob(X, y):
    mean, std = X.mean(axis=0), X.std(axis=0)
    X = (X - mean) / std
    #
    # X = np.nan_to_num(X, copy=False)
    #
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X)
    #
    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.RdYlBu)
    # plt.title("PCA")
    # plt.show()
    #
    # tsne = TSNE(n_components=2)
    # X_tsne = tsne.fit_transform(X)
    #
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=plt.cm.RdYlBu)
    # plt.title("t-SNE")
    # plt.show()

    # plot the distribution of each feature and color by class

    num_classes = len(np.unique(y))

    for i in range(X.shape[1]):
        feature = X.iloc[:, i]
        feature = feature[~np.isnan(feature)]

        # plot the 3 moments of the first feature

        plt.figure(figsize=(10, 5))

        for k in range(3):
            plt.subplot(1, 3, k + 1)

            for j in range(num_classes):
                feature_class = feature[y == j] ** (k + 1)

                sns.distplot(feature_class, hist=False, kde=True,
                             kde_kws={'linewidth': 3},
                             label="Class {}".format(j),
                             norm_hist=True)

            plt.title("Feature {}^{}".format(i, k + 1))
            plt.xlabel("Feature value")
            plt.ylabel("Density")
            plt.legend()

        plt.show()

        # plt.legend()
        # plt.title("Feature {}".format(i))
        # plt.show()


if __name__ == "__main__":
    dataset = args.dataset
    data = load_data(args.dataset)

    if data[-1] == "tabular":
        train, val, test, target_col = data[:-1]
        edges = None

    else:
        train, val, test, edges, target_col = data[:-1]

    train_X, train_y = split_X_y(train, target_col)
    val_X, val_y = split_X_y(val, target_col)
    test_X, test_y = split_X_y(test, target_col)

    plot_prob(train_X, train_y)
