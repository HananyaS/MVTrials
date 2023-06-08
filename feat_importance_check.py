import os
import argparse
import warnings

import numpy as np

from data.load_data import load_data, split_X_y
import matplotlib.pyplot as plt

import xgboost as xgb
import seaborn

from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

PROJECT_DIR = "."
os.chdir(PROJECT_DIR)

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="Iris")

args = parser.parse_args()


if __name__ == "__main__":
    datasets = ["Iris", "Banknote", "Wifi", "Accent"]

    for dataset in datasets:
        data = load_data(dataset)

        if data[-1] == "tabular":
            train, val, test, target_col = data[:-1]
            edges = None

        else:
            train, val, test, edges, target_col = data[:-1]

        train_X, train_y = split_X_y(train, target_col)
        val_X, val_y = split_X_y(val, target_col)
        test_X, test_y = split_X_y(test, target_col)

        # calculate feature importance

        feat_importance = np.zeros((train_X.shape[1], test_X.shape[1]))
        train_X_ = train_X.copy()

        columns_remain = np.arange(train_X.shape[1])
        corr = train_X.corr()
        X, y = [], []
        prev_max_indices = []

        for i in range(train_X.shape[1]):
            model = xgb.XGBClassifier()
            model.fit(train_X_, train_y)
            feat_importance[columns_remain, i] = model.feature_importances_

            if i > 0:
                for j in set(range(train_X.shape[1])) - set(prev_max_indices):
                    X.append([feat_importance[j, i-1], corr.iloc[j, prev_max_indices[-1]]])
                    y.append(feat_importance[j, i])

            max_idx = np.argmax(model.feature_importances_)
            col2remove = train_X_.columns[max_idx]
            train_X_ = train_X_.drop(col2remove, axis=1)
            col_num = train_X.columns.get_loc(col2remove)

            columns_remain = np.delete(columns_remain, max_idx)
            prev_max_indices.append(col_num)

        X = np.array(X)
        y = np.array(y)

        # plot the feature importance with values

        plt.figure(figsize=(10, 10))
        seaborn.heatmap(feat_importance[prev_max_indices], annot=True, cmap="YlGnBu")
        plt.title(f"Feature Importance for {dataset}")
        plt.xlabel("Feature")
        plt.ylabel("Iteration")
        plt.show()

        feat_importance_pred_model = LinearRegression()
        # X = np.concatenate([X, y.reshape(-1, 1)], axis=1)
        feat_importance_pred_model.fit(X, y)

        # calc r2 score
        score = feat_importance_pred_model.score(X, y)

        print(f"Feature Importance Prediction Score for {dataset}:\t{feat_importance_pred_model.score(X, y)}")





