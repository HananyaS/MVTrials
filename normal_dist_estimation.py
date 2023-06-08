import numpy as np
from data.load_data import load_data, split_X_y

import torch
from torch import nn


if __name__ == "__main__":
    data = load_data("Iris")

    if data[-1] == "tabular":
        train, val, test, target_col = data[:-1]
        edges = None

    else:
        train, val, test, edges, target_col = data[:-1]

    train_X, train_y = split_X_y(train, target_col)
    val_X, val_y = split_X_y(val, target_col)
    test_X, test_y = split_X_y(test, target_col)

    mean, std = train_X.mean(), train_X.std()
    train_X = (train_X - mean) / std

    train_X = np.nan_to_num(train_X, copy=False)
    train_y = train_y.values

    model = CauchyConvertorModel(train_X.shape[1], len(np.unique(train_y)))
    model.fit(train_X, train_y)

    train_score = model.score(train_X, train_y)
    print("Train score: {}".format(train_score))

    val_X = (val_X - mean) / std
    val_X = np.nan_to_num(val_X, copy=False)
    val_y = val_y.values

    val_score = model.score(val_X, val_y)
    print("Val score: {}".format(val_score))
