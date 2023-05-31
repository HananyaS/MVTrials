import os
import time
import argparse
import warnings

import torch
import pandas as pd
import numpy as np

from data.load_data import load_data, split_X_y, get_folds

warnings.filterwarnings("ignore")

PROJECT_DIR = "."
os.chdir(PROJECT_DIR)

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="Iris")

args = parser.parse_args()


if __name__ == "__main__":
    data = load_data(args.dataset)

    if data[-1] == "tabular":
        train, val, test, target_col = data[:-1]
        edges = None

    else:
        train, val, test, edges, target_col = data[:-1]

    train_X, train_y = ...