import os
import argparse
import warnings

import numpy as np

from torch.utils.data import DataLoader

from data.load_data import load_data, split_X_y
from regModel import RegModel as Model
from dataset import Dataset

from itertools import product

warnings.filterwarnings("ignore")

PROJECT_DIR = "."
os.chdir(PROJECT_DIR)

parser = argparse.ArgumentParser()

str2bool = lambda x: (str(x).lower() == "true")

parser.add_argument("--dataset", type=str, default="Wifi")
parser.add_argument("--full", type=str2bool, default=True)

args = parser.parse_args()


def run_model(train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, p: float, dataset_name: str,
              save_res: bool = False, use_layer_norm: bool = True, feats_weighting: bool = True):
    assert p is None or 0 <= p < 1

    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Dataset:\t", dataset_name)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    model = Model(train_loader.dataset.X.shape[1], int(train_loader.dataset.X.shape[1] * .75),
                  len(train_loader.dataset.y.unique()), p=p, use_layer_norm=use_layer_norm)
    model.fit(train_loader, val_loader, n_epochs=300, verbose=False, dataset_name=dataset_name,
              feats_weighting=feats_weighting)

    test_X = test_loader.dataset.X
    test_y = test_loader.dataset.y

    model_score_full = model.score(test_X, test_y)

    print()
    print(
        f"All features score:\t{model_score_full}"
    )

    scores_partial = []

    for j in range(test_X.shape[1]):
        new_test_X = test_X.clone()
        new_test_X[:, j] = 0

        score = model.score(new_test_X, test_y)
        scores_partial.append(score)

    print(
        f"Partial score:\t{np.mean(scores_partial).round(3)} +- {np.std(scores_partial).round(3)}")

    if save_res:
        to_save = {
            "model_with_regularization_all_features": model_score_full,
            "model_with_regularization_partial": f"{np.mean(scores_partial).round(3)} +- {np.std(scores_partial).round(3)}"
        }

        if not os.path.isfile("model_results.csv"):
            f = open("model_results.csv", "a")
            f.write(','.join(["dataset", *list(to_save.keys())]) + "\n")
            f.close()

        with open("model_results.csv", "a") as f:
            f.write(','.join([dataset_name, *[str(v) for v in to_save.values()]]) + "\n")

        return to_save

    return model_score_full, np.mean(scores_partial).round(3), np.std(scores_partial).round(3)


def main(resfile: str = "all_results.csv"):
    datasets = os.listdir("../data/Tabular")
    datasets.remove("Sensorless")

    # datasets = ["Ecoli", "Accent", "Iris"]
    datasets = ["Iris"]

    res = []

    if not os.path.isfile(resfile):
        if os.path.dirname(resfile) != "":
            os.makedirs(os.path.dirname(resfile), exist_ok=True)
        f = open(resfile, "w")
        f.write(','.join(
            ["dataset", "use_aug", "use_layer_norm", "feats_weighting", "score_full_mean", "score_partial_mean",
             "score_partial_std"]) + "\n")

    else:
        f = open(resfile, "a")

    for dataset, use_aug, use_layer_norm, feats_weighting in product(datasets, [True, False], [True, False],
                                                                     [True, False]):
        data = load_data(dataset, args.full)

        if data[-1] == "tabular":
            train, val, test, target_col = data[:-1]
            edges = None

        else:
            train, val, test, edges, target_col = data[:-1]

        train_X, train_y = split_X_y(train, target_col)
        val_X, val_y = split_X_y(val, target_col)
        test_X, test_y = split_X_y(test, target_col)

        # mean = train_X.mean()
        # std = train_X.std()
        #
        # train_X = (train_X - mean) / std
        # val_X = (val_X - mean) / std
        # test_X = (test_X - mean) / std
        #
        # train_X = train_X.fillna(0)
        # val_X = val_X.fillna(0)
        # test_X = test_X.fillna(0)

        train_ds = Dataset(train_X, train_y, norm=False, add_aug=False)
        val_ds = Dataset(val_X, val_y, norm=False, add_aug=False)
        test_ds = Dataset(test_X, test_y, norm=False, add_aug=False)

        mean, std = train_ds.norm()
        val_ds.norm(mean, std)
        test_ds.norm(mean, std)

        if use_aug:
            train_ds.add_augmentations()

        train_loader = DataLoader(
            train_ds, batch_size=32, shuffle=True
        )

        val_loader = DataLoader(
            val_ds, batch_size=32, shuffle=True
        )

        test_loader = DataLoader(
            test_ds, batch_size=32, shuffle=True
        )

        score_full, score_partial_mean, score_partial_std = run_model(train_loader, val_loader, test_loader, 0,
                                                                      dataset_name=dataset,
                                                                      use_layer_norm=use_layer_norm,
                                                                      feats_weighting=feats_weighting)

        res.append(
            [dataset, use_aug, use_layer_norm, feats_weighting, score_full, score_partial_mean, score_partial_std])

        f.write(','.join([str(v) for v in res[-1]]) + "\n")


if __name__ == '__main__':
    main(resfile="all_results.csv")
