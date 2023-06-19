import os
import argparse
import warnings
from itertools import product

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from data.load_data import load_data, split_X_y
from regModel import RegModel as Model
from dataset import Dataset
from evaluations_plots import plot_res

warnings.filterwarnings("ignore")

PROJECT_DIR = "."
os.chdir(PROJECT_DIR)

parser = argparse.ArgumentParser()

str2bool = lambda x: (str(x).lower() == "true")

parser.add_argument("--dataset", type=str, default="Wifi")
parser.add_argument("--full", type=str2bool, default=True)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_model(train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, p: float, dataset_name: str,
              save_res: bool = False, use_layer_norm: bool = True, feats_weighting: bool = True,
              weight_type: str = 'avg'):
    assert p is None or 0 <= p < 1

    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Dataset:\t", dataset_name)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    model = Model(train_loader.dataset.X.shape[1], int(train_loader.dataset.X.shape[1] * .75),
                  len(train_loader.dataset.y.unique()), p=p, use_layer_norm=use_layer_norm)
    model.fit(train_loader, val_loader, n_epochs=300, verbose=False, dataset_name=dataset_name,
              feats_weighting=feats_weighting, weight_type=weight_type)

    test_X = test_loader.dataset.X
    test_y = test_loader.dataset.y

    model_score_full = model.score(test_X, test_y)

    print()
    # print(
    #     f"All features score:\t{model_score_full}"
    # )

    scores_partial = []

    for j in range(test_X.shape[1]):
        new_test_X = test_X.clone()
        new_test_X[:, j] = 0

        score = model.score(new_test_X, test_y)
        scores_partial.append(score)

    # print(
    #     f"Partial score:\t{np.mean(scores_partial).round(3)} +- {np.std(scores_partial).round(3)}")

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


def get_split_data(dataset: str, use_aug: bool = False, norm: bool = True, as_loader: bool = True):
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

    if norm:
        mean, std = train_ds.norm()
        val_ds.norm(mean, std)
        test_ds.norm(mean, std)

    if use_aug:
        train_ds.add_augmentations()

    if not as_loader:
        return train_ds, val_ds, test_ds

    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=True
    )

    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=True
    )

    return train_loader, val_loader, test_loader


def main(resfile: str = "all_results.csv"):
    datasets = os.listdir("data/Tabular")
    datasets.remove("Sensorless")
    datasets.remove("Credit")
    datasets.append("Credit")
    # runs_names.append("Sensorless")

    # runs_names = ["Ecoli", "Accent", "Iris"]

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
        if dataset.lower() in ["wine", "htru2", "parkinson"]:
            continue

        train_loader, val_loader, test_loader = get_split_data(dataset, use_aug)
        score_full, score_partial_mean, score_partial_std = run_model(train_loader, val_loader, test_loader, 0,
                                                                      dataset_name=dataset,
                                                                      use_layer_norm=use_layer_norm,
                                                                      feats_weighting=feats_weighting, )

        res.append(
            [dataset, use_aug, use_layer_norm, feats_weighting, score_full, score_partial_mean, score_partial_std])

        f.write(','.join([str(v) for v in res[-1]]) + "\n")


def main_one_run(dataset: str, verbose: bool = False, **kwargs):
    if "use_aug" in kwargs:
        use_aug = kwargs.pop("use_aug")
    else:
        use_aug = False

    train_loader, val_loader, test_loader = get_split_data(dataset, use_aug=use_aug)

    full_score, partial_score_mean, partial_score_std = run_model(train_loader, val_loader, test_loader, 0,
                                                                  dataset_name=dataset, **kwargs)

    # full_type1, partial_type1_mean, partial_type1_std = run_model(train_loader, val_loader, test_loader, 0,
    #                                                               dataset_name=dataset, **kwargs)

    # print(f"Full score:\t{full_type0} / {full_type1}")
    # print(f"Partial score:\t{partial_type0_mean} +- {partial_type0_std} / {partial_type1_mean} +- {partial_type1_std}")

    if verbose:
        print(f"Full score:\t{full_score}")
        print(f"Partial score:\t{partial_score_mean} +- {partial_score_std}")

    return full_score, partial_score_mean, partial_score_std


def run_many_datasets(datasets: list = None, save_res: bool = False, **kwargs):
    if datasets is None:
        datasets = os.listdir("data/Tabular")
        datasets.remove("Sensorless")
        datasets.remove("Credit")
        datasets.remove("HTRU2")
        datasets.remove("Ecoli")

        # datasets = ["Iris", "Accent", "Ecoli"]

    all_res = {}

    for dataset in datasets:
        res = main_one_run(dataset, **kwargs)
        all_res[dataset] = res

    if save_res:
        res_df = pd.DataFrame(all_res).T
        res_df.to_csv(f"fw_{kwargs['weight_type']}.csv")

    return all_res


if __name__ == '__main__':
    res_avg = run_many_datasets(weight_type='avg', verbose=True)
    res_loss = run_many_datasets(weight_type='loss', verbose=True)

    all_res = {k: [*res_avg[k], *res_loss[k]] for k in res_avg}
    all_res = pd.DataFrame(all_res).T
    all_res.columns = ["full_score_avg_fw", "partial_score_mean_avg_fw", "partial_score_std_avg_fw",
                       "full_score_loss_fw", "partial_score_mean_loss_fw", "partial_score_std_loss_fw"]
    all_res.to_csv("all_res_different_fw.csv")

    full_scores_avg = [res_avg[k][0] for k in res_avg]
    full_scores_loss = [res_loss[k][0] for k in res_loss]

    partial_scores_avg = [res_avg[k][1] for k in res_avg]
    partial_scores_loss = [res_loss[k][1] for k in res_loss]

    full_scores = [*full_scores_avg, *full_scores_loss]
    partial_scores = [*partial_scores_avg, *partial_scores_loss]
    raw_datasets = [*res_avg.keys(), *res_loss.keys()]
    fw = ["avg"] * len(res_avg) + ["loss"] * len(res_loss)
    title_datasets = [f"{d} ({fw[i]})" for i, d in enumerate(raw_datasets)]

    plot_res(runs_names=title_datasets, raw_datasets_names=raw_datasets, labels=fw, full_results=full_scores,
             partial_results=partial_scores, savefile="all_res_different_fw.png")
