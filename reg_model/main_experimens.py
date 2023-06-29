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
from evaluations_plots import plot_res, load_xgb_res

warnings.filterwarnings("ignore")

PROJECT_DIR = "."
os.chdir(PROJECT_DIR)

parser = argparse.ArgumentParser()

str2bool = lambda x: (str(x).lower() == "true")

parser.add_argument("--dataset", type=str, default="Wifi")
parser.add_argument("--full", type=str2bool, default=True)
parser.add_argument("--verbose", type=str2bool, default=False)

parser.add_argument("--reg_type", type=str, default="var")
parser.add_argument("--weight_type", type=str, default="loss")
parser.add_argument("--alpha", type=float, default=0)
parser.add_argument("--use_layer_norm", type=str2bool, default=False)
parser.add_argument("--use_aug", type=str2bool, default=False)

parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--batch_size", type=int, default=32)

parser.add_argument("--n_epochs", type=int, default=300)
parser.add_argument("--early_stopping", type=int, default=30)

parser.add_argument("--resfile", type=str, default=None)

parser.add_argument("--run_grid", type=str2bool, default=False)

args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
XGB_FULL_RES, XGB_PARTIAL_RES = load_xgb_res()

assert args.reg_type in ["l1", "l2", "max", "var"]
assert args.weight_type in ["loss", "avg", "mult"]
assert args.resfile is None or args.resfile.endswith(".csv"), "resfile must be None or end with .csv"


# def run_model(train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, p: float, dataset_name: str,
#               save_res: bool = False, use_layer_norm: bool = True, feats_weighting: bool = True,
#               weight_type: str = 'avg', reg_type: str = 'var', verbose: bool = True, alpha: float = 1):

def run_model(train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, dataset_name: str,
              resfile: str = None, **kwargs):
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Dataset:\t", dataset_name)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    model = Model(train_loader.dataset.X.shape[1], int(train_loader.dataset.X.shape[1] * .75),
                  len(train_loader.dataset.y.unique()), use_layer_norm=kwargs.pop("use_layer_norm", True), )
    model.fit(train_loader, val_loader, dataset_name=dataset_name, **kwargs)

    test_X = test_loader.dataset.X
    test_y = test_loader.dataset.y

    model_score_full = model.score(test_X, test_y)

    print()

    scores_partial = []

    for j in range(test_X.shape[1]):
        new_test_X = test_X.clone()
        new_test_X[:, j] = 0

        score = model.score(new_test_X, test_y)
        scores_partial.append(score)

    if resfile is not None:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Results saved to:", resfile)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        # to_save = {
        #     "model_with_regularization_all_features": model_score_full,
        #     "model_with_regularization_partial": f"{np.mean(scores_partial).round(3)} +- {np.std(scores_partial).round(3)}"
        # }

        if not os.path.isfile(resfile):
            f = open(resfile, "a")
            f.write(','.join(["dataset", *list(kwargs.keys()), "full_score", "partial_score"]) + "\n")
            f.close()

        with open(resfile, "a") as f:
            # f.write(','.join([dataset_name, *[str(v) for v in to_save.values()]]) + "\n")
            f.write(','.join([dataset_name, *[str(v) for v in kwargs.values()], str(model_score_full),
                              f"{np.mean(scores_partial)} +- {np.std(scores_partial)}"]) + "\n")

        # return to_save

    return model_score_full, np.mean(scores_partial).round(3), np.std(scores_partial).round(3)


def get_split_data(dataset: str, use_aug: bool = False, norm: bool = True, as_loader: bool = True,
                   batch_size: int = 32):
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
        train_ds, batch_size=batch_size, shuffle=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=True
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=True
    )

    return train_loader, val_loader, test_loader


def main(resfile: str = "full_results.csv"):
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

    train_loader, val_loader, test_loader = get_split_data(dataset, use_aug=use_aug,
                                                           batch_size=kwargs.pop("batch_size"))

    full_score, partial_score_mean, partial_score_std = run_model(train_loader, val_loader, test_loader,
                                                                  dataset_name=dataset, **kwargs, verbose=verbose)

    # full_type1, partial_type1_mean, partial_type1_std = run_model(train_loader, val_loader, test_loader, 0,
    #                                                               dataset_name=dataset, **kwargs)

    # print(f"Full score:\t{full_type0} / {full_type1}")
    # print(f"Partial score:\t{partial_type0_mean} +- {partial_type0_std} / {partial_type1_mean} +- {partial_type1_std}")

    # if verbose:

    print(f"Full score:\t{full_score}")
    print(f"XGBoost full score:\t{XGB_FULL_RES[dataset]}")
    print(f"Partial score:\t{partial_score_mean} +- {partial_score_std}")
    print(f"XGBoost partial score:\t{XGB_PARTIAL_RES[dataset]}")

    return full_score, partial_score_mean, partial_score_std


# def run_many_datasets(datasets: list = None, save_res: bool = False, **kwargs):
def run_many_datasets(datasets: list = None, **kwargs):
    if datasets is None:
        datasets = os.listdir("data/Tabular")
        datasets.remove("Sensorless")
        datasets.remove("Credit")
        datasets.remove("HTRU2")
        datasets.remove("Ecoli")

        # datasets = ["Banknote", "Parkinson"]

    all_res = {}

    for dataset in datasets:
        res = main_one_run(dataset, **kwargs)
        all_res[dataset] = res

    # if save_res:
    #     res_df = pd.DataFrame(all_res).T
    #     res_df.to_csv(f"fw_{kwargs['weight_type']}.csv")

    return all_res


def run_grid_search(dataset: str, search_space: dict, resfile: str, **kwargs):
    all_confs = list(product(*search_space.values()))

    for i, vals in enumerate(all_confs):
        kwargs.update({k: v for k, v in zip(search_space.keys(), vals)})
        main_one_run(dataset, resfile=resfile, **kwargs)
        print(f"Done conf {i + 1}/{len(all_confs)}!")


if __name__ == '__main__':
    if args.run_grid:
        assert args.reg_type in ["l1", "l2", "max", "var"]
        assert args.weight_type in ["loss", "avg", "mult"]

        search_space = {
            "reg_type": ["l1", "l2", "max", "var"],
            "weight_type": [None, "loss", "avg", "mult"],
            "alpha": [0, 0.5, 1, 2, 5],
            "use_layer_norm": [True, False],
            "use_aug": [True, False],
            "lr": [0.001, 0.01, 0.1],
            "batch_size": [32, 64],
        }

        # create a sample space with just one value per hyperparameter
        # search_space = {k: [v[0]] for k, v in search_space.items()}
        # search_space["reg_type"] = ["l1", "l2"]

        run_grid_search(args.dataset, search_space, args.resfile, n_epochs=args.n_epochs)

    else:
        res_avg = run_many_datasets(weight_type=args.weight_type, verbose=args.verbose, reg_type=args.reg_type,
                                    alpha=args.alpha, use_layer_norm=args.use_layer_norm, use_aug=args.use_aug,
                                    feats_weighting=args.weight_type is not None, lr=args.lr, n_epochs=args.n_epochs,
                                    batch_size=args.batch_size, resfile=args.resfile)

        # all_res = {k: [*res_avg[k], *res_loss[k]] for k in res_avg}
        # all_res = pd.DataFrame(all_res).T
        # all_res.columns = ["full_score_avg_fw", "partial_score_mean_avg_fw", "partial_score_std_avg_fw",
        #                    "full_score_loss_fw", "partial_score_mean_loss_fw", "partial_score_std_loss_fw"]
        # all_res.to_csv("all_res_different_fw.csv")
        #
        # full_scores_avg = [res_avg[k][0] for k in res_avg]
        # full_scores_loss = [res_loss[k][0] for k in res_loss]
        #
        # partial_scores_avg = [res_avg[k][1] for k in res_avg]
        # partial_scores_loss = [res_loss[k][1] for k in res_loss]
        #
        # full_scores = [*full_scores_avg, *full_scores_loss]
        # partial_scores = [*partial_scores_avg, *partial_scores_loss]
        # raw_datasets = [*res_avg.keys(), *res_loss.keys()]
        # fw = ["avg"] * len(res_avg) + ["loss"] * len(res_loss)
        # title_datasets = [f"{d} ({fw[i]})" for i, d in enumerate(raw_datasets)]

        # plot_res(runs_names=title_datasets, raw_datasets_names=raw_datasets, labels=fw, full_results=full_scores,
        #          partial_results=partial_scores, savefile="all_res_different_fw.png")
