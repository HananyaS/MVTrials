import os
import argparse
import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

from data.load_data import load_data, split_X_y
from regModel import RegModel as Model
from dataset import Dataset

warnings.filterwarnings("ignore")

PROJECT_DIR = "."
os.chdir(PROJECT_DIR)

parser = argparse.ArgumentParser()

str2bool = lambda x: (str(x).lower() == "true")

parser.add_argument("--dataset", type=str, default="Wifi")
parser.add_argument("--full", type=str2bool, default=True)

args = parser.parse_args()


def run_model_old(train_loader, val_loader, test_loader, p, dataset_name):
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Dataset:\t", dataset_name)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    model_with_reg = Model(train_loader.dataset.X.shape[1], int(train_loader.dataset.X.shape[1] * .75),
                           len(train_loader.dataset.y.unique()), p=p,
                           learn_p=False, use_reg=True)
    model_with_reg.fit(train_loader, val_loader, n_epochs=300, verbose=False, dataset_name=dataset_name)

    # model_without_reg = Model(train_loader.dataset.X.shape[1], int(train_loader.dataset.X.shape[1] * .75),
    #                           len(train_loader.dataset.y.unique()), p=p,
    #                           learn_p=False, use_reg=False)
    # model_without_reg.fit(train_loader, val_loader, n_epochs=300, verbose=False, dataset_name=dataset_name)
    #
    # xgb = XGBClassifier()
    # xgb.fit(train_loader.dataset.X, train_loader.dataset.y)

    test_X = test_loader.dataset.X
    test_y = test_loader.dataset.y

    model_with_reg_score_full = model_with_reg.score(test_X, test_y)
    # model_without_reg_score_full = model_without_reg.score(test_X, test_y)
    # xgb_score_full = xgb.score(test_X, test_y)

    print(
        f"All features:\n"
        f"\tModel With Regularization Score:\t{model_with_reg_score_full}\n"
        # f"\tModel Without Regularization Score:\t{model_without_reg_score_full}\n"
        # f"\tXGB Score:\t{xgb_score_full}\n"
    )

    # new_test_X = test_X.clone()
    # new_test_X[:, [0, 1]] = 0
    # print("Without features 0 and 1: ", model.score(new_test_X, test_y))
    #

    model_with_reg_scores_partial = []
    # model_without_reg_scores_partial = []
    # xgb_scores_partial_mean = []
    # xgb_scores_partial_nan = []

    for j in range(test_X.shape[1]):
        new_test_X = test_X.clone()
        new_test_X[:, j] = 0

        model_with_reg_score = model_with_reg.score(new_test_X, test_y)
        # model_without_reg_score = model_without_reg.score(new_test_X, test_y)
        # xgb_score_mean = xgb.score(new_test_X, test_y)
        #
        # new_test_X[:, j] = np.nan
        # xgb_score_nan = xgb.score(new_test_X, test_y)

        print(f"Without feature {j}:\n"
              f"\tModel With Regularization Score:\t{model_with_reg_score}\n"
              # f"\tModel Without Regularization Score:\t{model_without_reg_score}\n"
              # f"\tXGB Mean Score:\t{xgb_score_mean}\n"
              # f"\tXGB NaN Score:\t{xgb_score_nan}\n"
              )

        model_with_reg_scores_partial.append(model_with_reg_score)
        # model_without_reg_scores_partial.append(model_without_reg_score)
        # xgb_scores_partial_mean.append(xgb_score_mean)
        # xgb_scores_partial_nan.append(xgb_score_nan)

    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print(f"Final probs: {model.probs}")

    to_save = {
        # "model_without_regularization_all_features": model_without_reg_score_full,
        # "xgb_all_features": xgb_score_full,
        "model_with_regularization_all_features": model_with_reg_score_full,
        # "model_without_regularization_partial": f"{np.mean(model_without_reg_scores_partial).round(3)} +- {np.std(model_without_reg_scores_partial).round(3)}",
        # "xgb_partial_features_mean": f"{np.mean(xgb_scores_partial_mean).round(3)} +- {np.std(xgb_scores_partial_mean).round(3)}",
        # "xgb_partial_features_nan": f"{np.mean(xgb_scores_partial_nan).round(3)} +- {np.std(xgb_scores_partial_nan).round(3)}",
        "model_with_regularization_partial": f"{np.mean(model_with_reg_scores_partial).round(3)} +- {np.std(model_with_reg_scores_partial).round(3)}"
        # "xgb_all_features": xgb_score_full,
        # "model_partial_features": f"{np.mean(model_scores_partial)} +- {np.std(model_scores_partial)}",
        # "xgb_partial_features": f"{np.mean(xgb_scores_partial)} +- {np.std(xgb_scores_partial)}",
        # "diffs_model_xgb_partial": f"{np.mean(diffs)} +- {np.std(diffs)}"
    }

    if not os.path.isfile("model_results.csv"):
        f = open("model_results.csv", "a")
        # f.write(
        #     "dataset,model_all_features,xgb_all_features,model_partial_features,xgb_partial_features,diffs_model_xgb_partial\n")
        f.write(','.join(["dataset", *list(to_save.keys())]) + "\n")
        f.close()

    with open("model_results.csv", "a") as f:
        # f.write(
        #     f"{dataset_name},{to_save['model_all_features']},{to_save['xgb_all_features']},{to_save['model_partial_features']},{to_save['xgb_partial_features']},{to_save['diffs_model_xgb_partial']}\n")
        f.write(','.join([dataset_name, *[str(v) for v in to_save.values()]]) + "\n")

    return to_save


def run_model_different_reg(train_loader, val_loader, test_loader, p, dataset_name):
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Dataset:\t", dataset_name)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    model_1l = Model(train_loader.dataset.X.shape[1], int(train_loader.dataset.X.shape[1] * .75),
                     len(train_loader.dataset.y.unique()), p=p,
                     learn_p=False, use_reg=True)
    model_1l.fit(train_loader, val_loader, n_epochs=300, verbose=False, dataset_name=dataset_name, reg_type="l1")

    model_max = Model(train_loader.dataset.X.shape[1], int(train_loader.dataset.X.shape[1] * .75),
                      len(train_loader.dataset.y.unique()), p=p,
                      learn_p=False, use_reg=False)
    model_max.fit(train_loader, val_loader, n_epochs=300, verbose=False, dataset_name=dataset_name, reg_type="max")

    model_l2 = Model(train_loader.dataset.X.shape[1], int(train_loader.dataset.X.shape[1] * .75),
                     len(train_loader.dataset.y.unique()), p=p,
                     learn_p=False, use_reg=True)

    model_l2.fit(train_loader, val_loader, n_epochs=300, verbose=False, dataset_name=dataset_name, reg_type="l2")

    test_X = test_loader.dataset.X
    test_y = test_loader.dataset.y

    model_l1_score_full = model_1l.score(test_X, test_y)
    model_max_score_full = model_max.score(test_X, test_y)
    model_l2_score_full = model_l2.score(test_X, test_y)

    print(
        f"All features:\n"
        f"\tModel L1:\t{model_l1_score_full}\n"
        f"\tModel Max:\t{model_max_score_full}\n"
        f"\tModel L2:\t{model_l2_score_full}\n"
    )

    # new_test_X = test_X.clone()
    # new_test_X[:, [0, 1]] = 0
    # print("Without features 0 and 1: ", model.score(new_test_X, test_y))
    #

    model_l1_scores_partial = []
    model_max_scores_partial = []
    model_l2_scores_partial = []

    for j in range(test_X.shape[1]):
        new_test_X = test_X.clone()
        new_test_X[:, j] = 0

        model_l1_score = model_1l.score(new_test_X, test_y)
        model_max_score = model_max.score(new_test_X, test_y)
        model_l2_score = model_l2.score(new_test_X, test_y)

        print(f"Without feature {j}:\n"
              f"\tModel L1 Score:\t{model_l1_score}\n"
              f"\tModel Max Score:\t{model_max_score}\n"
              f"\tModel L2 Score:\t{model_l2_score}\n"
              )

        model_l1_scores_partial.append(model_l1_score)
        model_l2_scores_partial.append(model_max_score)
        model_max_scores_partial.append(model_l2_score)

    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print(f"Final probs: {model.probs}")

    to_save = {
        "model_l1_all_features": model_l1_score_full,
        "model_max_all_features": model_max_score_full,
        "model_l2_all_features": model_l2_score_full,
        "model_l1_partial": f"{np.mean(model_l1_scores_partial).round(3)} +- {np.std(model_l1_scores_partial).round(3)}",
        "model_max_partial": f"{np.mean(model_max_scores_partial).round(3)} +- {np.std(model_max_scores_partial).round(3)}",
        "model_l2_partial": f"{np.mean(model_l2_scores_partial).round(3)} +- {np.std(model_l2_scores_partial).round(3)}",
    }

    filename = "model_results_different_reg_types.csv"

    if not os.path.isfile(filename):
        f = open(filename, "a")
        # f.write(
        #     "dataset,model_all_features,xgb_all_features,model_partial_features,xgb_partial_features,diffs_model_xgb_partial\n")
        f.write(','.join(["dataset", *list(to_save.keys())]) + "\n")
        f.close()

    with open(filename, "a") as f:
        # f.write(
        #     f"{dataset_name},{to_save['model_all_features']},{to_save['xgb_all_features']},{to_save['model_partial_features']},{to_save['xgb_partial_features']},{to_save['diffs_model_xgb_partial']}\n")
        f.write(','.join([dataset_name, *[str(v) for v in to_save.values()]]) + "\n")

    return to_save


def preprocess_dfcn(n_feats2remain):
    train = pd.read_csv("dfcn_data/raw/training.csv", index_col=0)
    test = pd.read_csv("dfcn_data/raw/test.csv", index_col=0)

    train.drop(columns=["our_cxr_score"], inplace=True)
    test.drop(columns=["our_cxr_score"], inplace=True)

    test.rename(columns={"Gender": "Sex"}, inplace=True)

    target_col = "PCR Result"

    # replace the order of the first and second columns in the test set

    tmp = test.iloc[:, 0].copy()
    test.iloc[:, 0] = test.iloc[:, 1]
    test.iloc[:, 1] = tmp

    col_1 = test.columns[0]
    test.rename(columns={test.columns[0]: test.columns[1], test.columns[1]: col_1}, inplace=True)

    train_idx = range(train.shape[0])
    test_idx = range(train.shape[0], train.shape[0] + test.shape[0])

    all_data = pd.concat([train, test], axis=0)

    # check which features are categorical or binary

    feats2onehot = list(
        filter(lambda x: (len(all_data[x].unique()) == 2 or all_data[x].dtype == "O") and x != target_col,
               all_data.columns))

    feats2normalize = list(filter(lambda x: x not in feats2onehot and x != target_col, all_data.columns))

    train = all_data.iloc[train_idx, :]
    test = all_data.iloc[test_idx, :]

    train_mean, train_std = train[feats2normalize].mean(), train[feats2normalize].std()

    train[feats2normalize] = (train[feats2normalize] - train_mean) / train_std
    test[feats2normalize] = (test[feats2normalize] - train_mean) / train_std

    train = pd.get_dummies(train, columns=feats2onehot)
    test = pd.get_dummies(test, columns=feats2onehot)

    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)

    train_X = train.drop(columns=[target_col])
    train_y = train[target_col]

    test_X = test.drop(columns=[target_col])
    test_y = test[target_col]

    if n_feats2remain > 0:
        # randomly sample features to remove
        feats2remove = set(list(train_X.columns)) - set(
            np.random.choice(list(set(list(train.columns)) - set(target_col)), n_feats2remain, replace=False))

        test_X.loc[:, feats2remove] = 0

    return train_X, train_y, test_X, test_y


def run_dfcn_data(save_res=True):
    all_scores = {}
    for i in range(1, 29):
        n_scores = []

        for j in range(3):
            train_X, train_y, test_X, test_y = preprocess_dfcn(n_feats2remain=i)

            # split to train val

            train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)

            train_ds = Dataset(train_X, train_y)
            val_ds = Dataset(val_X, val_y)
            test_ds = Dataset(test_X, test_y)

            train_loader = DataLoader(
                train_ds, batch_size=32, shuffle=True
            )

            val_loader = DataLoader(
                val_ds, batch_size=32, shuffle=True
            )

            # test_loader = DataLoader(
            #     test_ds, batch_size=32, shuffle=True
            # )

            # run_model(train_loader, val_loader, test_loader, 0, "dfcn_data")

            model = Model(train_loader.dataset.X.shape[1], int(train_loader.dataset.X.shape[1] * .75),
                          len(train_loader.dataset.y.unique()), p=0, use_layer_norm=True)
            model.fit(train_loader, val_loader, n_epochs=300, verbose=False, dataset_name="dfcn_data", beta=10)

            # calc score like in the paper

            test_X, test_y = test_ds.X, test_ds.y
            score = model.score(test_X, test_y, metric="auc")
            n_scores.append(score)

        all_scores[i] = (np.mean(n_scores), np.std(n_scores))
        print(f"Results for {i} features remaining: {round(all_scores[i][0], 3)} +- {round(all_scores[i][1], 3)}")

    if save_res:
        with open("dfcn_results.csv", "w") as f:
            f.write("n_feats,mean,std\n")

            for k, v in all_scores.items():
                f.write(f"{k},{v[0]},{v[1]}\n")


def check_alpha_beta():
    datasets = os.listdir("data/Tabular")
    datasets.remove("Sensorless")

    # datasets = ["Ecoli", "Accent", "Iris"]

    alphas = betas = [0, 0.5, 1, 5, 10]
    all_res = {}

    f = open("alpha_beta_results.csv", "w")
    f.write("dataset,alpha,beta,score\n")

    for dataset, alpha, beta in product(datasets, alphas, betas):
        data = load_data(dataset, args.full)

        if data[-1] == "tabular":
            train, val, test, target_col = data[:-1]
            edges = None

        else:
            train, val, test, edges, target_col = data[:-1]

        train_X, train_y = split_X_y(train, target_col)
        val_X, val_y = split_X_y(val, target_col)
        test_X, test_y = split_X_y(test, target_col)

        mean = train_X.mean()
        std = train_X.std()

        train_X = (train_X - mean) / std
        val_X = (val_X - mean) / std
        test_X = (test_X - mean) / std

        train_X = train_X.fillna(0)
        val_X = val_X.fillna(0)
        test_X = test_X.fillna(0)

        train_ds = Dataset(train_X, train_y)
        val_ds = Dataset(val_X, val_y)
        test_ds = Dataset(test_X, test_y)

        train_loader = DataLoader(
            train_ds, batch_size=32, shuffle=True
        )

        val_loader = DataLoader(
            val_ds, batch_size=32, shuffle=True
        )

        # test_loader = DataLoader(
        #     test_ds, batch_size=32, shuffle=True
        # )

        # run_model(train_loader, val_loader, test_loader, 0, dataset_name=dataset)
        # run_model_different_reg(train_loader, val_loader, test_loader, 0, dataset_name=dataset)
        model = Model(train_loader.dataset.X.shape[1], int(train_loader.dataset.X.shape[1] * .75),
                      len(train_loader.dataset.y.unique()), p=0,
                      learn_p=False, use_reg=True)

        model.fit(train_loader, val_loader, n_epochs=300, verbose=False, dataset_name=dataset, alpha=alpha, beta=beta)

        val_X, val_y = val_ds.X, val_ds.y
        val_scores = []

        for j in range(val_X.shape[1]):
            new_val_X = val_X.clone()
            new_val_X[:, j] = 0

            score = model.score(new_val_X, val_y)
            val_scores.append(score)

        print(
            f"Results for {dataset} with alpha={alpha} and beta={beta}:\t{round(np.mean(val_scores), 3)} +- {round(np.std(val_scores), 3)}")

        all_res[(dataset, alpha, beta)] = f"{round(np.mean(val_scores), 3)} +- {round(np.std(val_scores), 3)}"
        f.write(f"{dataset},{alpha},{beta},{score}\n")

    f.close()
