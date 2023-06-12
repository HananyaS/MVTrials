import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from matplotlib import pyplot as plt

from sklearn.metrics import roc_auc_score

from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam


class RegModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, p=0.5, use_layer_norm=False):
        super(RegModel, self).__init__()

        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.decoder = torch.nn.Linear(hidden_dim, input_dim)

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.p = p

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(input_dim)

        self.double()

    @staticmethod
    def rand_bernoulli(x, p, gamma=100):
        uni_probs = torch.rand(x.shape[1])
        mask = 1 / (1 + torch.exp(-gamma * (uni_probs - p)))
        return mask * x

    def forward(self, x, drop=True):
        if self.training and drop:
            x = self.rand_bernoulli(x, torch.ones(x.shape[1]) * self.p)
            # x[:, mask.flatten()] = 0

        x = self.fc1(x)

        if "layer_norm" in self.__dict__:
            x = self.layer_norm(x)

        x = self.relu(x)
        reconstruction = self.decoder(x)
        output = self.fc2(x)
        output = self.softmax(output)

        return output, reconstruction

    def predict(self, x):
        with torch.no_grad():
            return self(x)[0]

    def score(self, x, y, metric="accuracy"):
        if metric == "accuracy":
            with torch.no_grad():
                return (self.predict(x).argmax(dim=1) == y).float().mean().item()

        elif metric.lower() == "auc":
            with torch.no_grad():
                return roc_auc_score(y, self.predict(x)[:, 1].numpy())

        else:
            raise NotImplementedError

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, dataset_name: str, lr: float = 1e-2,
            n_epochs: int = 300, verbose: bool = True, early_stopping: int = 30,
            reg_type: str = "l1", alpha: float = 1, beta: float = 0, feats_weighting: bool = False):
        optimizer = Adam(self.parameters(), lr=lr)
        min_val_loss = np.inf
        best_model = None

        bce_criterion = nn.CrossEntropyLoss(reduction="sum")
        diff_criterion = nn.MSELoss(reduction="sum") if reg_type == "l2" else nn.L1Loss(reduction="sum")
        l2_criterion = nn.MSELoss(reduction="sum")

        train_acc, val_acc = [], []
        full_loss_train, partial_loss_train = [], []
        full_loss_val, partial_loss_val = [], []

        losses_weight_feats = torch.ones(train_loader.dataset.X.shape[1])
        losses_weight_feats = losses_weight_feats / losses_weight_feats.sum()
        losses_weight_feats.requires_grad = False

        for epoch in tqdm(range(n_epochs)):
            epoch_loss_full_train, epoch_loss_partial_train = [], []
            epoch_loss_full_val, epoch_loss_partial_val = [], []

            for i, (X, y) in enumerate(train_loader):
                optimizer.zero_grad()

                y_pred_full, _ = self(X, drop=False)
                loss_full = bce_criterion(y_pred_full, y)

                diffs_partial = torch.zeros(X.shape[1])
                reconstruction_partial = torch.zeros(X.shape[1])

                for f in range(X.shape[1]):
                    X_f = X.clone()
                    X_f[:, f] = 0
                    y_pred_f, reconstruction_f = self(X_f, drop=False)

                    diffs_partial[f] = diff_criterion(y_pred_f, y_pred_full) * losses_weight_feats[f].item()
                    reconstruction_partial[f] = l2_criterion(reconstruction_f[:, f], X[:, f]) * losses_weight_feats[
                        f].item()

                    # if f == 0:
                    #     loss_partial = mse_criterion(y_pred_f, y_pred_full)
                    # else:
                    #     loss_partial += mse_criterion(y_pred_f, y_pred_full)

                if reg_type == 'max':
                    loss_partial = diffs_partial.max()

                else:
                    loss_partial = diffs_partial.mean()

                loss_rec = reconstruction_partial.mean()

                loss = loss_full.float() + alpha * loss_partial.float() + beta * loss_rec.float()
                epoch_loss_partial_train.append(alpha * loss_partial.item() + beta * loss_rec.item())

                epoch_loss_full_train.append(loss_full.item())

                loss.backward()
                optimizer.step()

                if feats_weighting:
                    losses_weight_feats = losses_weight_feats.detach() * (diffs_partial + reconstruction_partial)
                    losses_weight_feats = losses_weight_feats / losses_weight_feats.sum()

            full_loss_train.append(sum(epoch_loss_full_train) / len(train_loader.dataset))
            partial_loss_train.append(sum(epoch_loss_partial_train) / len(train_loader.dataset))

            train_X, train_y = train_loader.dataset.X, train_loader.dataset.y
            val_X, val_y = val_loader.dataset.X, val_loader.dataset.y

            with torch.no_grad():
                for i, (X, y) in enumerate(val_loader):
                    y_pred_full, _ = self(X, drop=False)
                    loss_full = bce_criterion(y_pred_full, y)

                    diffs_partial = torch.zeros(X.shape[1])
                    reconstruction_partial = torch.zeros(X.shape[1])

                    for f in range(X.shape[1]):
                        X_f = X.clone()
                        X_f[:, f] = 0
                        y_pred_f, reconstruction_f = self(X_f, drop=False)

                        diffs_partial[f] = diff_criterion(y_pred_f, y_pred_full)
                        reconstruction_partial[f] = l2_criterion(reconstruction_f[f], X[f])

                    if reg_type == 'max':
                        loss_partial = diffs_partial.max()

                    else:
                        loss_partial = diffs_partial.mean()

                    loss_rec = reconstruction_partial.mean()

                    epoch_loss_partial_val.append(alpha * loss_partial.item() + beta * loss_rec.item())

                    epoch_loss_full_val.append(loss_full.item())

                full_loss_val.append(sum(epoch_loss_full_val) / len(val_loader.dataset))

                partial_loss_val.append(sum(epoch_loss_partial_val) / len(val_loader.dataset))
                last_loss_val = full_loss_val[-1] + partial_loss_val[-1]

                if last_loss_val < min_val_loss:
                    min_val_loss = last_loss_val
                    del best_model
                    best_model = deepcopy(self)
                    epochs_no_improve = 0

                else:
                    epochs_no_improve += 1

                    if epochs_no_improve == early_stopping:
                        print(f"Early stopping at epoch {epoch + 1}")
                        self.load_state_dict(best_model.state_dict())
                        break

            train_acc.append(self.score(train_X, train_y))
            val_acc.append(self.score(val_X, val_y))

            if verbose:
                print(f"Epoch {epoch + 1}:")
                print(f"\tTrain Acc: {train_acc[-1]:.3f}")
                print(f"\tTest Acc: {val_acc[-1]:.3f}")

        # plt.plot(list(range(1, 1 + n_epochs)), train_acc, label="Train Acc")
        # plt.plot(list(range(1, 1 + n_epochs)), test_acc, label="Test Acc")
        # plt.xlabel("Epoch")
        # plt.ylabel("Accuracy")
        # plt.title("Accuracy")
        # plt.legend()
        # plt.show()

        plt.plot(list(range(1, 1 + len(full_loss_train))), full_loss_train, label="Full Loss Train")
        plt.plot(list(range(1, 1 + len(full_loss_val))), full_loss_val, label="Full Loss Val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Full Loss - {dataset_name}")
        plt.legend()
        # plt.show()

        plt.plot(list(range(1, 1 + len(partial_loss_train))), partial_loss_train, label="Partial Loss Train")
        plt.plot(list(range(1, 1 + len(partial_loss_val))), partial_loss_val, label="Partial Loss Val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Partial Loss - {dataset_name}")
        plt.legend()
        # plt.show
