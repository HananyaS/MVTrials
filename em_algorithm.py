import numpy as np
from data.load_data import load_data, split_X_y

import torch
from torch.distributions import Cauchy
from torch import nn


class CauchyConvertorModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(CauchyConvertorModel, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes

        self.mu = nn.Parameter(torch.randn(num_classes, num_classes, num_features))
        self.gamma = nn.Parameter(torch.randn(num_classes, num_classes, num_features))

    def forward(self, x):
        x = x.unsqueeze(1)
        mu = self.mu.unsqueeze(0)
        gamma = self.gamma.unsqueeze(0)

        log_prob = sum(
            [Cauchy(mu[:, i], torch.exp(gamma[:, i])).log_prob(x).sum(dim=-1) for i in range(self.num_classes)]
        )

        return log_prob

    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

        for e in range(1000):
            log_prob = self.forward(X)
            loss = -log_prob[y].mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print("Epoch: {}, Loss: {}".format(e, loss.item()))

        return self.mu.detach().numpy(), self.gamma.detach().numpy()

    def predict(self, X):
        log_prob = self.forward(X)
        return log_prob.argmax(dim=-1).numpy()

    def score(self, X, y):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()

        log_prob = self.forward(X)
        return (log_prob.argmax(dim=-1) == y).float().mean().item()


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
