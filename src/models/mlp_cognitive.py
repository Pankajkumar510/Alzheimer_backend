from typing import Iterable, List

import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class CognitiveMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: Iterable[int] = (32, 16), dropout: float = 0.1):
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def fit_scaler(X):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def save_scaler(scaler: StandardScaler, path: str):
    joblib.dump(scaler, path)


def load_scaler(path: str) -> StandardScaler:
    return joblib.load(path)
