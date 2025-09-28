from typing import List

import numpy as np
import torch
import torch.nn as nn


def average_fusion(probabilities: List[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(probabilities, axis=0), axis=0)


def weighted_fusion(probabilities: List[np.ndarray], weights: List[float]) -> np.ndarray:
    w = np.array(weights, dtype=np.float32)
    w = w / (w.sum() + 1e-8)
    probs = np.stack(probabilities, axis=0)  # (M, C)
    return (probs * w[:, None]).sum(axis=0)


class MLPStacker(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)
