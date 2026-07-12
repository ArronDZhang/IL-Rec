"""Discriminator utilities for ILRec adversarial inverse RL."""

from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _as_hidden_sizes(hidden_sizes: Iterable[int]) -> Sequence[int]:
    sizes = tuple(int(size) for size in hidden_sizes)
    if any(size <= 0 for size in sizes):
        raise ValueError("hidden_sizes must contain only positive integers.")
    return sizes


class TransitionDiscriminator(nn.Module):
    """Estimate whether a state-action transition came from policy data.

    The output is ``D(s, a)`` from the ILRec plan: policy transitions are labeled
    as ``1`` and expert demonstration transitions are labeled as ``0``.
    """

    def __init__(self, input_dim, hidden_sizes=(128, 128), dropout=0.0):
        super().__init__()
        input_dim = int(input_dim)
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1).")

        layers = []
        last_dim = input_dim
        for hidden_dim in _as_hidden_sizes(hidden_sizes):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward_logits(self, state_action_features):
        features = torch.as_tensor(state_action_features)
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.ndim != 2:
            raise ValueError("state_action_features must be a 2D tensor.")
        return self.net(features.float()).squeeze(-1)

    def forward(self, state_action_features):
        return torch.sigmoid(self.forward_logits(state_action_features))

    def bce_loss(self, state_action_features, labels):
        logits = self.forward_logits(state_action_features)
        labels = torch.as_tensor(labels, dtype=logits.dtype, device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, labels)


def discriminator_bce_loss(probabilities, labels):
    probabilities = torch.as_tensor(probabilities, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=probabilities.dtype, device=probabilities.device)
    return F.binary_cross_entropy(probabilities, labels)


def discriminator_irl_reward(probabilities, eps=1e-8):
    probabilities = torch.as_tensor(probabilities, dtype=torch.float32)
    clamped = probabilities.clamp(min=float(eps))
    return -torch.log(clamped)
