"""Dedicated demonstration value and advantage utilities for ILRec."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DemoValueFitResult:
    model: nn.Module
    initial_loss: float
    final_loss: float
    steps: int


class DemoValueNetwork(nn.Module):
    """Small value network used only for demonstration-return baselines."""

    def __init__(self, input_dim, hidden_size=32):
        super().__init__()
        input_dim = int(input_dim)
        hidden_size = int(hidden_size)
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features):
        features = torch.as_tensor(features, dtype=torch.float32)
        if features.ndim != 2:
            raise ValueError("features must be a 2D tensor.")
        return self.net(features).squeeze(-1)


class DemoQNetwork(DemoValueNetwork):
    """Small state-action Q network for demonstration-return fitting."""


def compute_demo_advantages(demo_returns, demo_values, demo_q_values=None):
    returns = torch.as_tensor(demo_returns, dtype=torch.float32)
    values = torch.as_tensor(demo_values, dtype=torch.float32, device=returns.device)
    q_values = returns if demo_q_values is None else torch.as_tensor(
        demo_q_values,
        dtype=torch.float32,
        device=returns.device,
    )
    if returns.shape != values.shape:
        raise ValueError("demo_returns and demo_values must have matching shapes.")
    if q_values.shape != values.shape:
        raise ValueError("demo_q_values and demo_values must have matching shapes.")
    return q_values - values


def _fit_demo_scalar_network(model_cls, features, demo_returns, hidden_size=32, lr=1e-2, steps=100):
    features = torch.as_tensor(features, dtype=torch.float32)
    returns = torch.as_tensor(demo_returns, dtype=torch.float32, device=features.device)
    if features.ndim != 2:
        raise ValueError("features must be a 2D tensor.")
    if returns.ndim != 1:
        raise ValueError("demo_returns must be a 1D tensor.")
    if features.shape[0] != returns.shape[0]:
        raise ValueError("features and demo_returns batch sizes must match.")

    model = model_cls(features.shape[1], hidden_size=hidden_size).to(features.device)
    with torch.no_grad():
        initial_loss = float(F.mse_loss(model(features), returns).detach().cpu().item())

    steps = max(0, int(steps))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    final_loss = initial_loss
    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.mse_loss(model(features), returns)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())

    return DemoValueFitResult(
        model=model,
        initial_loss=initial_loss,
        final_loss=final_loss,
        steps=steps,
    )


def fit_demo_value_network(features, demo_returns, hidden_size=32, lr=1e-2, steps=100):
    return _fit_demo_scalar_network(
        DemoValueNetwork,
        features,
        demo_returns,
        hidden_size=hidden_size,
        lr=lr,
        steps=steps,
    )


def fit_demo_q_network(features, demo_returns, hidden_size=32, lr=1e-2, steps=100):
    return _fit_demo_scalar_network(
        DemoQNetwork,
        features,
        demo_returns,
        hidden_size=hidden_size,
        lr=lr,
        steps=steps,
    )
