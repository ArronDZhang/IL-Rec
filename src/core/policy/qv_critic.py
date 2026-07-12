"""Q/V critic helpers for the ILRec ILRec branch."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class QVCriticLoss:
    total_loss: torch.Tensor
    q_loss: torch.Tensor
    v_loss: torch.Tensor
    td_targets: torch.Tensor


class StateQVCritic(nn.Module):
    def __init__(self, input_dim, num_actions, hidden_size=64):
        super().__init__()
        input_dim = int(input_dim)
        num_actions = int(num_actions)
        hidden_size = int(hidden_size)
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if num_actions <= 0:
            raise ValueError("num_actions must be positive.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
        )
        self.q_head = nn.Linear(hidden_size, num_actions)
        self.v_head = nn.Linear(hidden_size, 1)

    def q_values(self, state_features):
        hidden = self.trunk(torch.as_tensor(state_features, dtype=torch.float32))
        return self.q_head(hidden)

    def values(self, state_features):
        hidden = self.trunk(torch.as_tensor(state_features, dtype=torch.float32))
        return self.v_head(hidden).squeeze(-1)

    def forward(self, state_features):
        hidden = self.trunk(torch.as_tensor(state_features, dtype=torch.float32))
        return self.q_head(hidden), self.v_head(hidden).squeeze(-1)


def compute_td_targets(rewards, next_values, dones, discount):
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    next_values = torch.as_tensor(next_values, dtype=torch.float32, device=rewards.device)
    dones = torch.as_tensor(dones, dtype=torch.bool, device=rewards.device)
    if rewards.shape != next_values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, next_values, and dones must have matching shapes.")
    return rewards + float(discount) * next_values * (~dones).float()


def weighted_critic_loss(predictions, targets, sample_weights=None):
    predictions = torch.as_tensor(predictions, dtype=torch.float32)
    targets = torch.as_tensor(targets, dtype=predictions.dtype, device=predictions.device)
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have matching shapes.")
    squared = (predictions - targets.detach()).pow(2)
    if sample_weights is None:
        return squared.mean()
    weights = torch.as_tensor(sample_weights, dtype=predictions.dtype, device=predictions.device)
    if weights.shape != predictions.shape:
        raise ValueError("sample_weights must match predictions shape.")
    return (squared * weights).mean()


def update_target_critic(target_critic, source_critic, tau=1.0):
    tau = float(tau)
    if tau <= 0.0 or tau > 1.0:
        raise ValueError("tau must be in (0, 1].")
    with torch.no_grad():
        for target_param, source_param in zip(target_critic.parameters(), source_critic.parameters()):
            target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)


def qv_critic_training_step(
    critic,
    target_critic,
    optimizer,
    state_features,
    actions,
    rewards,
    next_state_features,
    dones,
    discount,
    sample_weights=None,
):
    state_features = torch.as_tensor(state_features, dtype=torch.float32)
    next_state_features = torch.as_tensor(next_state_features, dtype=torch.float32, device=state_features.device)
    actions = torch.as_tensor(actions, dtype=torch.long, device=state_features.device)
    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=state_features.device)
    dones = torch.as_tensor(dones, dtype=torch.bool, device=state_features.device)
    if state_features.ndim != 2 or next_state_features.ndim != 2:
        raise ValueError("state_features and next_state_features must be 2D tensors.")
    if state_features.shape != next_state_features.shape:
        raise ValueError("state_features and next_state_features must have matching shapes.")
    batch_size = state_features.shape[0]
    for name, tensor in (("actions", actions), ("rewards", rewards), ("dones", dones)):
        if tensor.ndim != 1 or tensor.shape[0] != batch_size:
            raise ValueError(f"{name} must be 1D with batch size {batch_size}.")

    q_values, values = critic(state_features)
    if torch.any(actions < 0) or torch.any(actions >= q_values.shape[1]):
        raise ValueError("actions contains an action outside critic action range.")
    selected_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_values = target_critic.values(next_state_features)
        targets = compute_td_targets(rewards, next_values, dones, discount)
    q_loss = weighted_critic_loss(selected_q, targets, sample_weights=sample_weights)
    v_loss = F.mse_loss(values, targets.detach())
    total_loss = q_loss + v_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return QVCriticLoss(
        total_loss=total_loss.detach(),
        q_loss=q_loss.detach(),
        v_loss=v_loss.detach(),
        td_targets=targets.detach(),
    )
