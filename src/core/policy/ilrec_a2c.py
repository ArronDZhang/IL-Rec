"""ILRec policy-side utilities.

This module starts with small, testable pieces used by the later ILRec policy
class. It intentionally does not modify ROLeR's existing A2C policy.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DemoWeights:
    w_env: torch.Tensor
    w_irl: torch.Tensor
    weights: torch.Tensor


@dataclass(frozen=True)
class ILRecPolicyLoss:
    total_loss: torch.Tensor
    actor_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy_loss: torch.Tensor
    imitation_loss: torch.Tensor


@dataclass(frozen=True)
class MixedReplayPolicyLoss:
    total_loss: torch.Tensor
    actor_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy_loss: torch.Tensor
    imitation_loss: torch.Tensor
    sample_count: int
    demo_sample_count: int
    env_sample_count: int


def _to_float_tensor(value):
    return torch.as_tensor(value, dtype=torch.float32)


def _validate_weight_params(beta, alpha=None, irl_gamma=None, clip_min=None, clip_max=None):
    if beta <= 0:
        raise ValueError("beta must be positive.")
    if alpha is not None and not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    if irl_gamma is not None and irl_gamma < 0:
        raise ValueError("irl_gamma must be non-negative.")
    if clip_min is not None and clip_min <= 0:
        raise ValueError("clip_min must be positive.")
    if clip_max is not None and clip_max <= 0:
        raise ValueError("clip_max must be positive.")
    if clip_min is not None and clip_max is not None and clip_min > clip_max:
        raise ValueError("clip_min must be <= clip_max.")


def _clip_weights(weights, clip_min, clip_max):
    if clip_min is None and clip_max is None:
        return weights
    min_value = 0.0 if clip_min is None else float(clip_min)
    max_value = float("inf") if clip_max is None else float(clip_max)
    return weights.clamp(min=min_value, max=max_value)


def compute_env_weights(demo_advantages, beta, clip_min=None, clip_max=None):
    """Compute ``w_env(s, a) = exp(A_demo(s, a) / beta)``."""

    _validate_weight_params(beta, clip_min=clip_min, clip_max=clip_max)
    advantages = _to_float_tensor(demo_advantages)
    scaled_advantages = (advantages / float(beta)).clamp(min=-80.0, max=80.0)
    weights = torch.exp(scaled_advantages)
    return _clip_weights(weights, clip_min, clip_max)


def compute_irl_weights(discriminator_probs, irl_gamma, eps=1e-8, clip_min=None, clip_max=None):
    """Compute ``w_irl(s, a) = (1 / D(s, a) - 1) ** irl_gamma``."""

    _validate_weight_params(1.0, irl_gamma=irl_gamma, clip_min=clip_min, clip_max=clip_max)
    if eps <= 0:
        raise ValueError("eps must be positive.")
    probs = _to_float_tensor(discriminator_probs).clamp(
        min=float(eps),
        max=1.0 - float(eps),
    )
    weights = torch.pow(1.0 / probs - 1.0, float(irl_gamma))
    return _clip_weights(weights, clip_min, clip_max)


def normalize_weights(weights, eps=1e-8):
    if eps <= 0:
        raise ValueError("eps must be positive.")
    weights = _to_float_tensor(weights)
    if weights.numel() == 0:
        return torch.ones_like(weights)
    mean = weights.mean()
    if not torch.isfinite(mean) or mean.abs() < eps:
        return torch.ones_like(weights)
    return weights / mean.clamp(min=float(eps))


def compute_demo_weights(
    demo_advantages,
    discriminator_probs,
    beta,
    alpha,
    irl_gamma,
    eps=1e-8,
    clip_min=1e-6,
    clip_max=1e6,
    normalize=True,
):
    """Compute normalized ILRec demonstration weights.

    ``irl_gamma`` is the exponent from the ILRec weighting formula and is named
    separately from the RL discount factor to avoid accidental reuse.
    """

    _validate_weight_params(
        beta,
        alpha=alpha,
        irl_gamma=irl_gamma,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    w_env = compute_env_weights(demo_advantages, beta, clip_min, clip_max)
    w_irl = compute_irl_weights(discriminator_probs, irl_gamma, eps, clip_min, clip_max)
    if w_env.shape != w_irl.shape:
        raise ValueError("demo_advantages and discriminator_probs must have matching shapes.")

    weights = torch.pow(w_env, float(alpha)) * torch.pow(w_irl, 1.0 - float(alpha))
    weights = _clip_weights(weights, clip_min, clip_max)
    if normalize:
        weights = normalize_weights(weights, eps=eps)
        weights = _clip_weights(weights, clip_min, clip_max)
    return DemoWeights(w_env=w_env, w_irl=w_irl, weights=weights)


def weighted_imitation_loss(action_logits, demo_actions, demo_weights):
    """Compute ``-E[w(s, a) * log pi(a | s)]`` for discrete actions."""

    logits = torch.as_tensor(action_logits, dtype=torch.float32)
    if logits.ndim != 2:
        raise ValueError("action_logits must be a 2D tensor of shape [batch, actions].")

    actions = torch.as_tensor(demo_actions, dtype=torch.long, device=logits.device)
    weights = torch.as_tensor(demo_weights, dtype=logits.dtype, device=logits.device)
    if actions.ndim != 1 or weights.ndim != 1:
        raise ValueError("demo_actions and demo_weights must be 1D tensors.")
    if logits.shape[0] != actions.shape[0] or logits.shape[0] != weights.shape[0]:
        raise ValueError("action_logits, demo_actions, and demo_weights batch sizes must match.")
    if torch.any(actions < 0) or torch.any(actions >= logits.shape[1]):
        raise ValueError("demo_actions contains an action outside the logits range.")

    log_probs = F.log_softmax(logits, dim=-1)
    selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    return -(weights * selected_log_probs).mean()


def _validate_policy_batch(logits, actions, advantages, values, returns):
    if logits.ndim != 2:
        raise ValueError("action_logits must be a 2D tensor of shape [batch, actions].")
    batch_size = logits.shape[0]
    for name, tensor in (
        ("actions", actions),
        ("advantages", advantages),
        ("values", values),
        ("returns", returns),
    ):
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be a 1D tensor.")
        if tensor.shape[0] != batch_size:
            raise ValueError(f"{name} batch size must match action_logits.")
    if torch.any(actions < 0) or torch.any(actions >= logits.shape[1]):
        raise ValueError("actions contains an action outside the logits range.")


def compute_ilrec_policy_loss(
    action_logits,
    actions,
    advantages,
    values,
    returns,
    demo_action_logits=None,
    demo_actions=None,
    demo_weights=None,
    lambda_imit=0.0,
    vf_coef=0.5,
    alpha_ent=0.0,
    detach_demo_weights=True,
):
    """Compute combined actor-critic and weighted-imitation ILRec loss."""

    if lambda_imit < 0:
        raise ValueError("lambda_imit must be non-negative.")
    if vf_coef < 0:
        raise ValueError("vf_coef must be non-negative.")
    if alpha_ent < 0:
        raise ValueError("alpha_ent must be non-negative.")

    logits = torch.as_tensor(action_logits, dtype=torch.float32)
    actions = torch.as_tensor(actions, dtype=torch.long, device=logits.device)
    advantages = torch.as_tensor(advantages, dtype=logits.dtype, device=logits.device)
    values = torch.as_tensor(values, dtype=logits.dtype, device=logits.device)
    returns = torch.as_tensor(returns, dtype=logits.dtype, device=logits.device)
    _validate_policy_batch(logits, actions, advantages, values, returns)

    log_probs = F.log_softmax(logits, dim=-1)
    selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    actor_loss = -(selected_log_probs * advantages.detach()).mean()
    value_loss = F.mse_loss(values, returns)
    probs = torch.softmax(logits, dim=-1)
    entropy_loss = -(probs * log_probs).sum(dim=-1).mean()

    imitation_loss = logits.new_tensor(0.0)
    has_demo = demo_action_logits is not None or demo_actions is not None or demo_weights is not None
    if has_demo:
        if demo_action_logits is None or demo_actions is None or demo_weights is None:
            raise ValueError("demo_action_logits, demo_actions, and demo_weights must be provided together.")
        weights = demo_weights.detach() if detach_demo_weights and isinstance(demo_weights, torch.Tensor) else demo_weights
        imitation_loss = weighted_imitation_loss(demo_action_logits, demo_actions, weights)
    elif lambda_imit > 0.0:
        raise ValueError("lambda_imit > 0 requires demonstration logits, actions, and weights.")

    total_loss = (
        actor_loss
        + float(vf_coef) * value_loss
        - float(alpha_ent) * entropy_loss
        + float(lambda_imit) * imitation_loss
    )
    return ILRecPolicyLoss(
        total_loss=total_loss,
        actor_loss=actor_loss,
        value_loss=value_loss,
        entropy_loss=entropy_loss,
        imitation_loss=imitation_loss,
    )


def compute_mixed_replay_policy_loss(
    action_logits,
    actions,
    advantages,
    is_demo,
    demo_weights=None,
    lambda_imit=0.0,
    alpha_ent=0.0,
):
    """Compute the ILRec policy loss over mixed replay samples.

    The RL objective uses every sampled transition from ``B_env union B_demo``.
    The imitation objective is applied only to sampled demonstration records.
    """

    if lambda_imit < 0:
        raise ValueError("lambda_imit must be non-negative.")
    if alpha_ent < 0:
        raise ValueError("alpha_ent must be non-negative.")

    logits = torch.as_tensor(action_logits, dtype=torch.float32)
    if logits.ndim != 2:
        raise ValueError("action_logits must be a 2D tensor of shape [batch, actions].")
    batch_size = logits.shape[0]
    if batch_size == 0:
        raise ValueError("mixed replay policy loss requires at least one sample.")

    actions = torch.as_tensor(actions, dtype=torch.long, device=logits.device)
    advantages = torch.as_tensor(advantages, dtype=logits.dtype, device=logits.device)
    demo_mask = torch.as_tensor(is_demo, dtype=torch.bool, device=logits.device)
    for name, tensor in (("actions", actions), ("advantages", advantages), ("is_demo", demo_mask)):
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be a 1D tensor.")
        if tensor.shape[0] != batch_size:
            raise ValueError(f"{name} batch size must match action_logits.")
    if torch.any(actions < 0) or torch.any(actions >= logits.shape[1]):
        raise ValueError("actions contains an action outside the logits range.")

    log_probs = F.log_softmax(logits, dim=-1)
    selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    actor_loss = -(selected_log_probs * advantages.detach()).mean()
    value_loss = logits.new_tensor(0.0)
    probs = torch.softmax(logits, dim=-1)
    entropy_loss = -(probs * log_probs).sum(dim=-1).mean()

    imitation_loss = logits.new_tensor(0.0)
    demo_sample_count = int(demo_mask.sum().detach().cpu().item())
    if demo_sample_count > 0:
        if demo_weights is None:
            weights = torch.ones(batch_size, dtype=logits.dtype, device=logits.device)
        else:
            weights = torch.as_tensor(demo_weights, dtype=logits.dtype, device=logits.device)
            if weights.ndim != 1 or weights.shape[0] != batch_size:
                raise ValueError("demo_weights must be 1D and match action_logits batch size.")
        imitation_loss = weighted_imitation_loss(logits[demo_mask], actions[demo_mask], weights[demo_mask])

    total_loss = actor_loss - float(alpha_ent) * entropy_loss + float(lambda_imit) * imitation_loss
    return MixedReplayPolicyLoss(
        total_loss=total_loss,
        actor_loss=actor_loss,
        value_loss=value_loss,
        entropy_loss=entropy_loss,
        imitation_loss=imitation_loss,
        sample_count=int(batch_size),
        demo_sample_count=demo_sample_count,
        env_sample_count=int(batch_size - demo_sample_count),
    )
