"""World-model demonstration return utilities for ILRec ILRec."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DemoReturnResult:
    transitions: list
    diagnostics: dict


def _require_int(transition, key, index):
    if key not in transition:
        raise ValueError(f"transition {index} is missing {key}.")
    try:
        return int(transition[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"transition {index} has invalid {key}: {transition[key]!r}") from exc


def _trajectory_segments(transitions):
    segments = []
    current = []
    previous_trajectory_id = None
    for index, transition in enumerate(transitions):
        trajectory_id = transition.get("trajectory_id", transition.get("traj_id", f"transition-{index}"))
        if current and trajectory_id != previous_trajectory_id:
            segments.append(current)
            current = []
        current.append(index)
        previous_trajectory_id = trajectory_id
        if transition.get("done"):
            segments.append(current)
            current = []
            previous_trajectory_id = None
    if current:
        segments.append(current)
    return segments


def annotate_world_model_demo_returns(transitions, reward_lookup, discount):
    """Copy transitions and add world-model rewards plus discounted returns."""

    discount = float(discount)
    if discount < 0:
        raise ValueError("discount must be non-negative.")
    copied = [dict(transition) for transition in transitions]
    if not copied:
        return DemoReturnResult(
            transitions=[],
            diagnostics={
                "return_method": "world_model_discounted_return",
                "return_source": "world_model_lookup",
                "discount": discount,
                "transition_count": 0,
                "trajectory_count": 0,
                "override_reward_count": 0,
            },
        )

    user_ids = [_require_int(transition, "user_id", index) for index, transition in enumerate(copied)]
    action_ids = [_require_int(transition, "action_id", index) for index, transition in enumerate(copied)]
    rewards = np.asarray(reward_lookup(user_ids, action_ids), dtype=float).reshape(-1)
    if rewards.shape[0] != len(copied):
        raise ValueError(
            f"reward_lookup returned {rewards.shape[0]} rewards for {len(copied)} transitions."
        )
    override_reward_count = 0
    for transition, reward in zip(copied, rewards):
        if "world_model_reward_override" in transition:
            transition["world_model_reward"] = float(transition["world_model_reward_override"])
            transition["world_model_reward_source"] = "transition_override"
            override_reward_count += 1
        else:
            transition["world_model_reward"] = float(reward)
            transition["world_model_reward_source"] = "world_model_lookup"

    segments = _trajectory_segments(copied)
    for segment in segments:
        running = 0.0
        for index in reversed(segment):
            running = float(copied[index]["world_model_reward"]) + discount * running
            copied[index]["demo_return"] = float(running)
            copied[index]["demo_return_source"] = copied[index]["world_model_reward_source"]

    return_source = (
        "world_model_lookup_with_transition_overrides"
        if override_reward_count
        else "world_model_lookup"
    )

    diagnostics = {
        "return_method": "world_model_discounted_return",
        "return_source": return_source,
        "discount": discount,
        "transition_count": len(copied),
        "trajectory_count": len(segments),
        "override_reward_count": override_reward_count,
        "min_return": float(min(transition["demo_return"] for transition in copied)),
        "max_return": float(max(transition["demo_return"] for transition in copied)),
    }
    return DemoReturnResult(transitions=copied, diagnostics=diagnostics)
