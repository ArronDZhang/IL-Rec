"""Deterministic state-action text rendering for ILRec ILRec embeddings."""

TEMPLATE_VERSION = "ilrec_state_action_v1"


def _require_int(name, value):
    if value is None:
        raise ValueError(f"{name} is required.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def _as_list(value):
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _lookup_item_text(action_id, item_text=None, item_lookup=None):
    if item_text is not None:
        text = str(item_text).strip()
        return text if text else "<unknown>"
    if item_lookup is not None:
        if action_id in item_lookup:
            text = str(item_lookup[action_id]).strip()
            return text if text else "<unknown>"
        action_key = str(action_id)
        if action_key in item_lookup:
            text = str(item_lookup[action_key]).strip()
            return text if text else "<unknown>"
    return "<unknown>"


def _format_history(history_actions, history_rewards):
    actions = _as_list(history_actions)
    rewards = _as_list(history_rewards)
    if not actions:
        return "<empty>"
    parts = []
    for index, action in enumerate(actions):
        action_id = _require_int("history action_id", action)
        reward = float(rewards[index]) if index < len(rewards) else 0.0
        parts.append(f"item_id={action_id} reward={reward:.6f}")
    return "[" + "; ".join(parts) + "]"


def render_state_action_text(
    user_id,
    action_id,
    history_actions=None,
    history_rewards=None,
    item_text=None,
    item_lookup=None,
    dataset=None,
):
    """Render one transition into the stable text format used for LLaMA input."""

    user_id = _require_int("user_id", user_id)
    action_id = _require_int("action_id", action_id)
    dataset_text = "<unknown>" if dataset is None else str(dataset).strip() or "<unknown>"
    history_text = _format_history(history_actions, history_rewards)
    action_text = _lookup_item_text(action_id, item_text=item_text, item_lookup=item_lookup)
    return "\n".join(
        [
            f"template={TEMPLATE_VERSION}",
            f"dataset={dataset_text}",
            f"user_id={user_id}",
            f"history={history_text}",
            f"candidate=item_id={action_id} text={action_text}",
        ]
    )


def render_transition_state_action_text(transition, item_lookup=None, dataset=None):
    """Render expert and policy transition dictionaries through one template."""

    if not isinstance(transition, dict):
        raise ValueError("transition must be a dictionary.")
    history_actions = transition.get("history_action_ids", transition.get("history_actions"))
    history_rewards = transition.get("history_rewards")
    item_text = transition.get("item_text", transition.get("action_text"))
    transition_dataset = dataset if dataset is not None else transition.get("dataset")
    return render_state_action_text(
        user_id=transition.get("user_id"),
        action_id=transition.get("action_id"),
        history_actions=history_actions,
        history_rewards=history_rewards,
        item_text=item_text,
        item_lookup=item_lookup,
        dataset=transition_dataset,
    )
