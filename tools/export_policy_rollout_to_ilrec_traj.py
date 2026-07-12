#!/usr/bin/env python
"""Export policy rollouts to il-rec trajectory JSON format."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert policy rollout JSON into trajs_agent-compatible records."
    )
    parser.add_argument(
        "--rollout-json",
        required=True,
        type=Path,
        help="Input rollout JSON with records containing userid and steps.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output trajectory JSON path.",
    )
    parser.add_argument(
        "--ilrec-root",
        type=Path,
        default=Path("/home/hehui/il-rec"),
        help="il-rec repository root used for default output path.",
    )
    parser.add_argument(
        "--dataset",
        choices=("amazon", "steam"),
        default="amazon",
        help="Dataset label used only for default output naming.",
    )
    parser.add_argument("--message", default="ilrec_gpt35_seed0")
    parser.add_argument(
        "--terminal-failure-reward",
        type=float,
        default=-1000.0,
        help="Reward text emitted for terminal failure steps.",
    )
    return parser.parse_args()


def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def default_output_path(ilrec_root, dataset, message):
    return ilrec_root / "trajs_agent" / "ilrec_eval" / f"{dataset}_{message}.json"


def normalize_rollouts(payload):
    if isinstance(payload, list):
        return [(str(index), record) for index, record in enumerate(payload)]
    if isinstance(payload, dict):
        return [(str(key), record) for key, record in payload.items()]
    raise ValueError("rollout JSON must contain a list or object.")


def item_text_from_step(step):
    for key in ("item", "item_text", "raw_item_text", "grounded_item_text"):
        value = step.get(key)
        if value is not None:
            return str(value)
    if "action_id" in step:
        return f"item_id:{step['action_id']}"
    return "unknown"


def reward_from_step(step, terminal_failure_reward):
    if step.get("terminal_failure"):
        return float(terminal_failure_reward)
    if "reward" not in step:
        raise ValueError(f"Rollout step is missing reward: {step!r}")
    return float(step["reward"])


def format_reward(value):
    return f"{float(value):.6f}"


def record_to_ilrec(record_key, record, terminal_failure_reward):
    if not isinstance(record, dict):
        raise ValueError(f"Rollout record {record_key} must be an object.")
    steps = record.get("steps")
    if not isinstance(steps, list):
        raise ValueError(f"Rollout record {record_key} must contain a steps list.")

    userid = record.get("userid", record.get("user_id", record_key))
    lines = [f"The user's policy rollout is for user {userid}."]
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"Rollout record {record_key} step {index} must be an object.")
        item_text = item_text_from_step(step)
        reward = reward_from_step(step, terminal_failure_reward)
        done = bool(step.get("done", index == len(steps)))
        lines.append(f"Thought {index}: Continue the learned policy rollout.")
        lines.append(f"Action {index}: recommend[{item_text}]")
        if done:
            if step.get("terminal_failure"):
                status = "Episode finished, User Stop"
            else:
                status = "Episode finished"
        else:
            status = "Episode continue"
        lines.append(f"Observation {index}: {status}, reward={format_reward(reward)}")

    return {
        "userid": userid,
        "prompt": lines[0],
        "traj": "\n".join(lines),
        "traj_by_line": lines,
        "source": "policy_rollout_export",
    }


def export_rollouts(payload, terminal_failure_reward=-1000.0):
    output = {}
    for record_key, record in normalize_rollouts(payload):
        trajectory_id = record.get("trajectory_id", record_key) if isinstance(record, dict) else record_key
        output[str(trajectory_id)] = record_to_ilrec(
            str(trajectory_id),
            record,
            terminal_failure_reward,
        )
    return output


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    output_path = args.output or default_output_path(args.ilrec_root, args.dataset, args.message)
    exported = export_rollouts(load_json(args.rollout_json), args.terminal_failure_reward)
    write_json(output_path, exported)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "records": len(exported),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
