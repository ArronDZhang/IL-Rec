#!/usr/bin/env python
"""Aggregate ILRec evaluation summaries across evaluation seeds."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


CSV_FIELDS = [
    "dataset",
    "env",
    "setting",
    "seed_count",
    "mean_avg_length",
    "std_avg_length",
    "mean_avg_reward",
    "std_avg_reward",
    "mean_avg_return",
    "std_avg_return",
    "pooled_total_length",
    "pooled_total_return",
    "pooled_avg_reward",
    "eval_action_selection",
    "policy_logit_clamp",
    "eval_logit_clamp",
    "combined_termination_reasons",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate ILRec summary JSON files.")
    parser.add_argument("--dataset", required=True, choices=("amazon", "steam"))
    parser.add_argument("--env", required=True, choices=("AmazonEnv-v0", "SteamEnv-v0"))
    parser.add_argument("--setting", default="FB", choices=("FB",))
    parser.add_argument(
        "--summary",
        action="append",
        type=Path,
        required=True,
        help="Per-seed summary JSON. Pass once per evaluation seed.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args(argv)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_reason_counts(value):
    if value is None:
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("termination_reasons must be a dict or JSON object string.")
    return {str(key): int(count) for key, count in value.items()}


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values):
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def config_value(summary, key, default=None):
    config = summary.get("config")
    if isinstance(config, dict) and key in config:
        return config[key]
    return summary.get(key, default)


def aggregate(dataset, env, setting, summaries):
    if not summaries:
        raise ValueError("At least one summary is required.")

    lengths = [float(summary["avg_length"]) for summary in summaries]
    rewards = [float(summary["avg_reward"]) for summary in summaries]
    returns = [float(summary["avg_return"]) for summary in summaries]
    user_counts = [int(summary.get("user_count", 0)) for summary in summaries]
    total_lengths = [
        float(summary.get("total_length", lengths[index] * user_counts[index]))
        for index, summary in enumerate(summaries)
    ]
    total_returns = [
        float(summary.get("total_return", returns[index] * user_counts[index]))
        for index, summary in enumerate(summaries)
    ]

    reasons = Counter()
    for summary in summaries:
        reasons.update(parse_reason_counts(summary.get("termination_reasons")))

    pooled_total_length = sum(total_lengths)
    pooled_total_return = sum(total_returns)
    first = summaries[0]
    result = {
        "method": "ILRec",
        "dataset": dataset,
        "env": env,
        "setting": setting,
        "evaluation_scope": "5_eval_seeds_on_standard_100users",
        "seed_count": len(summaries),
        "eval_seeds": [int(summary.get("eval_seed", index)) for index, summary in enumerate(summaries)],
        "mean_avg_length": mean(lengths),
        "std_avg_length": std(lengths),
        "mean_avg_reward": mean(rewards),
        "std_avg_reward": std(rewards),
        "mean_avg_return": mean(returns),
        "std_avg_return": std(returns),
        "pooled_total_length": pooled_total_length,
        "pooled_total_return": pooled_total_return,
        "pooled_avg_reward": pooled_total_return / pooled_total_length if pooled_total_length else 0.0,
        "combined_termination_reasons": dict(sorted(reasons.items())),
        "eval_action_selection": config_value(first, "eval_action_selection", "sample"),
        "policy_logit_clamp": float(config_value(first, "policy_logit_clamp", 0.0)),
        "eval_logit_clamp": float(config_value(first, "eval_logit_clamp", config_value(first, "policy_logit_clamp", 0.0))),
        "config": {
            "env": env,
            "setting": setting,
            "train_episodes": int(config_value(first, "train_episodes", 100000)),
            "discount": float(config_value(first, "discount", 0.5)),
            "train_action_selection": config_value(first, "train_action_selection", "sample"),
            "eval_action_selection": config_value(first, "eval_action_selection", "sample"),
            "mixed_replay_sampling": config_value(first, "mixed_replay_sampling", "global_priority"),
            "mixed_replay_env_priority_scale": float(config_value(first, "mixed_replay_env_priority_scale", 0.05)),
            "train_normalize_advantages": bool(config_value(first, "train_normalize_advantages", True)),
            "train_advantage_clip": float(config_value(first, "train_advantage_clip", 5.0)),
            "train_actor_row_norm_project": float(config_value(first, "train_actor_row_norm_project", 10.0)),
            "train_actor_bias_clamp": float(config_value(first, "train_actor_bias_clamp", 5.0)),
            "policy_logit_clamp": float(config_value(first, "policy_logit_clamp", 0.0)),
            "policy_logit_clamp_mode": config_value(first, "policy_logit_clamp_mode", "tanh"),
            "eval_logit_clamp": float(config_value(first, "eval_logit_clamp", config_value(first, "policy_logit_clamp", 0.0))),
        },
    }
    return result


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_csv(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(payload)
    row["combined_termination_reasons"] = json.dumps(
        row["combined_termination_reasons"],
        sort_keys=True,
    )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def main(argv=None):
    args = parse_args(argv)
    summaries = [load_json(path) for path in args.summary]
    payload = aggregate(args.dataset, args.env, args.setting, summaries)
    write_json(args.output_json, payload)
    write_csv(args.output_csv, payload)


if __name__ == "__main__":
    main()
