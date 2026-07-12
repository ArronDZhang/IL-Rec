#!/usr/bin/env python
"""Build ROLeR-compatible CSV data from il-rec Steam/Amazon resources."""

import argparse
import csv
import sys
from pathlib import Path

ROLER_ROOT = Path(__file__).resolve().parents[1]
if str(ROLER_ROOT) not in sys.path:
    sys.path.insert(0, str(ROLER_ROOT))

from environments.ILRec.paths import (
    load_item2id as load_ilrec_item2id,
    load_json,
    load_reward_matrix,
)


INTERACTION_FIELDS = ["user_id", "item_id", "rating", "timestamp"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert il-rec Steam/Amazon sequence data to ROLeR CSV tables."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("amazon", "steam"),
        help="Dataset to convert.",
    )
    parser.add_argument(
        "--ilrec-root",
        required=True,
        type=Path,
        help="Path to the il-rec repository root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "environments"
        / "ILRec"
        / "data",
        help="Directory that will receive the dataset subdirectory.",
    )
    return parser.parse_args()


def load_records(ilrec_root, dataset, split):
    path = ilrec_root / "data" / dataset / f"{split}.json"
    records = load_json(path)
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON list of records.")
    return records


def load_item2id(ilrec_root, dataset):
    return load_ilrec_item2id(ilrec_root, dataset)


def format_rating(value):
    return str(float(value))


def build_interactions(records, item2id, datamaps_path, reward_matrix):
    rows = []
    users = set()

    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {record_index} must be a JSON object.")
        if "userid_encoded" not in record:
            raise ValueError(f"Record {record_index} is missing userid_encoded.")
        if "pos_seq_name" not in record:
            raise ValueError(f"Record {record_index} is missing pos_seq_name.")

        try:
            user_id = int(record["userid_encoded"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Record {record_index} has invalid userid_encoded: "
                f"{record['userid_encoded']!r}"
            ) from exc

        sequence = record["pos_seq_name"]
        if not isinstance(sequence, list):
            raise ValueError(f"Record {record_index} pos_seq_name must be a list.")

        users.add(user_id)
        for timestamp, item_name in enumerate(sequence):
            if item_name not in item2id:
                raise ValueError(
                    f"Item {item_name!r} from record {record_index} is missing "
                    f"from datamaps file {datamaps_path}."
                )
            item_id = item2id[item_name]
            validate_matrix_indices(
                reward_matrix, user_id, item_id, record_index, item_name
            )
            rows.append(
                {
                    "user_id": str(user_id),
                    "item_id": str(item_id),
                    "rating": format_rating(reward_matrix[user_id, item_id]),
                    "timestamp": str(timestamp),
                }
            )

    return rows, users


def validate_matrix_indices(reward_matrix, user_id, item_id, record_index, item_name):
    if user_id < 0 or user_id >= reward_matrix.shape[0]:
        raise ValueError(
            f"Record {record_index} user_id {user_id} is outside reward matrix "
            f"user range 0..{reward_matrix.shape[0] - 1}."
        )
    if item_id < 0 or item_id >= reward_matrix.shape[1]:
        raise ValueError(
            f"Item {item_name!r} maps to item_id {item_id}, outside reward matrix "
            f"item range 0..{reward_matrix.shape[1] - 1}."
        )


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(dataset, ilrec_root, output_root):
    ilrec_root = ilrec_root.resolve()
    item2id, datamaps_path = load_item2id(ilrec_root, dataset)

    train_matrix = load_reward_matrix(ilrec_root, dataset, "train")
    test_matrix = load_reward_matrix(ilrec_root, dataset, "test")
    train_rows, train_users = build_interactions(
        load_records(ilrec_root, dataset, "train"),
        item2id,
        datamaps_path,
        train_matrix,
    )
    test_rows, test_users = build_interactions(
        load_records(ilrec_root, dataset, "test"),
        item2id,
        datamaps_path,
        test_matrix,
    )

    dataset_out = output_root / dataset
    write_csv(dataset_out / "train.csv", INTERACTION_FIELDS, train_rows)
    write_csv(dataset_out / "test.csv", INTERACTION_FIELDS, test_rows)
    write_csv(
        dataset_out / "user.csv",
        ["user_id"],
        [{"user_id": str(user_id)} for user_id in sorted(train_users | test_users)],
    )
    write_csv(
        dataset_out / "item.csv",
        ["item_id"],
        [{"item_id": str(item_id)} for item_id in sorted(set(item2id.values()))],
    )


def main():
    args = parse_args()
    try:
        build_dataset(args.dataset, args.ilrec_root, args.output_root)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
