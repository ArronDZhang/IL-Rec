"""Shared path and datamap helpers for ILRec resources."""

import json
from pathlib import Path

import numpy as np


def env_dataset_root(ilrec_root, dataset):
    return Path(ilrec_root) / "env" / dataset


def datamaps_path(ilrec_root, dataset):
    return env_dataset_root(ilrec_root, dataset) / "datamaps.json"


def reward_matrix_path(ilrec_root, dataset, split):
    return env_dataset_root(ilrec_root, dataset) / f"{dataset}_{split}.npy"


def distance_matrix_path(ilrec_root, dataset, split):
    return env_dataset_root(ilrec_root, dataset) / f"{split}_distance_mat.pickle"


def embedding_path(ilrec_root, dataset):
    return env_dataset_root(ilrec_root, dataset) / f"{dataset}_embedding_task.pt"


def grounding_cache_path(ilrec_root, dataset):
    return env_dataset_root(ilrec_root, dataset) / f"{dataset}_grounding_cache.json"


def load_json(path):
    try:
        with Path(path).open(encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise ValueError(f"Required input file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def load_datamaps(ilrec_root, dataset):
    path = datamaps_path(ilrec_root, dataset)
    datamaps = load_json(path)
    if not isinstance(datamaps, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return datamaps, path


def load_item2id(ilrec_root, dataset):
    datamaps, path = load_datamaps(ilrec_root, dataset)
    item2id = datamaps.get("item2id_dict")
    if not isinstance(item2id, dict):
        raise ValueError(f"{path} must contain item2id_dict.")

    converted = {}
    for item_text, item_id in item2id.items():
        try:
            converted[item_text] = int(item_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid item id for {item_text!r} in {path}: {item_id!r}"
            ) from exc
    return converted, path


def load_id2item(ilrec_root, dataset):
    datamaps, path = load_datamaps(ilrec_root, dataset)
    id2item = datamaps.get("id2item_dict")
    if not isinstance(id2item, dict):
        raise ValueError(f"{path} must contain id2item_dict.")
    return {str(item_id): item_text for item_id, item_text in id2item.items()}, path


def load_reward_matrix(ilrec_root, dataset, split, mmap_mode="r"):
    path = reward_matrix_path(ilrec_root, dataset, split)
    if not path.exists():
        raise ValueError(f"Required reward matrix is missing: {path}")
    return np.load(path, mmap_mode=mmap_mode)
