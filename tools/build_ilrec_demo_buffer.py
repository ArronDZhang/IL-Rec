#!/usr/bin/env python
"""Build ILRec GPT3.5 demonstration buffers for ROLeR ILRec training."""

import argparse
import glob
import json
import pickle
import re
import sys
from pathlib import Path

ROLER_ROOT = Path(__file__).resolve().parents[1]
if str(ROLER_ROOT) not in sys.path:
    sys.path.insert(0, str(ROLER_ROOT))

from environments.ILRec.paths import (
    embedding_path as ilrec_embedding_path,
    grounding_cache_path as ilrec_grounding_cache_path,
    load_id2item,
    load_item2id as load_ilrec_item2id,
    load_json,
    load_reward_matrix,
)


ACTION_RE = re.compile(r"\bAction\s*\d*\s*:\s*recommend\[(.*?)\]", re.IGNORECASE)
REPLACEMENT_RE = re.compile(r"\binstead\s*,?\s*recommend\[(.*?)\]", re.IGNORECASE)
REWARD_RE = re.compile(
    r"\breward\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
PROMPT_TEMPLATES = {
    "amazon": "The type of {item} book is",
    "steam": "The type of {item} is",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse il-rec GPT3.5 trajectories into a ROLeR demo buffer."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("amazon", "steam"),
        help="ILRec dataset name.",
    )
    parser.add_argument(
        "--ilrec-root",
        required=True,
        type=Path,
        help="Path to the il-rec repository root.",
    )
    parser.add_argument(
        "--trajectory",
        action="append",
        type=Path,
        default=[],
        help="Trajectory JSON file to parse. May be repeated.",
    )
    parser.add_argument(
        "--traj-glob",
        action="append",
        default=[],
        help="Glob for trajectory JSON files. May be repeated.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=("train", "test"),
        help="Reward matrix split used for reward lookup.",
    )
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        type=Path,
        default=None,
        help="Output pickle path. Defaults to environments/ILRec/data/{dataset}/demo_gpt35.pkl.",
    )
    parser.add_argument(
        "--embedding-fallback",
        action="store_true",
        help="Use LLaMA embeddings to ground items that are not exact datamap matches.",
    )
    parser.add_argument(
        "--embedding-path",
        type=Path,
        default=None,
        help="Optional override for {dataset}_embedding_task.pt.",
    )
    parser.add_argument(
        "--model-path",
        default="/home/hehui/llama2-7bhf",
        help="Local LLaMA model path used only when --embedding-fallback is set.",
    )
    parser.add_argument(
        "--grounding-batch-size",
        type=int,
        default=16,
        help="Batch size for LLaMA query embedding when --embedding-fallback is set.",
    )
    parser.add_argument(
        "--grounding-device",
        default=None,
        help="Torch device for query embedding. Defaults to cuda when available, else cpu.",
    )
    parser.add_argument(
        "--grounding-cache-path",
        type=Path,
        default=None,
        help="Optional persistent embedding-grounding cache path.",
    )
    parser.add_argument(
        "--grounding-similarity",
        choices=("cosine", "euclidean"),
        default="cosine",
        help="Similarity metric for embedding fallback grounding. Euclidean is diagnostic legacy behavior.",
    )
    parser.add_argument(
        "--terminal-fallback-policy",
        choices=("keep", "drop", "reward_minus1000", "reward_zero"),
        default="keep",
        help=(
            "How to handle terminal fallback transitions where an unavailable action "
            "was replaced by an Observation recommendation and the observed reward is terminal failure."
        ),
    )
    return parser.parse_args()


def load_item2id(ilrec_root, dataset):
    item2id, _ = load_ilrec_item2id(ilrec_root, dataset)
    return item2id


def default_output_path(dataset):
    return (
        Path(__file__).resolve().parents[1]
        / "environments"
        / "ILRec"
        / "data"
        / dataset
        / "demo_gpt35.pkl"
    )


def default_trajectory_glob(ilrec_root, dataset):
    return str(
        ilrec_root
        / "trajs_agent"
        / f"{dataset}_train_0_100_gpt-3.5-turbo-16k_0.5_*.json"
    )


def default_embedding_path(ilrec_root, dataset):
    return ilrec_embedding_path(ilrec_root, dataset)


def default_grounding_cache_path(ilrec_root, dataset):
    return ilrec_grounding_cache_path(ilrec_root, dataset)


def torch_load_weights(torch_module, path):
    try:
        return torch_module.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch_module.load(path, map_location="cpu")


def resolve_trajectory_paths(ilrec_root, dataset, trajectory_paths, trajectory_globs):
    paths = [Path(path) for path in trajectory_paths]
    globs = list(trajectory_globs)
    if not paths and not globs:
        globs.append(default_trajectory_glob(ilrec_root, dataset))

    for pattern in globs:
        paths.extend(Path(path) for path in sorted(glob.glob(pattern)))

    seen = set()
    unique_paths = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(resolved)
    if not unique_paths:
        raise ValueError("No trajectory files matched the requested inputs.")
    return unique_paths


def lines_from_record(record, record_id, source_path):
    if not isinstance(record, dict):
        raise ValueError(f"{source_path}:{record_id} must be a JSON object.")
    if isinstance(record.get("traj_by_line"), list):
        return [str(line) for line in record["traj_by_line"]]
    if isinstance(record.get("traj"), str):
        return record["traj"].splitlines()
    raise ValueError(f"{source_path}:{record_id} must contain traj_by_line or traj.")


def user_id_from_record(record, record_id, source_path):
    value = record.get("userid", record_id)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source_path}:{record_id} has invalid userid value: {value!r}"
        ) from exc


def parse_action_reward_pairs(lines, diagnostics):
    pairs = []
    pending = None
    action_index = 0
    for line in lines:
        action_match = ACTION_RE.search(line)
        if action_match:
            if pending is not None:
                diagnostics["skipped_missing_reward"] += 1
            item_text = action_match.group(1).strip()
            pending = {
                "item_text": item_text,
                "attempted_item_text": item_text,
                "action_replaced": False,
                "step_index": action_index,
            }
            replacement_match = REPLACEMENT_RE.search(line)
            if replacement_match:
                pending["item_text"] = replacement_match.group(1).strip()
                pending["action_replaced"] = True
                diagnostics["action_replacements"] += 1
            action_index += 1
            diagnostics["actions_seen"] += 1
            continue

        replacement_match = REPLACEMENT_RE.search(line)
        if pending is not None and replacement_match:
            pending["item_text"] = replacement_match.group(1).strip()
            pending["action_replaced"] = True
            diagnostics["action_replacements"] += 1
            continue

        reward_match = REWARD_RE.search(line)
        if reward_match:
            diagnostics["rewards_seen"] += 1
            if pending is None:
                continue
            pending["observed_reward"] = float(reward_match.group(1))
            pairs.append(pending)
            pending = None

    if pending is not None:
        diagnostics["skipped_missing_reward"] += 1
    return pairs


def validate_matrix_indices(matrix, user_id, item_id, source_path, record_id):
    if user_id < 0 or user_id >= matrix.shape[0]:
        raise ValueError(
            f"{source_path}:{record_id} user_id {user_id} is outside reward matrix "
            f"user range 0..{matrix.shape[0] - 1}."
        )
    if item_id < 0 or item_id >= matrix.shape[1]:
        raise ValueError(
            f"{source_path}:{record_id} item_id {item_id} is outside reward matrix "
            f"item range 0..{matrix.shape[1] - 1}."
        )


def empty_diagnostics():
    return {
        "records_seen": 0,
        "actions_seen": 0,
        "rewards_seen": 0,
        "transitions_written": 0,
        "skipped_records": 0,
        "skipped_missing_reward": 0,
        "action_replacements": 0,
        "terminal_fallback_transitions": 0,
        "terminal_fallback_transitions_dropped": 0,
        "terminal_fallback_reward_overrides": 0,
        "exact_groundings": 0,
        "embedding_groundings": 0,
        "grounding_failures": 0,
        "terminal_failures": 0,
        "unknown_items": [],
    }


class LlamaQueryEncoder:
    def __init__(self, dataset, model_path, batch_size=16, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.device == "cuda":
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
            ).to(self.device)

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token or self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2
        self.model.eval()

    def encode(self, item_texts):
        embeddings = []
        template = PROMPT_TEMPLATES[self.dataset]
        with self.torch.inference_mode():
            for start in range(0, len(item_texts), self.batch_size):
                batch_items = item_texts[start : start + self.batch_size]
                prompts = [template.format(item=item) for item in batch_items]
                encoded = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
                outputs = self.model(
                    encoded.input_ids,
                    attention_mask=encoded.attention_mask,
                    output_hidden_states=True,
                )
                embeddings.append(outputs.hidden_states[-1][:, -1, :].detach().cpu())
        return self.torch.cat(embeddings, dim=0).float()


class EmbeddingGrounder:
    def __init__(
        self,
        dataset,
        ilrec_root,
        embedding_path=None,
        model_path="/home/hehui/llama2-7bhf",
        batch_size=16,
        device=None,
        query_encoder=None,
        cache_path=None,
        grounding_similarity="cosine",
    ):
        import torch

        self.torch = torch
        self.dataset = dataset
        self.embedding_path = embedding_path or default_embedding_path(ilrec_root, dataset)
        self.cache_path = Path(cache_path) if cache_path else None
        self.grounding_similarity = str(grounding_similarity)
        if self.grounding_similarity not in {"cosine", "euclidean"}:
            raise ValueError("grounding_similarity must be 'cosine' or 'euclidean'.")
        embedding_data = torch_load_weights(torch, self.embedding_path)
        if "embeddings" not in embedding_data or "indexs" not in embedding_data:
            raise ValueError(f"{self.embedding_path} must contain embeddings and indexs.")
        self.embeddings = embedding_data["embeddings"].detach().cpu().float()
        self.indexs = embedding_data["indexs"].detach().cpu().long()
        if self.embeddings.ndim != 2 or self.indexs.ndim != 1:
            raise ValueError(f"{self.embedding_path} has invalid embedding/index shapes.")
        if self.embeddings.shape[0] != self.indexs.shape[0]:
            raise ValueError(f"{self.embedding_path} embeddings and indexs length mismatch.")

        self.id2item, self.datamaps_path = load_id2item(ilrec_root, dataset)
        self.query_encoder = query_encoder or LlamaQueryEncoder(
            dataset,
            model_path,
            batch_size=batch_size,
            device=device,
        )
        self.cache = {}
        self._load_cache()

    def _cache_metadata(self):
        return {
            "dataset": self.dataset,
            "embedding_path": str(self.embedding_path.resolve()),
            "embedding_mtime": self.embedding_path.stat().st_mtime,
            "datamaps_path": str(self.datamaps_path.resolve()),
            "datamaps_mtime": self.datamaps_path.stat().st_mtime,
            "grounding_similarity": self.grounding_similarity,
        }

    def _load_cache(self):
        if self.cache_path is None or not self.cache_path.exists():
            return
        try:
            payload = load_json(self.cache_path)
        except ValueError:
            return
        if payload.get("metadata") != self._cache_metadata():
            return
        entries = payload.get("entries", {})
        if not isinstance(entries, dict):
            return
        for item_text, result in entries.items():
            if not isinstance(result, dict):
                continue
            if "item_id" not in result:
                continue
            self.cache[str(item_text)] = {
                "item_id": int(result["item_id"]),
                "matched_item_text": str(result.get("matched_item_text", "")),
                "distance": float(result.get("distance", 0.0)),
                "similarity": float(result["similarity"]) if result.get("similarity") is not None else None,
                "metric": str(result.get("metric", self.grounding_similarity)),
            }

    def save_cache(self):
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self._cache_metadata(),
            "entries": self.cache,
        }
        tmp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        tmp_path.replace(self.cache_path)

    def ground(self, item_text):
        return self.ground_many([item_text]).get(item_text)

    def ground_many(self, item_texts):
        missing = []
        seen_missing = set()
        for item_text in item_texts:
            if item_text in self.cache or item_text in seen_missing:
                continue
            seen_missing.add(item_text)
            missing.append(item_text)

        if missing:
            query_embedding = self.query_encoder.encode(missing)
            self._cache_query_results(missing, query_embedding)

        return {item_text: self.cache[item_text] for item_text in item_texts if item_text in self.cache}

    def _cache_query_results(self, item_texts, query_embedding):
        query_embedding = self.torch.as_tensor(query_embedding).detach().cpu().float()
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.unsqueeze(0)
        if query_embedding.shape[0] != len(item_texts):
            raise ValueError(
                f"Query encoder returned {query_embedding.shape[0]} embeddings for {len(item_texts)} item texts."
            )
        if query_embedding.shape[1] != self.embeddings.shape[1]:
            raise ValueError(
                f"Query embedding dim {query_embedding.shape[1]} does not match "
                f"item embedding dim {self.embeddings.shape[1]}."
            )

        if self.grounding_similarity == "cosine":
            query_norm = self.torch.nn.functional.normalize(query_embedding, p=2, dim=1, eps=1e-12)
            item_norm = self.torch.nn.functional.normalize(self.embeddings, p=2, dim=1, eps=1e-12)
            scores = query_norm @ item_norm.T
            nearest_positions = self.torch.argmax(scores, dim=1)
        else:
            distances = self.torch.cdist(query_embedding, self.embeddings, p=2)
            nearest_positions = self.torch.argmin(distances, dim=1)
        for row_index, (item_text, nearest_pos) in enumerate(zip(item_texts, nearest_positions.tolist())):
            nearest_pos = int(nearest_pos)
            item_id = int(self.indexs[nearest_pos].item())
            if self.grounding_similarity == "cosine":
                similarity = float(scores[row_index, nearest_pos].item())
                distance = float(1.0 - similarity)
            else:
                distance = float(distances[row_index, nearest_pos].item())
                similarity = None
            self.cache[item_text] = {
                "item_id": item_id,
                "matched_item_text": self.id2item.get(str(item_id), ""),
                "distance": distance,
                "similarity": similarity,
                "metric": self.grounding_similarity,
            }


def ground_item(item_text, item2id, grounder, diagnostics):
    if item_text in item2id:
        diagnostics["exact_groundings"] += 1
        return {
            "item_id": item2id[item_text],
            "grounded_item_text": item_text,
            "grounding_status": "exact",
            "grounding_distance": 0.0,
            "grounding_similarity": 1.0,
            "grounding_metric": "exact",
        }

    if grounder is None:
        return None

    result = grounder.ground(item_text)
    if result is None:
        return None
    diagnostics["embedding_groundings"] += 1
    return {
        "item_id": int(result["item_id"]),
        "grounded_item_text": result.get("matched_item_text", ""),
        "grounding_status": "embedding",
        "grounding_distance": float(result.get("distance", 0.0)),
        "grounding_similarity": result.get("similarity"),
        "grounding_metric": result.get("metric", "euclidean"),
    }


def build_transition(source_path, record_id, user_id, pair, grounding, matrix):
    observed_reward = float(pair["observed_reward"])
    terminal_failure = observed_reward <= -1000.0
    item_id = grounding["item_id"]
    reward = float(matrix[user_id, item_id])
    return {
        "user_id": user_id,
        "obs": [user_id],
        "next_obs": [user_id],
        "action_id": item_id,
        "item_id": item_id,
        "reward": reward,
        "observed_reward": observed_reward,
        "metric_reward": 0.0 if terminal_failure else observed_reward,
        "done": terminal_failure,
        "terminal_failure": terminal_failure,
        "trajectory_id": f"{source_path.name}:{record_id}",
        "step_index": int(pair["step_index"]),
        "raw_item_text": pair["item_text"],
        "attempted_item_text": pair.get("attempted_item_text", pair["item_text"]),
        "action_replaced": bool(pair.get("action_replaced", False)),
        "grounded_item_text": grounding["grounded_item_text"],
        "grounding_status": grounding["grounding_status"],
        "grounding_distance": grounding["grounding_distance"],
        "grounding_similarity": grounding["grounding_similarity"],
        "grounding_metric": grounding["grounding_metric"],
    }


def build_buffer(dataset, ilrec_root, split, trajectory_paths, grounder=None, terminal_fallback_policy="keep"):
    if terminal_fallback_policy not in {"keep", "drop", "reward_minus1000", "reward_zero"}:
        raise ValueError("terminal_fallback_policy must be keep, drop, reward_minus1000, or reward_zero.")
    item2id = load_item2id(ilrec_root, dataset)
    reward_matrix = load_reward_matrix(ilrec_root, dataset, split)
    diagnostics = empty_diagnostics()
    transitions = []

    for source_path in trajectory_paths:
        data = load_json(source_path)
        if not isinstance(data, dict):
            raise ValueError(f"{source_path} must contain a JSON object.")

        for record_id, record in data.items():
            diagnostics["records_seen"] += 1
            user_id = user_id_from_record(record, record_id, source_path)
            pairs = parse_action_reward_pairs(
                lines_from_record(record, record_id, source_path),
                diagnostics,
            )
            record_transitions = []
            if grounder is not None:
                grounder.ground_many(
                    [
                        pair["item_text"]
                        for pair in pairs
                        if pair["item_text"] not in item2id
                    ]
                )
            for pair in pairs:
                item_text = pair["item_text"]
                grounding = ground_item(item_text, item2id, grounder, diagnostics)
                if grounding is None:
                    diagnostics["grounding_failures"] += 1
                    diagnostics["unknown_items"].append(
                        {
                            "source_file": str(source_path),
                            "trajectory_id": f"{source_path.name}:{record_id}",
                            "user_id": user_id,
                            "step_index": int(pair["step_index"]),
                            "item_text": item_text,
                        }
                    )
                    continue

                item_id = grounding["item_id"]
                validate_matrix_indices(
                    reward_matrix, user_id, item_id, source_path, record_id
                )
                transition = build_transition(
                    source_path, record_id, user_id, pair, grounding, reward_matrix
                )
                terminal_fallback = bool(transition["terminal_failure"] and transition["action_replaced"])
                if terminal_fallback:
                    diagnostics["terminal_fallback_transitions"] += 1
                    if terminal_fallback_policy == "drop":
                        diagnostics["terminal_fallback_transitions_dropped"] += 1
                        continue
                    if terminal_fallback_policy == "reward_minus1000":
                        transition["reward"] = -1000.0
                        transition["world_model_reward_override"] = -1000.0
                        transition["demo_reward_override_source"] = "terminal_fallback_observed_reward"
                        diagnostics["terminal_fallback_reward_overrides"] += 1
                    if terminal_fallback_policy == "reward_zero":
                        transition["reward"] = 0.0
                        transition["world_model_reward_override"] = 0.0
                        transition["demo_reward_override_source"] = "terminal_fallback_zero_reward"
                        diagnostics["terminal_fallback_reward_overrides"] += 1
                if transition["terminal_failure"]:
                    diagnostics["terminal_failures"] += 1
                record_transitions.append(transition)

            if record_transitions:
                record_transitions[-1]["done"] = True
                transitions.extend(record_transitions)
            else:
                diagnostics["skipped_records"] += 1

    diagnostics["transitions_written"] = len(transitions)
    return {
        "dataset": dataset,
        "split": split,
        "grounding_similarity": grounder.grounding_similarity if grounder is not None else "exact",
        "terminal_fallback_policy": terminal_fallback_policy,
        "source_files": [str(path) for path in trajectory_paths],
        "transitions": transitions,
        "diagnostics": diagnostics,
    }


def write_buffer(path, buffer):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(buffer, f)


def compact_diagnostics(diagnostics, preview_limit=5):
    compact = dict(diagnostics)
    unknown_items = compact.get("unknown_items", [])
    compact["unknown_items_preview"] = unknown_items[:preview_limit]
    compact["unknown_items"] = len(unknown_items)
    return compact


def main():
    args = parse_args()
    try:
        ilrec_root = args.ilrec_root.resolve()
        trajectory_paths = resolve_trajectory_paths(
            ilrec_root, args.dataset, args.trajectory, args.traj_glob
        )
        output_path = args.output or default_output_path(args.dataset)
        grounder = None
        if args.embedding_fallback:
            grounder = EmbeddingGrounder(
                args.dataset,
                ilrec_root,
                embedding_path=args.embedding_path,
                model_path=args.model_path,
                batch_size=args.grounding_batch_size,
                device=args.grounding_device,
                grounding_similarity=args.grounding_similarity,
                cache_path=args.grounding_cache_path
                or default_grounding_cache_path(ilrec_root, args.dataset),
            )
        buffer = build_buffer(
            args.dataset,
            ilrec_root,
            args.split,
            trajectory_paths,
            grounder=grounder,
            terminal_fallback_policy=args.terminal_fallback_policy,
        )
        if grounder is not None:
            grounder.save_cache()
        write_buffer(output_path, buffer)
        print(
            json.dumps(
                {
                    "output": str(output_path),
                    "dataset": buffer["dataset"],
                    "split": buffer["split"],
                    "source_files": len(buffer["source_files"]),
                    "diagnostics": compact_diagnostics(buffer["diagnostics"]),
                },
                indent=2,
                sort_keys=True,
            )
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
