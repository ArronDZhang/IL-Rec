import json
import pickle
import subprocess
import sys
import tempfile
import unittest
import importlib.util
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "build_ilrec_demo_buffer.py"


def load_builder_module():
    spec = importlib.util.spec_from_file_location("build_ilrec_demo_buffer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def write_fixture(root):
    env_root = root / "env" / "amazon"
    env_root.mkdir(parents=True, exist_ok=True)
    write_json(
        env_root / "datamaps.json",
        {
            "item2id_dict": {
                "Known Item": 0,
                "Boundary Item": 1,
                "Terminal Item": 2,
            },
            "id2item_dict": {
                "0": "Known Item",
                "1": "Boundary Item",
                "2": "Terminal Item",
            },
        },
    )
    np.save(
        env_root / "amazon_train.npy",
        np.array(
            [
                [4.25, 3.5, 2.0],
                [1.0, 4.0, 4.75],
            ],
            dtype=np.float32,
        ),
    )


def run_builder(ilrec_root, traj_paths, output_path, extra_args=None):
    command = [
        sys.executable,
        str(SCRIPT),
        "--dataset",
        "amazon",
        "--ilrec-root",
        str(ilrec_root),
        "--output",
        str(output_path),
    ]
    if extra_args:
        command.extend(extra_args)
    for traj_path in traj_paths:
        command.extend(["--trajectory", str(traj_path)])
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True)


class TestBuildILRecDemoBuffer(unittest.TestCase):
    def test_builds_ordered_transitions_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            ilrec_root = Path(tmp) / "il-rec"
            write_fixture(ilrec_root)
            first_traj = ilrec_root / "trajs_agent" / "first.json"
            second_traj = ilrec_root / "trajs_agent" / "second.json"
            write_json(
                first_traj,
                {
                    "u0": {
                        "userid": 0,
                        "traj_by_line": [
                            "Thought 1: start",
                            "Action 1: recommend[Known Item]",
                            "Observation 1: Episode continue, reward=9.5",
                            "Action 2: recommend[Missing Item]",
                            "Observation 2: Episode continue, reward=1.0",
                            "Action 3: recommend[Terminal Item]",
                            "Observation 3: Episode finished, User Stop, reward=-1000.000",
                        ],
                    }
                },
            )
            write_json(
                second_traj,
                {
                    "u1": {
                        "userid": 1,
                        "traj_by_line": [
                            "Action 1: recommend[Boundary Item]",
                            "Observation 1: Episode continue, reward=0",
                        ],
                    }
                },
            )
            output_path = Path(tmp) / "demo_gpt35.pkl"

            result = run_builder(ilrec_root, [first_traj, second_traj], output_path)

            self.assertEqual(result.returncode, 0, result.stderr)
            with output_path.open("rb") as f:
                buffer = pickle.load(f)

        transitions = buffer["transitions"]
        diagnostics = buffer["diagnostics"]
        self.assertEqual(buffer["dataset"], "amazon")
        self.assertEqual(buffer["split"], "train")
        self.assertEqual(
            [transition["raw_item_text"] for transition in transitions],
            ["Known Item", "Terminal Item", "Boundary Item"],
        )
        self.assertEqual([transition["user_id"] for transition in transitions], [0, 0, 1])
        self.assertEqual([transition["action_id"] for transition in transitions], [0, 2, 1])
        self.assertEqual([transition["step_index"] for transition in transitions], [0, 2, 0])
        self.assertEqual([transition["trajectory_id"] for transition in transitions], ["first.json:u0", "first.json:u0", "second.json:u1"])
        self.assertEqual([transition["grounding_status"] for transition in transitions], ["exact", "exact", "exact"])
        self.assertEqual([transition["done"] for transition in transitions], [False, True, True])
        self.assertEqual([transition["reward"] for transition in transitions], [4.25, 2.0, 4.0])
        self.assertEqual([transition["observed_reward"] for transition in transitions], [9.5, -1000.0, 0.0])
        self.assertEqual([transition["metric_reward"] for transition in transitions], [9.5, 0.0, 0.0])
        self.assertEqual(transitions[0]["obs"], [0])
        self.assertEqual(transitions[0]["next_obs"], [0])

        self.assertEqual(diagnostics["records_seen"], 2)
        self.assertEqual(diagnostics["actions_seen"], 4)
        self.assertEqual(diagnostics["transitions_written"], 3)
        self.assertEqual(diagnostics["grounding_failures"], 1)
        self.assertEqual(diagnostics["terminal_failures"], 1)
        self.assertEqual(diagnostics["skipped_missing_reward"], 0)
        self.assertEqual(diagnostics["unknown_items"][0]["item_text"], "Missing Item")

    def test_empty_trajectory_file_writes_empty_buffer(self):
        with tempfile.TemporaryDirectory() as tmp:
            ilrec_root = Path(tmp) / "il-rec"
            write_fixture(ilrec_root)
            traj_path = ilrec_root / "trajs_agent" / "empty.json"
            write_json(traj_path, {})
            output_path = Path(tmp) / "demo_gpt35.pkl"

            result = run_builder(ilrec_root, [traj_path], output_path)

            self.assertEqual(result.returncode, 0, result.stderr)
            with output_path.open("rb") as f:
                buffer = pickle.load(f)

        self.assertEqual(buffer["transitions"], [])
        self.assertEqual(buffer["diagnostics"]["records_seen"], 0)
        self.assertEqual(buffer["diagnostics"]["transitions_written"], 0)

    def test_uses_replacement_recommendation_from_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            ilrec_root = Path(tmp) / "il-rec"
            write_fixture(ilrec_root)
            traj_path = ilrec_root / "trajs_agent" / "fallback.json"
            write_json(
                traj_path,
                {
                    "u0": {
                        "userid": 0,
                        "traj_by_line": [
                            "Action 1: recommend[Unavailable Item]",
                            "Observation 1: [Unavailable Item] can not be recommened, instead, recommend[Boundary Item]",
                            "Observation 1: Episode continue, reward=3.5",
                        ],
                    }
                },
            )
            output_path = Path(tmp) / "demo_gpt35.pkl"

            result = run_builder(ilrec_root, [traj_path], output_path)

            self.assertEqual(result.returncode, 0, result.stderr)
            with output_path.open("rb") as f:
                buffer = pickle.load(f)

        transitions = buffer["transitions"]
        diagnostics = buffer["diagnostics"]
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]["raw_item_text"], "Boundary Item")
        self.assertEqual(transitions[0]["attempted_item_text"], "Unavailable Item")
        self.assertTrue(transitions[0]["action_replaced"])
        self.assertEqual(transitions[0]["action_id"], 1)
        self.assertEqual(transitions[0]["reward"], 3.5)
        self.assertEqual(transitions[0]["observed_reward"], 3.5)
        self.assertEqual(diagnostics["action_replacements"], 1)
        self.assertEqual(diagnostics["grounding_failures"], 0)

    def test_terminal_fallback_policy_drop_removes_transition(self):
        with tempfile.TemporaryDirectory() as tmp:
            ilrec_root = Path(tmp) / "il-rec"
            write_fixture(ilrec_root)
            traj_path = ilrec_root / "trajs_agent" / "fallback_terminal.json"
            write_json(
                traj_path,
                {
                    "u0": {
                        "userid": 0,
                        "traj_by_line": [
                            "Action 1: recommend[Known Item]",
                            "Observation 1: Episode continue, reward=4.25",
                            "Action 2: recommend[Unavailable Item]",
                            "Observation 2: [Unavailable Item] can not be recommened, instead, recommend[Boundary Item]",
                            "Observation 2: Episode finished, User Stop, reward=-1000.000",
                        ],
                    }
                },
            )
            output_path = Path(tmp) / "demo_gpt35.pkl"

            result = run_builder(
                ilrec_root,
                [traj_path],
                output_path,
                extra_args=["--terminal-fallback-policy", "drop"],
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            with output_path.open("rb") as f:
                buffer = pickle.load(f)

        transitions = buffer["transitions"]
        diagnostics = buffer["diagnostics"]
        self.assertEqual(buffer["terminal_fallback_policy"], "drop")
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]["raw_item_text"], "Known Item")
        self.assertTrue(transitions[0]["done"])
        self.assertEqual(diagnostics["terminal_fallback_transitions"], 1)
        self.assertEqual(diagnostics["terminal_fallback_transitions_dropped"], 1)
        self.assertEqual(diagnostics["terminal_fallback_reward_overrides"], 0)
        self.assertEqual(diagnostics["terminal_failures"], 0)
        self.assertEqual(diagnostics["transitions_written"], 1)

    def test_terminal_fallback_policy_reward_minus1000_marks_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            ilrec_root = Path(tmp) / "il-rec"
            write_fixture(ilrec_root)
            traj_path = ilrec_root / "trajs_agent" / "fallback_terminal.json"
            write_json(
                traj_path,
                {
                    "u0": {
                        "userid": 0,
                        "traj_by_line": [
                            "Action 1: recommend[Unavailable Item]",
                            "Observation 1: [Unavailable Item] can not be recommened, instead, recommend[Boundary Item]",
                            "Observation 1: Episode finished, User Stop, reward=-1000.000",
                        ],
                    }
                },
            )
            output_path = Path(tmp) / "demo_gpt35.pkl"

            result = run_builder(
                ilrec_root,
                [traj_path],
                output_path,
                extra_args=["--terminal-fallback-policy", "reward_minus1000"],
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            with output_path.open("rb") as f:
                buffer = pickle.load(f)

        transitions = buffer["transitions"]
        diagnostics = buffer["diagnostics"]
        self.assertEqual(buffer["terminal_fallback_policy"], "reward_minus1000")
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]["raw_item_text"], "Boundary Item")
        self.assertEqual(transitions[0]["attempted_item_text"], "Unavailable Item")
        self.assertTrue(transitions[0]["action_replaced"])
        self.assertEqual(transitions[0]["observed_reward"], -1000.0)
        self.assertEqual(transitions[0]["reward"], -1000.0)
        self.assertEqual(transitions[0]["world_model_reward_override"], -1000.0)
        self.assertEqual(
            transitions[0]["demo_reward_override_source"],
            "terminal_fallback_observed_reward",
        )
        self.assertEqual(diagnostics["terminal_fallback_transitions"], 1)
        self.assertEqual(diagnostics["terminal_fallback_transitions_dropped"], 0)
        self.assertEqual(diagnostics["terminal_fallback_reward_overrides"], 1)
        self.assertEqual(diagnostics["terminal_failures"], 1)

    def test_terminal_fallback_policy_reward_zero_marks_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            ilrec_root = Path(tmp) / "il-rec"
            write_fixture(ilrec_root)
            traj_path = ilrec_root / "trajs_agent" / "fallback_terminal.json"
            write_json(
                traj_path,
                {
                    "u0": {
                        "userid": 0,
                        "traj_by_line": [
                            "Action 1: recommend[Unavailable Item]",
                            "Observation 1: [Unavailable Item] can not be recommened, instead, recommend[Boundary Item]",
                            "Observation 1: Episode finished, User Stop, reward=-1000.000",
                        ],
                    }
                },
            )
            output_path = Path(tmp) / "demo_gpt35.pkl"

            result = run_builder(
                ilrec_root,
                [traj_path],
                output_path,
                extra_args=["--terminal-fallback-policy", "reward_zero"],
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            with output_path.open("rb") as f:
                buffer = pickle.load(f)

        transitions = buffer["transitions"]
        diagnostics = buffer["diagnostics"]
        self.assertEqual(buffer["terminal_fallback_policy"], "reward_zero")
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]["raw_item_text"], "Boundary Item")
        self.assertEqual(transitions[0]["observed_reward"], -1000.0)
        self.assertEqual(transitions[0]["reward"], 0.0)
        self.assertEqual(transitions[0]["world_model_reward_override"], 0.0)
        self.assertEqual(
            transitions[0]["demo_reward_override_source"],
            "terminal_fallback_zero_reward",
        )
        self.assertEqual(diagnostics["terminal_fallback_transitions"], 1)
        self.assertEqual(diagnostics["terminal_fallback_transitions_dropped"], 0)
        self.assertEqual(diagnostics["terminal_fallback_reward_overrides"], 1)
        self.assertEqual(diagnostics["terminal_failures"], 1)

    def test_reports_invalid_trajectory_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            ilrec_root = Path(tmp) / "il-rec"
            write_fixture(ilrec_root)
            traj_path = ilrec_root / "trajs_agent" / "bad.json"
            write_json(traj_path, [])
            output_path = Path(tmp) / "demo_gpt35.pkl"

            result = run_builder(ilrec_root, [traj_path], output_path)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must contain a JSON object", result.stderr)

    def test_embedding_fallback_uses_tiny_tensor_and_prefers_exact_match(self):
        builder = load_builder_module()

        class FakeQueryEncoder:
            def __init__(self):
                self.calls = []

            def encode(self, item_texts):
                self.calls.extend(item_texts)
                return torch.tensor([[2.1, 0.0]], dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmp:
            ilrec_root = Path(tmp) / "il-rec"
            write_fixture(ilrec_root)
            embedding_path = ilrec_root / "env" / "amazon" / "amazon_embedding_task.pt"
            torch.save(
                {
                    "embeddings": torch.tensor(
                        [[0.0, 1.0], [2.0, 0.0], [0.5, 0.5]],
                        dtype=torch.float32,
                    ),
                    "indexs": torch.tensor([0, 1, 2], dtype=torch.float32),
                },
                embedding_path,
            )
            traj_path = ilrec_root / "trajs_agent" / "fallback.json"
            write_json(
                traj_path,
                {
                    "u0": {
                        "userid": 0,
                        "traj_by_line": [
                            "Action 1: recommend[Known Item]",
                            "Observation 1: Episode continue, reward=4.25",
                            "Action 2: recommend[Semantically Close Item]",
                            "Observation 2: Episode continue, reward=3.0",
                        ],
                    }
                },
            )
            fake_encoder = FakeQueryEncoder()
            grounder = builder.EmbeddingGrounder(
                "amazon",
                ilrec_root,
                embedding_path=embedding_path,
                query_encoder=fake_encoder,
            )

            buffer = builder.build_buffer(
                "amazon",
                ilrec_root,
                "train",
                [traj_path],
                grounder=grounder,
            )

        transitions = buffer["transitions"]
        diagnostics = buffer["diagnostics"]
        self.assertEqual(fake_encoder.calls, ["Semantically Close Item"])
        self.assertEqual([transition["action_id"] for transition in transitions], [0, 1])
        self.assertEqual(
            [transition["grounding_status"] for transition in transitions],
            ["exact", "embedding"],
        )
        self.assertEqual(transitions[1]["raw_item_text"], "Semantically Close Item")
        self.assertEqual(transitions[1]["grounded_item_text"], "Boundary Item")
        self.assertEqual(transitions[1]["grounding_metric"], "cosine")
        self.assertAlmostEqual(transitions[1]["grounding_similarity"], 1.0, places=5)
        self.assertEqual(diagnostics["exact_groundings"], 1)
        self.assertEqual(diagnostics["embedding_groundings"], 1)
        self.assertEqual(diagnostics["grounding_failures"], 0)

    def test_embedding_fallback_uses_cosine_similarity_not_euclidean_distance(self):
        builder = load_builder_module()

        class FakeQueryEncoder:
            def encode(self, item_texts):
                self.item_texts = list(item_texts)
                return torch.tensor([[10.0, 0.0]], dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmp:
            ilrec_root = Path(tmp) / "il-rec"
            write_fixture(ilrec_root)
            embedding_path = ilrec_root / "env" / "amazon" / "amazon_embedding_task.pt"
            torch.save(
                {
                    "embeddings": torch.tensor(
                        [
                            [9.0, 1.0],
                            [100.0, 0.0],
                            [0.0, 1.0],
                        ],
                        dtype=torch.float32,
                    ),
                    "indexs": torch.tensor([0, 1, 2], dtype=torch.float32),
                },
                embedding_path,
            )
            encoder = FakeQueryEncoder()
            grounder = builder.EmbeddingGrounder(
                "amazon",
                ilrec_root,
                embedding_path=embedding_path,
                query_encoder=encoder,
                grounding_similarity="cosine",
            )

            result = grounder.ground("Cosine Match")

        self.assertEqual(encoder.item_texts, ["Cosine Match"])
        self.assertEqual(result["item_id"], 1)
        self.assertEqual(result["matched_item_text"], "Boundary Item")
        self.assertAlmostEqual(result["similarity"], 1.0, places=5)
        self.assertEqual(result["metric"], "cosine")

    def test_embedding_grounder_batches_uncached_items_and_reuses_hits(self):
        builder = load_builder_module()

        class FakeQueryEncoder:
            def __init__(self):
                self.calls = []

            def encode(self, item_texts):
                self.calls.append(list(item_texts))
                vectors = {
                    "Close Known": [0.0, 1.1],
                    "Close Boundary": [2.1, 0.0],
                }
                return torch.tensor([vectors[item] for item in item_texts], dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmp:
            ilrec_root = Path(tmp) / "il-rec"
            write_fixture(ilrec_root)
            embedding_path = ilrec_root / "env" / "amazon" / "amazon_embedding_task.pt"
            torch.save(
                {
                    "embeddings": torch.tensor(
                        [[0.0, 1.0], [2.0, 0.0], [0.5, 0.5]],
                        dtype=torch.float32,
                    ),
                    "indexs": torch.tensor([0, 1, 2], dtype=torch.float32),
                },
                embedding_path,
            )
            fake_encoder = FakeQueryEncoder()
            grounder = builder.EmbeddingGrounder(
                "amazon",
                ilrec_root,
                embedding_path=embedding_path,
                query_encoder=fake_encoder,
            )

            first = grounder.ground_many(["Close Known", "Close Boundary", "Close Known"])
            second = grounder.ground_many(["Close Boundary"])

        self.assertEqual(fake_encoder.calls, [["Close Known", "Close Boundary"]])
        self.assertEqual(first["Close Known"]["item_id"], 0)
        self.assertEqual(first["Close Boundary"]["item_id"], 1)
        self.assertEqual(second["Close Boundary"]["item_id"], 1)

    def test_persistent_grounding_cache_avoids_reencoding_on_rerun(self):
        builder = load_builder_module()

        class FakeQueryEncoder:
            def __init__(self):
                self.calls = []

            def encode(self, item_texts):
                self.calls.append(list(item_texts))
                return torch.tensor([[2.1, 0.0] for _ in item_texts], dtype=torch.float32)

        class FailingQueryEncoder:
            def encode(self, item_texts):
                raise AssertionError(f"cache miss for {item_texts}")

        with tempfile.TemporaryDirectory() as tmp:
            ilrec_root = Path(tmp) / "il-rec"
            write_fixture(ilrec_root)
            embedding_path = ilrec_root / "env" / "amazon" / "amazon_embedding_task.pt"
            cache_path = ilrec_root / "env" / "amazon" / "grounding_cache.json"
            torch.save(
                {
                    "embeddings": torch.tensor(
                        [[0.0, 1.0], [2.0, 0.0], [0.5, 0.5]],
                        dtype=torch.float32,
                    ),
                    "indexs": torch.tensor([0, 1, 2], dtype=torch.float32),
                },
                embedding_path,
            )
            traj_path = ilrec_root / "trajs_agent" / "fallback.json"
            write_json(
                traj_path,
                {
                    "u0": {
                        "userid": 0,
                        "traj_by_line": [
                            "Action 1: recommend[Semantically Close Item]",
                            "Observation 1: Episode continue, reward=3.0",
                        ],
                    }
                },
            )
            fake_encoder = FakeQueryEncoder()
            first_grounder = builder.EmbeddingGrounder(
                "amazon",
                ilrec_root,
                embedding_path=embedding_path,
                query_encoder=fake_encoder,
                cache_path=cache_path,
            )
            first_buffer = builder.build_buffer(
                "amazon",
                ilrec_root,
                "train",
                [traj_path],
                grounder=first_grounder,
            )
            first_grounder.save_cache()

            second_grounder = builder.EmbeddingGrounder(
                "amazon",
                ilrec_root,
                embedding_path=embedding_path,
                query_encoder=FailingQueryEncoder(),
                cache_path=cache_path,
            )
            second_buffer = builder.build_buffer(
                "amazon",
                ilrec_root,
                "train",
                [traj_path],
                grounder=second_grounder,
            )

            self.assertEqual(fake_encoder.calls, [["Semantically Close Item"]])
            self.assertTrue(cache_path.exists())
            self.assertEqual(first_buffer["transitions"][0]["action_id"], 1)
            self.assertEqual(second_buffer["transitions"][0]["action_id"], 1)
            self.assertEqual(first_buffer["grounding_similarity"], "cosine")
            self.assertEqual(second_buffer["transitions"][0]["grounding_metric"], "cosine")

    def test_grounding_cache_invalidates_when_similarity_method_changes(self):
        builder = load_builder_module()

        class FakeQueryEncoder:
            def __init__(self, vector):
                self.vector = vector
                self.calls = []

            def encode(self, item_texts):
                self.calls.append(list(item_texts))
                return torch.tensor([self.vector for _ in item_texts], dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmp:
            ilrec_root = Path(tmp) / "il-rec"
            write_fixture(ilrec_root)
            embedding_path = ilrec_root / "env" / "amazon" / "amazon_embedding_task.pt"
            cache_path = ilrec_root / "env" / "amazon" / "grounding_cache.json"
            torch.save(
                {
                    "embeddings": torch.tensor(
                        [[9.0, 1.0], [100.0, 0.0], [0.0, 1.0]],
                        dtype=torch.float32,
                    ),
                    "indexs": torch.tensor([0, 1, 2], dtype=torch.float32),
                },
                embedding_path,
            )
            first_encoder = FakeQueryEncoder([10.0, 0.0])
            first_grounder = builder.EmbeddingGrounder(
                "amazon",
                ilrec_root,
                embedding_path=embedding_path,
                query_encoder=first_encoder,
                cache_path=cache_path,
                grounding_similarity="euclidean",
            )
            first_grounder.ground("Metric Change")
            first_grounder.save_cache()

            second_encoder = FakeQueryEncoder([10.0, 0.0])
            second_grounder = builder.EmbeddingGrounder(
                "amazon",
                ilrec_root,
                embedding_path=embedding_path,
                query_encoder=second_encoder,
                cache_path=cache_path,
                grounding_similarity="cosine",
            )
            result = second_grounder.ground("Metric Change")

        self.assertEqual(first_encoder.calls, [["Metric Change"]])
        self.assertEqual(second_encoder.calls, [["Metric Change"]])
        self.assertEqual(result["item_id"], 1)
        self.assertEqual(result["metric"], "cosine")


if __name__ == "__main__":
    unittest.main()
