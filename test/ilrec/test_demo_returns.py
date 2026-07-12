import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class MatrixLookup:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=np.float32)
        self.calls = []

    def __call__(self, user_ids, action_ids):
        self.calls.append((list(user_ids), list(action_ids)))
        return self.matrix[np.asarray(user_ids, dtype=int), np.asarray(action_ids, dtype=int)]


class TestDemoReturns(unittest.TestCase):
    def test_computes_discounted_returns_per_trajectory(self):
        from core.policy.demo_returns import annotate_world_model_demo_returns

        transitions = [
            {"trajectory_id": "a", "user_id": 0, "action_id": 0, "observed_reward": 99.0},
            {"trajectory_id": "a", "user_id": 0, "action_id": 1, "observed_reward": 99.0, "done": True},
            {"trajectory_id": "b", "user_id": 1, "action_id": 0, "observed_reward": -1000.0, "done": True},
        ]
        lookup = MatrixLookup([[4.0, 2.0], [3.0, 1.0]])

        result = annotate_world_model_demo_returns(transitions, lookup, discount=0.5)

        self.assertEqual([round(item["demo_return"], 6) for item in result.transitions], [5.0, 2.0, 3.0])
        self.assertEqual([item["world_model_reward"] for item in result.transitions], [4.0, 2.0, 3.0])
        self.assertEqual(result.diagnostics["trajectory_count"], 2)
        self.assertEqual(result.diagnostics["return_source"], "world_model_lookup")
        self.assertEqual(lookup.calls, [([0, 0, 1], [0, 1, 0])])

    def test_done_splits_repeated_trajectory_ids(self):
        from core.policy.demo_returns import annotate_world_model_demo_returns

        transitions = [
            {"trajectory_id": "a", "user_id": 0, "action_id": 0, "done": True},
            {"trajectory_id": "a", "user_id": 0, "action_id": 1, "done": True},
        ]

        result = annotate_world_model_demo_returns(transitions, MatrixLookup([[4.0, 2.0]]), discount=0.9)

        self.assertEqual([item["demo_return"] for item in result.transitions], [4.0, 2.0])
        self.assertEqual(result.diagnostics["trajectory_count"], 2)

    def test_uses_world_model_reward_instead_of_observed_llm_reward(self):
        from core.policy.demo_returns import annotate_world_model_demo_returns

        transitions = [
            {
                "trajectory_id": "a",
                "user_id": 0,
                "action_id": 0,
                "reward": 123.0,
                "observed_reward": -1000.0,
                "done": True,
            }
        ]

        result = annotate_world_model_demo_returns(transitions, MatrixLookup([[4.5]]), discount=0.5)

        self.assertEqual(result.transitions[0]["world_model_reward"], 4.5)
        self.assertEqual(result.transitions[0]["demo_return"], 4.5)
        self.assertEqual(result.transitions[0]["observed_reward"], -1000.0)

    def test_world_model_reward_override_replaces_lookup_reward(self):
        from core.policy.demo_returns import annotate_world_model_demo_returns

        transitions = [
            {
                "trajectory_id": "a",
                "user_id": 0,
                "action_id": 0,
                "world_model_reward_override": -1000.0,
            },
            {"trajectory_id": "a", "user_id": 0, "action_id": 1, "done": True},
        ]

        result = annotate_world_model_demo_returns(transitions, MatrixLookup([[4.0, 2.0]]), discount=0.5)

        self.assertEqual([item["world_model_reward"] for item in result.transitions], [-1000.0, 2.0])
        self.assertEqual([item["world_model_reward_source"] for item in result.transitions], ["transition_override", "world_model_lookup"])
        self.assertEqual([item["demo_return"] for item in result.transitions], [-999.0, 2.0])
        self.assertEqual(result.diagnostics["override_reward_count"], 1)
        self.assertEqual(result.diagnostics["return_source"], "world_model_lookup_with_transition_overrides")

    def test_rejects_missing_transition_fields(self):
        from core.policy.demo_returns import annotate_world_model_demo_returns

        with self.assertRaisesRegex(ValueError, "action_id"):
            annotate_world_model_demo_returns(
                [{"trajectory_id": "bad", "user_id": 0}],
                MatrixLookup([[1.0]]),
                discount=0.5,
            )


if __name__ == "__main__":
    unittest.main()
