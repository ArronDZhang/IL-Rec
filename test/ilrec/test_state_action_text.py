import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TestStateActionText(unittest.TestCase):
    def test_render_state_action_text_is_deterministic(self):
        from core.policy.state_action_text import render_state_action_text

        kwargs = {
            "dataset": "amazon",
            "user_id": 42,
            "action_id": 9,
            "history_actions": [1, 2],
            "history_rewards": [4.5, 0.0],
            "item_text": "Space exploration strategy game",
        }

        first = render_state_action_text(**kwargs)
        second = render_state_action_text(**kwargs)

        self.assertEqual(first, second)
        self.assertIn("template=ilrec_state_action_v1", first)
        self.assertIn("dataset=amazon", first)
        self.assertIn("user_id=42", first)
        self.assertIn("history=[item_id=1 reward=4.500000; item_id=2 reward=0.000000]", first)
        self.assertIn("candidate=item_id=9 text=Space exploration strategy game", first)

    def test_render_state_action_text_handles_empty_history(self):
        from core.policy.state_action_text import render_state_action_text

        text = render_state_action_text(dataset="steam", user_id=0, action_id=0)

        self.assertIn("dataset=steam", text)
        self.assertIn("user_id=0", text)
        self.assertIn("history=<empty>", text)
        self.assertIn("candidate=item_id=0 text=<unknown>", text)

    def test_render_state_action_text_supports_boundary_ids_and_lookup(self):
        from core.policy.state_action_text import render_state_action_text

        text = render_state_action_text(
            dataset="amazon",
            user_id=999999,
            action_id=888888,
            history_actions=[0, 888887],
            item_lookup={888888: "Boundary item"},
        )

        self.assertIn("user_id=999999", text)
        self.assertIn("item_id=888887 reward=0.000000", text)
        self.assertIn("candidate=item_id=888888 text=Boundary item", text)

    def test_transition_renderer_uses_same_format_for_expert_and_policy(self):
        from core.policy.state_action_text import render_transition_state_action_text

        expert = {
            "source": "expert",
            "user_id": 3,
            "action_id": 5,
            "history_action_ids": [1],
            "history_rewards": [2.25],
            "item_text": "Detective mystery",
        }
        policy = dict(expert)
        policy["source"] = "policy"

        self.assertEqual(
            render_transition_state_action_text(expert, dataset="amazon"),
            render_transition_state_action_text(policy, dataset="amazon"),
        )

    def test_transition_renderer_rejects_missing_action(self):
        from core.policy.state_action_text import render_transition_state_action_text

        with self.assertRaisesRegex(ValueError, "action_id"):
            render_transition_state_action_text({"user_id": 1}, dataset="amazon")


if __name__ == "__main__":
    unittest.main()
