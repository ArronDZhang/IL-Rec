import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "run_Policy_ILRec.py"
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_runner():
    spec = importlib.util.spec_from_file_location("run_Policy_ILRec", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestRunPolicyILRec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def parse_minimal(self, env):
        return self.runner.parse_args(
            [
                "--env",
                env,
                "--demo-buffer",
                "demo_gpt35.pkl",
            ]
        )

    def test_amazon_defaults_match_public_fb_standard(self):
        args = self.parse_minimal("AmazonEnv-v0")

        self.assertEqual(args.train_episodes, 100000)
        self.assertEqual(args.discount, 0.5)
        self.assertEqual(args.train_action_selection, "sample")
        self.assertEqual(args.eval_action_selection, "sample")
        self.assertFalse(hasattr(args, "eval_" + "action_" + "mask"))
        self.assertFalse(hasattr(args, "eval_" + "repeat_" + "penalty"))
        self.assertIsNone(args.summary_json)
        self.assertEqual(args.train_env_source, "precomputed_matpre")
        self.assertEqual(args.eval_env_source, "precomputed_matpre")
        self.assertEqual(args.state_action_feature_mode, "llama")
        self.assertEqual(args.state_tracker_type, "roler_attention")
        self.assertEqual(args.mixed_replay_sampling, "global_priority")
        self.assertEqual(args.mixed_replay_env_priority_scale, 0.05)
        self.assertEqual(args.train_advantage_clip, 5.0)
        self.assertTrue(args.train_normalize_advantages)
        self.assertEqual(args.train_actor_row_norm_project, 10.0)
        self.assertEqual(args.train_actor_bias_clamp, 5.0)
        self.assertEqual(args.eval_episodes, 100)
        self.assertEqual(args.policy_logit_clamp, 1.0)
        self.assertEqual(args.policy_logit_clamp_mode, "tanh")

    def test_steam_defaults_use_dataset_specific_clamp(self):
        args = self.parse_minimal("SteamEnv-v0")

        self.assertEqual(args.policy_logit_clamp, 15.0)
        self.assertEqual(args.discount, 0.5)
        self.assertEqual(args.beta, 10.0)
        self.assertEqual(args.lambda_imit, 0.25)

    def test_only_tanh_logit_clamp_is_supported(self):
        logits = torch.tensor([-100.0, 0.0, 100.0])

        clamped = self.runner.apply_policy_logit_clamp(logits, clamp_value=5.0, mode="tanh")

        self.assertTrue(torch.all(clamped <= 5.0))
        self.assertTrue(torch.all(clamped >= -5.0))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self.runner.apply_policy_logit_clamp(logits, clamp_value=5.0, mode="hard")

    def test_fb_action_selection_samples_from_unmasked_logits(self):
        torch.manual_seed(0)

        action = self.runner.select_evaluation_action(
            torch.tensor([100.0, -100.0, -100.0]),
            valid_actions=3,
            action_selection="sample",
        )

        self.assertEqual(action, 0)


if __name__ == "__main__":
    unittest.main()
