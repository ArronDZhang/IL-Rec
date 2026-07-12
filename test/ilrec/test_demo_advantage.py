import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TestDemoAdvantage(unittest.TestCase):
    def test_demo_value_fit_reduces_loss(self):
        from core.policy.demo_advantage import fit_demo_value_network

        torch.manual_seed(0)
        features = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float32)
        returns = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

        result = fit_demo_value_network(
            features,
            returns,
            hidden_size=8,
            lr=0.05,
            steps=120,
        )

        self.assertLess(result.final_loss, result.initial_loss)
        with torch.no_grad():
            predictions = result.model(features)
        self.assertEqual(tuple(predictions.shape), (4,))

    def test_demo_q_fit_reduces_loss(self):
        from core.policy.demo_advantage import fit_demo_q_network

        torch.manual_seed(0)
        features = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float32)
        returns = torch.tensor([1.0, 3.0, 5.0, 7.0], dtype=torch.float32)

        result = fit_demo_q_network(
            features,
            returns,
            hidden_size=8,
            lr=0.05,
            steps=120,
        )

        self.assertLess(result.final_loss, result.initial_loss)
        with torch.no_grad():
            predictions = result.model(features)
        self.assertEqual(tuple(predictions.shape), (4,))

    def test_demo_advantage_is_return_minus_v_demo(self):
        from core.policy.demo_advantage import compute_demo_advantages

        advantages = compute_demo_advantages(
            demo_returns=torch.tensor([5.0, 2.0]),
            demo_values=torch.tensor([1.5, -0.5]),
        )

        self.assertTrue(torch.allclose(advantages, torch.tensor([3.5, 2.5])))

    def test_demo_advantage_accepts_explicit_q_demo_values(self):
        from core.policy.demo_advantage import compute_demo_advantages

        advantages = compute_demo_advantages(
            demo_returns=torch.tensor([9.0, 9.0]),
            demo_values=torch.tensor([1.5, -0.5]),
            demo_q_values=torch.tensor([5.0, 2.0]),
        )

        self.assertTrue(torch.allclose(advantages, torch.tensor([3.5, 2.5])))

    def test_shape_mismatch_is_rejected(self):
        from core.policy.demo_advantage import compute_demo_advantages

        with self.assertRaisesRegex(ValueError, "matching shapes"):
            compute_demo_advantages(torch.tensor([1.0, 2.0]), torch.tensor([1.0]))


if __name__ == "__main__":
    unittest.main()
