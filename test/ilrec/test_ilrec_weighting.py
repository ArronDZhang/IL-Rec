import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TestILRecDemonstrationWeighting(unittest.TestCase):
    def test_environment_weights_are_finite(self):
        from core.policy.ilrec_a2c import compute_env_weights

        advantages = torch.tensor([-2.0, 0.0, 2.0])
        weights = compute_env_weights(advantages, beta=2.0)

        self.assertTrue(torch.all(torch.isfinite(weights)))
        self.assertTrue(torch.allclose(weights, torch.exp(advantages / 2.0)))

    def test_irl_weights_are_finite_for_extreme_discriminator_outputs(self):
        from core.policy.ilrec_a2c import compute_irl_weights

        probs = torch.tensor([0.0, 1e-12, 0.5, 1.0])
        weights = compute_irl_weights(probs, irl_gamma=0.5, eps=1e-6)

        self.assertTrue(torch.all(torch.isfinite(weights)))
        self.assertTrue(torch.all(weights >= 0.0))
        self.assertGreater(weights[0].item(), weights[2].item())
        self.assertLess(weights[-1].item(), 1e-2)

    def test_fused_weights_are_normalized_to_unit_mean(self):
        from core.policy.ilrec_a2c import compute_demo_weights

        result = compute_demo_weights(
            demo_advantages=torch.tensor([0.0, 1.0, 2.0]),
            discriminator_probs=torch.tensor([0.2, 0.4, 0.6]),
            beta=1.0,
            alpha=0.25,
            irl_gamma=0.5,
        )

        self.assertAlmostEqual(result.weights.mean().item(), 1.0, places=6)
        self.assertEqual(tuple(result.w_env.shape), (3,))
        self.assertEqual(tuple(result.w_irl.shape), (3,))
        self.assertEqual(tuple(result.weights.shape), (3,))

    def test_weight_clipping_bounds_components_and_fused_weights(self):
        from core.policy.ilrec_a2c import compute_demo_weights

        result = compute_demo_weights(
            demo_advantages=torch.tensor([-100.0, 100.0]),
            discriminator_probs=torch.tensor([1e-12, 1.0]),
            beta=0.5,
            alpha=0.5,
            irl_gamma=2.0,
            clip_min=0.25,
            clip_max=4.0,
            normalize=False,
            eps=1e-6,
        )

        for tensor in (result.w_env, result.w_irl, result.weights):
            self.assertTrue(torch.all(tensor >= 0.25))
            self.assertTrue(torch.all(tensor <= 4.0))

    def test_invalid_parameters_raise_clear_errors(self):
        from core.policy.ilrec_a2c import compute_demo_weights

        with self.assertRaisesRegex(ValueError, "beta"):
            compute_demo_weights(
                torch.tensor([1.0]),
                torch.tensor([0.5]),
                beta=0.0,
                alpha=0.5,
                irl_gamma=1.0,
            )
        with self.assertRaisesRegex(ValueError, "alpha"):
            compute_demo_weights(
                torch.tensor([1.0]),
                torch.tensor([0.5]),
                beta=1.0,
                alpha=1.5,
                irl_gamma=1.0,
            )


if __name__ == "__main__":
    unittest.main()
