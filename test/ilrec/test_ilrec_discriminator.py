import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TestILRecDiscriminator(unittest.TestCase):
    def test_forward_shape_and_probability_range(self):
        from core.policy.discriminator import TransitionDiscriminator

        torch.manual_seed(0)
        discriminator = TransitionDiscriminator(input_dim=5, hidden_sizes=(8,))
        probs = discriminator(torch.randn(7, 5))

        self.assertEqual(tuple(probs.shape), (7,))
        self.assertTrue(torch.all(probs >= 0.0))
        self.assertTrue(torch.all(probs <= 1.0))

    def test_bce_loss_uses_expert_zero_and_policy_one_labels(self):
        from core.policy.discriminator import discriminator_bce_loss

        probs = torch.tensor([0.2, 0.8, 0.4, 0.6])
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0])

        loss = discriminator_bce_loss(probs, labels)

        self.assertTrue(torch.allclose(loss, F.binary_cross_entropy(probs, labels)))

    def test_irl_reward_is_finite_and_rewards_expert_like_samples(self):
        from core.policy.discriminator import discriminator_irl_reward

        probs = torch.tensor([0.0, 1e-6, 0.5, 1.0])
        rewards = discriminator_irl_reward(probs, eps=1e-6)

        self.assertTrue(torch.all(torch.isfinite(rewards)))
        self.assertGreater(rewards[0].item(), rewards[2].item())
        self.assertGreater(rewards[2].item(), rewards[3].item())

    def test_separable_toy_data_converges(self):
        from core.policy.discriminator import TransitionDiscriminator

        torch.manual_seed(7)
        expert = torch.randn(32, 3) * 0.05 - 1.0
        policy = torch.randn(32, 3) * 0.05 + 1.0
        features = torch.cat([expert, policy], dim=0)
        labels = torch.cat([torch.zeros(32), torch.ones(32)], dim=0)

        discriminator = TransitionDiscriminator(input_dim=3, hidden_sizes=(8,))
        optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.05)
        for _ in range(120):
            optimizer.zero_grad()
            loss = discriminator.bce_loss(features, labels)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            predictions = (discriminator(features) >= 0.5).float()
            accuracy = (predictions == labels).float().mean().item()

        self.assertGreaterEqual(accuracy, 0.95)


if __name__ == "__main__":
    unittest.main()
