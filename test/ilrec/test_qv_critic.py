import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TestQVCritic(unittest.TestCase):
    def test_td_target_uses_target_value_and_done_mask(self):
        from core.policy.qv_critic import compute_td_targets

        rewards = torch.tensor([1.0, 2.0])
        next_values = torch.tensor([10.0, 10.0])
        dones = torch.tensor([False, True])

        targets = compute_td_targets(rewards, next_values, dones, discount=0.5)

        self.assertTrue(torch.allclose(targets, torch.tensor([6.0, 2.0])))

    def test_demo_weighted_critic_loss_applies_sample_weights(self):
        from core.policy.qv_critic import weighted_critic_loss

        predictions = torch.tensor([1.0, 3.0])
        targets = torch.tensor([2.0, 1.0])
        weights = torch.tensor([1.0, 2.0])

        loss = weighted_critic_loss(predictions, targets, weights)

        self.assertAlmostEqual(loss.item(), 4.5)

    def test_target_network_update_is_deterministic(self):
        from core.policy.qv_critic import StateQVCritic, update_target_critic

        source = StateQVCritic(input_dim=2, num_actions=2)
        target = StateQVCritic(input_dim=2, num_actions=2)
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.fill_(2.0)
            for parameter in target.parameters():
                parameter.fill_(0.0)

        update_target_critic(target, source, tau=0.25)

        for parameter in target.parameters():
            self.assertTrue(torch.allclose(parameter, torch.full_like(parameter, 0.5)))

    def test_qv_training_step_returns_losses(self):
        from core.policy.qv_critic import StateQVCritic, qv_critic_training_step

        torch.manual_seed(0)
        critic = StateQVCritic(input_dim=2, num_actions=3)
        target = StateQVCritic(input_dim=2, num_actions=3)
        target.load_state_dict(critic.state_dict())
        optimizer = torch.optim.Adam(critic.parameters(), lr=0.01)

        result = qv_critic_training_step(
            critic,
            target,
            optimizer,
            state_features=torch.randn(4, 2),
            actions=torch.tensor([0, 1, 2, 1]),
            rewards=torch.tensor([1.0, 2.0, 3.0, 4.0]),
            next_state_features=torch.randn(4, 2),
            dones=torch.tensor([False, False, True, False]),
            discount=0.5,
            sample_weights=torch.tensor([1.0, 2.0, 1.0, 1.0]),
        )

        self.assertGreater(result.total_loss.item(), 0.0)
        self.assertGreaterEqual(result.q_loss.item(), 0.0)
        self.assertGreaterEqual(result.v_loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
