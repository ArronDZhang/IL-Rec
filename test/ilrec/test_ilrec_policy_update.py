import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TestILRecPolicyUpdate(unittest.TestCase):
    def test_combined_loss_backpropagates_to_actor_and_critic_inputs(self):
        from core.policy.ilrec_a2c import compute_ilrec_policy_loss

        action_logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
        values = torch.tensor([0.25, -0.25], requires_grad=True)
        demo_logits = torch.tensor([[0.0, 2.0], [2.0, 0.0]], requires_grad=True)

        loss_parts = compute_ilrec_policy_loss(
            action_logits=action_logits,
            actions=torch.tensor([1, 0]),
            advantages=torch.tensor([1.0, 0.5]),
            values=values,
            returns=torch.tensor([1.0, 0.0]),
            demo_action_logits=demo_logits,
            demo_actions=torch.tensor([1, 0]),
            demo_weights=torch.tensor([1.0, 2.0]),
            lambda_imit=0.5,
            vf_coef=0.25,
            alpha_ent=0.1,
        )
        loss_parts.total_loss.backward()

        self.assertGreater(action_logits.grad.abs().sum().item(), 0.0)
        self.assertGreater(values.grad.abs().sum().item(), 0.0)
        self.assertGreater(demo_logits.grad.abs().sum().item(), 0.0)
        self.assertGreater(loss_parts.imitation_loss.item(), 0.0)

    def test_actor_critic_loss_runs_without_demo_loss(self):
        from core.policy.ilrec_a2c import compute_ilrec_policy_loss

        action_logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
        values = torch.tensor([0.0, 0.5], requires_grad=True)

        loss_parts = compute_ilrec_policy_loss(
            action_logits=action_logits,
            actions=torch.tensor([1, 0]),
            advantages=torch.tensor([1.0, 1.0]),
            values=values,
            returns=torch.tensor([0.5, 0.5]),
            lambda_imit=0.0,
        )
        loss_parts.total_loss.backward()

        self.assertEqual(loss_parts.imitation_loss.item(), 0.0)
        self.assertGreater(action_logits.grad.abs().sum().item(), 0.0)
        self.assertGreaterEqual(values.grad.abs().sum().item(), 0.0)

    def test_lambda_imit_controls_imitation_contribution(self):
        from core.policy.ilrec_a2c import compute_ilrec_policy_loss

        kwargs = dict(
            action_logits=torch.tensor([[0.0, 1.0]]),
            actions=torch.tensor([1]),
            advantages=torch.tensor([1.0]),
            values=torch.tensor([0.0]),
            returns=torch.tensor([0.0]),
            demo_action_logits=torch.tensor([[2.0, 0.0]]),
            demo_actions=torch.tensor([1]),
            demo_weights=torch.tensor([1.0]),
            vf_coef=0.0,
            alpha_ent=0.0,
        )

        without_imit = compute_ilrec_policy_loss(**kwargs, lambda_imit=0.0)
        with_imit = compute_ilrec_policy_loss(**kwargs, lambda_imit=2.0)

        expected_delta = 2.0 * with_imit.imitation_loss
        self.assertTrue(torch.allclose(with_imit.total_loss - without_imit.total_loss, expected_delta))

    def test_entropy_coefficient_reduces_total_loss(self):
        from core.policy.ilrec_a2c import compute_ilrec_policy_loss

        kwargs = dict(
            action_logits=torch.tensor([[0.0, 0.0]]),
            actions=torch.tensor([0]),
            advantages=torch.tensor([0.0]),
            values=torch.tensor([0.0]),
            returns=torch.tensor([0.0]),
            vf_coef=0.0,
            lambda_imit=0.0,
        )

        no_entropy = compute_ilrec_policy_loss(**kwargs, alpha_ent=0.0)
        with_entropy = compute_ilrec_policy_loss(**kwargs, alpha_ent=0.5)

        self.assertLess(with_entropy.total_loss.item(), no_entropy.total_loss.item())

    def test_discriminator_is_not_updated_by_actor_loss_by_default(self):
        from core.policy.discriminator import TransitionDiscriminator
        from core.policy.ilrec_a2c import compute_demo_weights, compute_ilrec_policy_loss

        discriminator = TransitionDiscriminator(input_dim=2, hidden_sizes=(4,))
        demo_features = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        demo_probs = discriminator(demo_features)
        demo_weights = compute_demo_weights(
            demo_advantages=torch.tensor([1.0, 1.0]),
            discriminator_probs=demo_probs,
            beta=1.0,
            alpha=0.5,
            irl_gamma=1.0,
        ).weights

        loss_parts = compute_ilrec_policy_loss(
            action_logits=torch.tensor([[0.0, 1.0]], requires_grad=True),
            actions=torch.tensor([1]),
            advantages=torch.tensor([1.0]),
            values=torch.tensor([0.0], requires_grad=True),
            returns=torch.tensor([0.0]),
            demo_action_logits=torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True),
            demo_actions=torch.tensor([1, 0]),
            demo_weights=demo_weights,
            lambda_imit=1.0,
        )
        loss_parts.total_loss.backward()

        discriminator_grads = [param.grad for param in discriminator.parameters()]
        self.assertTrue(all(grad is None or grad.abs().sum().item() == 0.0 for grad in discriminator_grads))

    def test_mixed_replay_policy_loss_uses_all_samples_for_rl_and_demo_for_imitation(self):
        from core.policy.ilrec_a2c import compute_mixed_replay_policy_loss, weighted_imitation_loss

        logits = torch.tensor(
            [
                [0.0, 1.0],
                [2.0, 0.0],
                [0.0, 0.0],
            ],
            requires_grad=True,
        )
        actions = torch.tensor([1, 1, 0])
        advantages = torch.tensor([1.0, 2.0, -1.0])
        is_demo = torch.tensor([False, True, True])
        demo_weights = torch.tensor([1.0, 3.0, 2.0])

        loss_parts = compute_mixed_replay_policy_loss(
            action_logits=logits,
            actions=actions,
            advantages=advantages,
            is_demo=is_demo,
            demo_weights=demo_weights,
            lambda_imit=0.5,
            alpha_ent=0.0,
        )
        expected_imitation = weighted_imitation_loss(logits[is_demo], actions[is_demo], demo_weights[is_demo])

        loss_parts.total_loss.backward()

        self.assertEqual(loss_parts.sample_count, 3)
        self.assertEqual(loss_parts.demo_sample_count, 2)
        self.assertEqual(loss_parts.env_sample_count, 1)
        self.assertTrue(torch.allclose(loss_parts.imitation_loss, expected_imitation))
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)

    def test_mixed_replay_policy_loss_all_env_batch_has_zero_imitation(self):
        from core.policy.ilrec_a2c import compute_mixed_replay_policy_loss

        loss_parts = compute_mixed_replay_policy_loss(
            action_logits=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            actions=torch.tensor([1, 0]),
            advantages=torch.tensor([1.0, 1.0]),
            is_demo=torch.tensor([False, False]),
            lambda_imit=1.0,
        )

        self.assertEqual(loss_parts.demo_sample_count, 0)
        self.assertEqual(loss_parts.env_sample_count, 2)
        self.assertEqual(loss_parts.imitation_loss.item(), 0.0)

    def test_mixed_replay_entropy_coefficient_reduces_total_loss(self):
        from core.policy.ilrec_a2c import compute_mixed_replay_policy_loss

        kwargs = dict(
            action_logits=torch.tensor([[0.0, 0.0]]),
            actions=torch.tensor([0]),
            advantages=torch.tensor([0.0]),
            is_demo=torch.tensor([False]),
            lambda_imit=0.0,
        )

        no_entropy = compute_mixed_replay_policy_loss(**kwargs, alpha_ent=0.0)
        with_entropy = compute_mixed_replay_policy_loss(**kwargs, alpha_ent=0.5)

        self.assertLess(with_entropy.total_loss.item(), no_entropy.total_loss.item())

    def test_mixed_replay_policy_loss_rejects_invalid_batch_shapes(self):
        from core.policy.ilrec_a2c import compute_mixed_replay_policy_loss

        with self.assertRaises(ValueError):
            compute_mixed_replay_policy_loss(
                action_logits=torch.tensor([[0.0, 1.0]]),
                actions=torch.tensor([0]),
                advantages=torch.tensor([1.0]),
                is_demo=torch.tensor([True, False]),
            )


if __name__ == "__main__":
    unittest.main()
