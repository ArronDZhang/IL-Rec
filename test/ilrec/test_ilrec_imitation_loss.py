import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TestILRecWeightedImitationLoss(unittest.TestCase):
    def test_loss_selects_demonstrated_action(self):
        from core.policy.ilrec_a2c import weighted_imitation_loss

        logits = torch.tensor([[0.0, 2.0, -1.0]])
        actions = torch.tensor([1])
        weights = torch.tensor([1.0])

        loss = weighted_imitation_loss(logits, actions, weights)
        expected = -F.log_softmax(logits, dim=-1)[0, 1]

        self.assertTrue(torch.allclose(loss, expected))

    def test_weights_scale_per_sample_loss(self):
        from core.policy.ilrec_a2c import weighted_imitation_loss

        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        actions = torch.tensor([0, 0])
        weights = torch.tensor([1.0, 3.0])

        loss = weighted_imitation_loss(logits, actions, weights)
        log_probs = F.log_softmax(logits, dim=-1)
        expected = -(torch.tensor([1.0, 3.0]) * torch.stack([log_probs[0, 0], log_probs[1, 0]])).mean()

        self.assertTrue(torch.allclose(loss, expected))

    def test_zero_weight_sample_does_not_affect_loss(self):
        from core.policy.ilrec_a2c import weighted_imitation_loss

        logits_a = torch.tensor([[-100.0, 100.0], [0.0, 2.0]])
        logits_b = torch.tensor([[100.0, -100.0], [0.0, 2.0]])
        actions = torch.tensor([0, 1])
        weights = torch.tensor([0.0, 1.0])

        loss_a = weighted_imitation_loss(logits_a, actions, weights)
        loss_b = weighted_imitation_loss(logits_b, actions, weights)

        self.assertTrue(torch.allclose(loss_a, loss_b))

    def test_gradients_flow_to_logits(self):
        from core.policy.ilrec_a2c import weighted_imitation_loss

        logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
        actions = torch.tensor([1, 0])
        weights = torch.tensor([1.0, 1.0])

        loss = weighted_imitation_loss(logits, actions, weights)
        loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.all(torch.isfinite(logits.grad)))
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)

    def test_invalid_shapes_raise_clear_errors(self):
        from core.policy.ilrec_a2c import weighted_imitation_loss

        with self.assertRaisesRegex(ValueError, "2D"):
            weighted_imitation_loss(torch.tensor([1.0, 2.0]), torch.tensor([0]), torch.tensor([1.0]))
        with self.assertRaisesRegex(ValueError, "batch"):
            weighted_imitation_loss(torch.zeros(2, 3), torch.tensor([0]), torch.tensor([1.0]))


if __name__ == "__main__":
    unittest.main()
