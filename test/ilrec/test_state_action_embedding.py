import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        rows = []
        for text in texts:
            rows.append([float(len(text)), float(sum(ord(ch) for ch in text) % 17)])
        return torch.tensor(rows, dtype=torch.float32)


class FailingEncoder:
    def encode(self, texts):
        raise AssertionError(f"unexpected cache miss for {texts}")


class TestStateActionEmbeddingCache(unittest.TestCase):
    def test_cache_encodes_unique_texts_and_preserves_order(self):
        from core.policy.state_action_embedding import StateActionEmbeddingCache

        encoder = FakeEncoder()
        cache = StateActionEmbeddingCache(
            encoder=encoder,
            model_path="/models/llama",
            template_version="template-v1",
        )

        embeddings = cache.encode_texts(["alpha", "beta", "alpha"])

        self.assertEqual(encoder.calls, [["alpha", "beta"]])
        self.assertEqual(tuple(embeddings.shape), (3, 2))
        self.assertTrue(torch.equal(embeddings[0], embeddings[2]))
        self.assertFalse(torch.equal(embeddings[0], embeddings[1]))

    def test_cache_encodes_expert_and_policy_transitions_with_same_renderer(self):
        from core.policy.state_action_embedding import StateActionEmbeddingCache

        encoder = FakeEncoder()
        cache = StateActionEmbeddingCache(
            encoder=encoder,
            model_path="/models/llama",
            template_version="template-v1",
        )
        expert = {
            "source": "expert",
            "user_id": 1,
            "action_id": 2,
            "history_action_ids": [0],
            "history_rewards": [4.0],
            "item_text": "Known item",
        }
        policy = dict(expert)
        policy["source"] = "policy"

        embeddings = cache.encode_transitions([expert, policy], dataset="amazon")

        self.assertEqual(len(encoder.calls), 1)
        self.assertEqual(len(encoder.calls[0]), 1)
        self.assertIn("template=ilrec_state_action_v1", encoder.calls[0][0])
        self.assertEqual(tuple(embeddings.shape), (2, 2))
        self.assertTrue(torch.equal(embeddings[0], embeddings[1]))

    def test_persistent_cache_reuses_saved_embeddings(self):
        from core.policy.state_action_embedding import StateActionEmbeddingCache

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "state_action_embeddings.pt"
            first = StateActionEmbeddingCache(
                encoder=FakeEncoder(),
                cache_path=cache_path,
                model_path="/models/llama",
                template_version="template-v1",
            )
            expected = first.encode_texts(["cached text"])
            first.save()

            second = StateActionEmbeddingCache(
                encoder=FailingEncoder(),
                cache_path=cache_path,
                model_path="/models/llama",
                template_version="template-v1",
            )
            actual = second.encode_texts(["cached text"])

        self.assertTrue(torch.equal(actual, expected))

    def test_metadata_change_invalidates_persistent_cache(self):
        from core.policy.state_action_embedding import StateActionEmbeddingCache

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "state_action_embeddings.pt"
            first_encoder = FakeEncoder()
            first = StateActionEmbeddingCache(
                encoder=first_encoder,
                cache_path=cache_path,
                model_path="/models/llama-a",
                template_version="template-v1",
            )
            first.encode_texts(["same text"])
            first.save()

            second_encoder = FakeEncoder()
            second = StateActionEmbeddingCache(
                encoder=second_encoder,
                cache_path=cache_path,
                model_path="/models/llama-b",
                template_version="template-v1",
            )
            second.encode_texts(["same text"])

        self.assertEqual(first_encoder.calls, [["same text"]])
        self.assertEqual(second_encoder.calls, [["same text"]])

    def test_invalid_encoder_shape_is_rejected(self):
        from core.policy.state_action_embedding import StateActionEmbeddingCache

        class BadEncoder:
            def encode(self, texts):
                return torch.tensor([1.0, 2.0, 3.0])

        cache = StateActionEmbeddingCache(
            encoder=BadEncoder(),
            model_path="/models/llama",
            template_version="template-v1",
        )

        with self.assertRaisesRegex(ValueError, "2D"):
            cache.encode_texts(["bad"])


if __name__ == "__main__":
    unittest.main()
