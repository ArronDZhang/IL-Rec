import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TestReplayBuffers(unittest.TestCase):
    def test_env_replay_insertion_preserves_recent_capacity(self):
        from core.policy.replay_buffers import ReplayBuffer

        buffer = ReplayBuffer(capacity=2, source="env")
        buffer.add({"id": 1})
        buffer.add({"id": 2})
        buffer.add({"id": 3})

        self.assertEqual(len(buffer), 2)
        self.assertEqual([record["id"] for record in buffer.records()], [2, 3])
        self.assertEqual([record["source"] for record in buffer.records()], ["env", "env"])

    def test_demo_replay_insertion_keeps_weights(self):
        from core.policy.replay_buffers import DemoReplayBuffer

        buffer = DemoReplayBuffer()
        buffer.add_demo({"user_id": 0, "action_id": 1}, weight=2.5)

        records = buffer.records()
        self.assertEqual(records[0]["source"], "demo")
        self.assertEqual(records[0]["demo_weight"], 2.5)
        self.assertEqual(records[0]["action_id"], 1)

    def test_mixed_sampling_is_deterministic(self):
        from core.policy.replay_buffers import DemoReplayBuffer, MixedReplayBuffer, ReplayBuffer

        env = ReplayBuffer(source="env")
        demo = DemoReplayBuffer()
        for index in range(5):
            env.add({"id": f"env-{index}"})
            demo.add_demo({"id": f"demo-{index}"}, weight=float(index + 1))
        mixed = MixedReplayBuffer(env, demo)

        first = mixed.sample(batch_size=4, demo_fraction=0.5, seed=7)
        second = mixed.sample(batch_size=4, demo_fraction=0.5, seed=7)

        self.assertEqual(first, second)
        self.assertEqual([record["source"] for record in first].count("demo"), 2)
        self.assertEqual([record["source"] for record in first].count("env"), 2)

    def test_global_priority_sampling_ignores_fixed_demo_fraction(self):
        from core.policy.replay_buffers import DemoReplayBuffer, MixedReplayBuffer, ReplayBuffer

        env = ReplayBuffer(source="env")
        demo = DemoReplayBuffer()
        env.add({"id": "env-0"})
        env.add({"id": "env-1"})
        demo.add_demo({"id": "demo-zero"}, weight=0.0)

        samples = MixedReplayBuffer(env, demo).sample(
            batch_size=2,
            demo_fraction=1.0,
            sampling_mode=MixedReplayBuffer.GLOBAL_PRIORITY,
            seed=0,
        )

        self.assertEqual({record["source"] for record in samples}, {"env"})

    def test_global_priority_sampling_uses_demo_weights_against_env_priority(self):
        from core.policy.replay_buffers import DemoReplayBuffer, MixedReplayBuffer, ReplayBuffer

        env = ReplayBuffer(source="env")
        demo = DemoReplayBuffer()
        env.add({"id": "env-0"})
        demo.add_demo({"id": "demo-high"}, weight=1000.0)

        samples = MixedReplayBuffer(env, demo).sample(
            batch_size=1,
            sampling_mode=MixedReplayBuffer.GLOBAL_PRIORITY,
            seed=0,
        )

        self.assertEqual(samples, [{"id": "demo-high", "source": "demo", "demo_weight": 1000.0}])

    def test_mixed_sampling_handles_empty_env_buffer(self):
        from core.policy.replay_buffers import DemoReplayBuffer, MixedReplayBuffer, ReplayBuffer

        env = ReplayBuffer(source="env")
        demo = DemoReplayBuffer()
        demo.add_demo({"id": "demo-0"}, weight=1.0)

        samples = MixedReplayBuffer(env, demo).sample(batch_size=2, demo_fraction=0.5, seed=0)

        self.assertEqual(samples, [{"id": "demo-0", "source": "demo", "demo_weight": 1.0}])

    def test_large_buffer_sampling_does_not_materialize_all_records(self):
        from core.policy.replay_buffers import ReplayBuffer

        buffer = ReplayBuffer(source="env")
        for index in range(100):
            buffer.add({"id": index})

        def fail_records():
            raise AssertionError("sample should not materialize the full replay buffer")

        buffer.records = fail_records

        samples = buffer.sample(batch_size=3, seed=11)

        self.assertEqual(len(samples), 3)
        self.assertEqual({record["source"] for record in samples}, {"env"})

    def test_demo_sampling_prioritizes_demo_weights_deterministically(self):
        from core.policy.replay_buffers import DemoReplayBuffer

        buffer = DemoReplayBuffer()
        buffer.add_demo({"id": "low"}, weight=0.0)
        buffer.add_demo({"id": "high"}, weight=10.0)

        first = buffer.sample(batch_size=1, seed=3)
        second = buffer.sample(batch_size=1, seed=3)

        self.assertEqual(first, second)
        self.assertEqual(first, [{"id": "high", "source": "demo", "demo_weight": 10.0}])

    def test_mixed_priority_totals_include_env_and_demo_weights(self):
        from core.policy.replay_buffers import DemoReplayBuffer, MixedReplayBuffer, ReplayBuffer

        env = ReplayBuffer(source="env")
        demo = DemoReplayBuffer()
        env.add({"id": "env-0"})
        env.add({"id": "env-1"})
        demo.add_demo({"id": "demo-low"}, weight=0.25)
        demo.add_demo({"id": "demo-high"}, weight=2.75)

        totals = MixedReplayBuffer(env, demo).priority_totals()

        self.assertEqual(totals["env_priority"], 2.0)
        self.assertEqual(totals["demo_priority"], 3.0)
        self.assertEqual(totals["total_priority"], 5.0)

    def test_global_priority_sampling_can_downweight_env_priority(self):
        from core.policy.replay_buffers import DemoReplayBuffer, MixedReplayBuffer, ReplayBuffer

        env = ReplayBuffer(source="env")
        demo = DemoReplayBuffer()
        for index in range(10):
            env.add({"id": f"env-{index}"})
        demo.add_demo({"id": "demo"}, weight=1.0)

        mixed = MixedReplayBuffer(env, demo, env_priority_scale=0.05)
        totals = mixed.priority_totals()
        samples = mixed.sample(
            batch_size=1,
            sampling_mode=MixedReplayBuffer.GLOBAL_PRIORITY,
            seed=0,
        )

        self.assertEqual(totals["env_priority"], 0.5)
        self.assertEqual(totals["demo_priority"], 1.0)
        self.assertEqual(totals["env_priority_scale"], 0.05)
        self.assertEqual(samples, [{"id": "demo", "source": "demo", "demo_weight": 1.0}])

    def test_mixed_replay_rejects_negative_env_priority_scale(self):
        from core.policy.replay_buffers import DemoReplayBuffer, MixedReplayBuffer, ReplayBuffer

        with self.assertRaisesRegex(ValueError, "non-negative"):
            MixedReplayBuffer(ReplayBuffer(source="env"), DemoReplayBuffer(), env_priority_scale=-0.1)


if __name__ == "__main__":
    unittest.main()
