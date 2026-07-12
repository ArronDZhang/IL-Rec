import os
import pickle
import sys
import tempfile
import unittest
import importlib
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestILRecEnv(unittest.TestCase):
    def setUp(self):
        from environments.ILRec.env.ILRecEnv import AmazonEnv, SteamEnv

        self.amazon_cls = AmazonEnv
        self.steam_cls = SteamEnv
        self.mat = np.array(
            [
                [5.0, 1.5, 4.0],
                [3.0, 4.5, 2.5],
            ],
            dtype=np.float32,
        )
        self.distance = np.array(
            [
                [0.0, 20.0, 99.0],
                [20.0, 0.0, 8.0],
                [99.0, 8.0, 0.0],
            ],
            dtype=np.float32,
        )

    def make_env(self, **overrides):
        params = {
            "mat": self.mat,
            "mat_distance": self.distance,
            "num_leave_compute": 2,
            "leave_threshold": 15,
            "max_turn": 3,
        }
        params.update(overrides)
        return self.amazon_cls(**params)

    def test_reset_returns_user_observation_and_defines_spaces(self):
        env = self.make_env()

        obs = env.reset(user_id=1)

        np.testing.assert_array_equal(obs, np.array([1], dtype=np.int64))
        np.testing.assert_array_equal(env.state, obs)
        self.assertEqual(env.cur_user, 1)
        self.assertEqual(env.action_space.n, 3)
        self.assertEqual(env.observation_space.shape, (1,))
        self.assertEqual(env.MAX_R, 5.0)
        self.assertEqual(env.MIN_R, 1.5)

    def test_step_uses_matrix_reward_and_max_turn_done(self):
        env = self.make_env(max_turn=2, leave_threshold=0)
        env.reset(user_id=1)

        obs, reward, done, info = env.step(0)
        np.testing.assert_array_equal(obs, np.array([1], dtype=np.int64))
        self.assertEqual(reward, 3.0)
        self.assertFalse(done)
        self.assertEqual(info["reason"], "continue")
        self.assertEqual(info["user_id"], 1)
        self.assertEqual(info["item_id"], 0)

        _, reward, done, info = env.step(2)
        self.assertEqual(reward, 2.5)
        self.assertTrue(done)
        self.assertEqual(info["reason"], "max_turn")

    def test_low_reward_terminates_even_without_history(self):
        env = self.make_env(leave_threshold=0)
        env.reset(user_id=0)

        _, reward, done, info = env.step(1)

        self.assertEqual(reward, 1.5)
        self.assertTrue(done)
        self.assertEqual(info["reason"], "low_reward")

    def test_recent_distance_below_threshold_terminates(self):
        env = self.make_env()
        env.reset(user_id=1)

        _, _, done, info = env.step(1)
        self.assertFalse(done)
        self.assertEqual(info["reason"], "continue")

        _, reward, done, info = env.step(2)
        self.assertEqual(reward, 2.5)
        self.assertTrue(done)
        self.assertEqual(info["reason"], "distance")
        self.assertEqual(info["matched_history_item"], 1)

    def test_distance_check_uses_only_recent_window(self):
        distance = self.distance.copy()
        distance[2, 0] = 8.0
        distance[0, 2] = 8.0
        distance[2, 1] = 99.0
        distance[1, 2] = 99.0
        env = self.make_env(mat_distance=distance, num_leave_compute=1, max_turn=4)
        env.reset(user_id=1)

        env.step(0)
        env.step(1)
        _, _, done, info = env.step(2)

        self.assertFalse(done)
        self.assertEqual(info["reason"], "continue")

    def test_cur_user_can_be_assigned_as_roler_scalar(self):
        env = self.make_env(leave_threshold=0)
        env.reset(user_id=0)
        env.cur_user = 1

        obs, reward, done, _ = env.step(0)

        np.testing.assert_array_equal(obs, np.array([1], dtype=np.int64))
        self.assertEqual(reward, 3.0)
        self.assertFalse(done)

    def test_minimal_single_user_item_env(self):
        env = self.amazon_cls(
            mat=np.array([[2.0]], dtype=np.float32),
            mat_distance=np.array([[0.0]], dtype=np.float32),
            max_turn=1,
        )

        obs = env.reset(user_id=0)
        next_obs, reward, done, info = env.step(0)

        np.testing.assert_array_equal(obs, np.array([0], dtype=np.int64))
        np.testing.assert_array_equal(next_obs, np.array([0], dtype=np.int64))
        self.assertEqual(reward, 2.0)
        self.assertTrue(done)
        self.assertEqual(info["reason"], "max_turn")

    def test_invalid_user_or_action_raises_value_error(self):
        env = self.make_env()

        with self.assertRaisesRegex(ValueError, "user_id"):
            env.reset(user_id=2)

        env.reset(user_id=0)
        with self.assertRaisesRegex(ValueError, "action"):
            env.step(3)

        with self.assertRaisesRegex(ValueError, "mat"):
            self.amazon_cls(mat=np.empty((0, 1)), mat_distance=np.empty((1, 1)))

    def test_steam_defaults_match_plan(self):
        env = self.steam_cls(mat=self.mat, mat_distance=self.distance)

        self.assertEqual(env.leave_threshold, 50)
        self.assertEqual(env.num_leave_compute, 4)
        self.assertEqual(env.max_turn, 100)

    def test_from_files_reuses_cached_resources_without_shared_episode_state(self):
        ilrec_env_module = importlib.import_module("environments.ILRec.env.ILRecEnv")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mat_path = tmp_path / "mat.npy"
            distance_path = tmp_path / "distance.pickle"
            np.save(mat_path, self.mat)
            with distance_path.open("wb") as f:
                pickle.dump(self.distance, f)

            self.amazon_cls.clear_resource_cache()
            with mock.patch.object(ilrec_env_module.np, "load", wraps=ilrec_env_module.np.load) as np_load:
                with mock.patch.object(ilrec_env_module.pickle, "load", wraps=ilrec_env_module.pickle.load) as pickle_load:
                    first = self.amazon_cls.from_files(
                        mat_path,
                        distance_path,
                        num_leave_compute=2,
                        leave_threshold=15,
                        max_turn=3,
                    )
                    second = self.amazon_cls.from_files(
                        mat_path,
                        distance_path,
                        num_leave_compute=2,
                        leave_threshold=15,
                        max_turn=3,
                    )

            self.assertEqual(np_load.call_count, 1)
            self.assertEqual(pickle_load.call_count, 1)

            first.reset(user_id=1)
            second.reset(user_id=1)
            first.step(1)

            self.assertEqual(first.history_action, [1])
            self.assertEqual(second.history_action, [])
            np.testing.assert_array_equal(second.state, np.array([1], dtype=np.int64))

    def test_lookup_rewards_vectorizes_matrix_access_and_validates_bounds(self):
        env = self.make_env()

        rewards = env.lookup_rewards([0, 1, 1], [0, 1, 2])

        np.testing.assert_allclose(rewards, np.array([5.0, 4.5, 2.5]))
        with self.assertRaisesRegex(ValueError, "user_id"):
            env.lookup_rewards([0, 2], [0, 1])
        with self.assertRaisesRegex(ValueError, "action"):
            env.lookup_rewards([0, 1], [0, 3])


if __name__ == "__main__":
    unittest.main()
