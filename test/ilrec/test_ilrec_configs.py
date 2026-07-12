import argparse
import csv
import importlib
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_table_fixture(root, dataset):
    dataset_root = root / dataset
    write_csv(
        dataset_root / "train.csv",
        ["user_id", "item_id", "rating", "timestamp"],
        [
            {"user_id": "0", "item_id": "2", "rating": "4.5", "timestamp": "0"},
            {"user_id": "1", "item_id": "0", "rating": "2.0", "timestamp": "0"},
        ],
    )
    write_csv(
        dataset_root / "test.csv",
        ["user_id", "item_id", "rating", "timestamp"],
        [{"user_id": "1", "item_id": "2", "rating": "3.5", "timestamp": "0"}],
    )
    write_csv(
        dataset_root / "user.csv",
        ["user_id"],
        [{"user_id": "0"}, {"user_id": "1"}],
    )
    write_csv(
        dataset_root / "item.csv",
        ["item_id"],
        [{"item_id": "0"}, {"item_id": "1"}, {"item_id": "2"}],
    )


def write_env_fixture(root, dataset):
    env_root = root / "env" / dataset
    env_root.mkdir(parents=True, exist_ok=True)
    np.save(env_root / f"{dataset}_test.npy", np.array([[5.0, 1.0], [3.0, 4.0]]))
    with (env_root / "test_distance_mat.pickle").open("wb") as f:
        pickle.dump(np.array([[0.0, 20.0], [20.0, 0.0]]), f)


class TestILRecConfigs(unittest.TestCase):
    def setUp(self):
        from core import configs

        self.configs = configs

    def test_get_features_for_ilrec_envs(self):
        self.assertEqual(
            self.configs.get_features("AmazonEnv-v0"),
            (["user_id"], ["item_id"], ["rating"]),
        )
        self.assertEqual(
            self.configs.get_features("SteamEnv-v0"),
            (["user_id"], ["item_id"], ["rating"]),
        )

    def test_training_and_val_data_load_indexed_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data"
            write_table_fixture(data_root, "amazon")

            with mock.patch.object(self.configs, "ILREC_DATA_ROOT", data_root, create=True):
                df_train, df_user, df_item, list_feat = self.configs.get_training_data("AmazonEnv-v0")
                df_val, df_user_val, df_item_val, list_feat_val = self.configs.get_val_data("AmazonEnv-v0")

            self.assertEqual(df_train.to_dict("records")[0]["rating"], 4.5)
            self.assertEqual(df_train.to_dict("records")[1]["item_id"], 0)
            self.assertEqual(df_val.to_dict("records"), [{"user_id": 1, "item_id": 2, "rating": 3.5, "timestamp": 0}])
            self.assertTrue(pd.api.types.is_integer_dtype(df_train["user_id"]))
            self.assertTrue(pd.api.types.is_integer_dtype(df_train["item_id"]))
            self.assertTrue(pd.api.types.is_integer_dtype(df_train["timestamp"]))
            self.assertTrue(pd.api.types.is_float_dtype(df_train["rating"]))
            self.assertEqual(df_user.index.name, "user_id")
            self.assertEqual(df_user.index.tolist(), [0, 1])
            self.assertEqual(df_user.columns.tolist(), [])
            self.assertEqual(df_item.index.name, "item_id")
            self.assertEqual(df_item.index.tolist(), [0, 1, 2])
            self.assertEqual(df_item.columns.tolist(), [])
            self.assertEqual(df_user_val.index.tolist(), [0, 1])
            self.assertEqual(df_item_val.index.tolist(), [0, 1, 2])
            self.assertEqual(list_feat, [])
            self.assertEqual(list_feat_val, [])

    def test_ilrec_data_root_can_be_overridden_by_environment(self):
        original = os.environ.get("ILREC_DATA_ROOT")
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.environ["ILREC_DATA_ROOT"] = tmp
                reloaded = importlib.reload(self.configs)
                self.assertEqual(reloaded.ILREC_DATA_ROOT, Path(tmp))
            finally:
                if original is None:
                    os.environ.pop("ILREC_DATA_ROOT", None)
                else:
                    os.environ["ILREC_DATA_ROOT"] = original
                self.configs = importlib.reload(self.configs)

    def test_common_args_defaults_match_algorithm_plan(self):
        amazon_args = self.configs.get_common_args(argparse.Namespace(env="AmazonEnv-v0"))
        steam_args = self.configs.get_common_args(argparse.Namespace(env="SteamEnv-v0"))

        self.assertEqual(amazon_args.yfeat, "rating")
        self.assertEqual(amazon_args.leave_threshold, 15)
        self.assertEqual(amazon_args.num_leave_compute, 4)
        self.assertEqual(amazon_args.max_turn, 100)
        self.assertFalse(amazon_args.need_transform)
        self.assertFalse(amazon_args.is_binarize)

        self.assertEqual(steam_args.yfeat, "rating")
        self.assertEqual(steam_args.leave_threshold, 50)
        self.assertEqual(steam_args.num_leave_compute, 4)
        self.assertEqual(steam_args.max_turn, 100)
        self.assertFalse(steam_args.need_transform)
        self.assertFalse(steam_args.is_binarize)

    def test_get_true_env_builds_amazon_and_steam_envs(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_root = Path(tmp) / "il-rec"
            write_env_fixture(source_root, "amazon")
            write_env_fixture(source_root, "steam")

            with mock.patch.object(self.configs, "ILREC_SOURCE_ROOT", source_root, create=True):
                amazon_args = self.configs.get_common_args(argparse.Namespace(env="AmazonEnv-v0"))
                steam_args = self.configs.get_common_args(argparse.Namespace(env="SteamEnv-v0"))
                amazon_env, amazon_cls, amazon_kwargs = self.configs.get_true_env(amazon_args)
                steam_env, steam_cls, steam_kwargs = self.configs.get_true_env(steam_args)

        self.assertEqual(amazon_cls.__name__, "AmazonEnv")
        self.assertEqual(steam_cls.__name__, "SteamEnv")
        self.assertEqual(amazon_kwargs["leave_threshold"], 15)
        self.assertEqual(steam_kwargs["leave_threshold"], 50)
        self.assertEqual(amazon_env.mat.shape, (2, 2))
        self.assertEqual(steam_env.mat.shape, (2, 2))
        amazon_env.reset(user_id=1)
        _, reward, done, _ = amazon_env.step(1)
        self.assertEqual(reward, 4.0)
        self.assertFalse(done)

    def test_missing_ilrec_table_reports_file_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(self.configs, "ILREC_DATA_ROOT", Path(tmp), create=True):
                with self.assertRaisesRegex(FileNotFoundError, "train.csv"):
                    self.configs.get_training_data("AmazonEnv-v0")

if __name__ == "__main__":
    unittest.main()
