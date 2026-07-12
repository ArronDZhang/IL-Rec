import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


class TestILRecPaths(unittest.TestCase):
    def test_path_helpers_resolve_root_level_ilrec_resources(self):
        from environments.ILRec import paths

        ilrec_root = Path("/tmp/il-rec")

        self.assertEqual(
            paths.datamaps_path(ilrec_root, "steam"),
            ilrec_root / "env" / "steam" / "datamaps.json",
        )
        self.assertEqual(
            paths.reward_matrix_path(ilrec_root, "steam", "train"),
            ilrec_root / "env" / "steam" / "steam_train.npy",
        )
        self.assertEqual(
            paths.distance_matrix_path(ilrec_root, "steam", "test"),
            ilrec_root / "env" / "steam" / "test_distance_mat.pickle",
        )
        self.assertEqual(
            paths.embedding_path(ilrec_root, "amazon"),
            ilrec_root / "env" / "amazon" / "amazon_embedding_task.pt",
        )
        self.assertEqual(
            paths.grounding_cache_path(ilrec_root, "amazon"),
            ilrec_root / "env" / "amazon" / "amazon_grounding_cache.json",
        )

    def test_load_item2id_converts_ids_and_returns_datamaps_path(self):
        from environments.ILRec import paths

        with tempfile.TemporaryDirectory() as tmp:
            ilrec_root = Path(tmp) / "il-rec"
            datamaps = ilrec_root / "env" / "amazon" / "datamaps.json"
            write_json(
                datamaps,
                {
                    "item2id_dict": {"Book A": "1", "Book B": 2},
                    "id2item_dict": {"1": "Book A", "2": "Book B"},
                },
            )

            item2id, datamaps_path = paths.load_item2id(ilrec_root, "amazon")

        self.assertEqual(item2id, {"Book A": 1, "Book B": 2})
        self.assertEqual(datamaps_path, datamaps)

    def test_load_id2item_normalizes_keys_to_strings(self):
        from environments.ILRec import paths

        with tempfile.TemporaryDirectory() as tmp:
            ilrec_root = Path(tmp) / "il-rec"
            write_json(
                ilrec_root / "env" / "amazon" / "datamaps.json",
                {
                    "item2id_dict": {"Book A": 1},
                    "id2item_dict": {1: "Book A"},
                },
            )

            id2item, _ = paths.load_id2item(ilrec_root, "amazon")

        self.assertEqual(id2item, {"1": "Book A"})

    def test_invalid_datamaps_are_reported_with_source_path(self):
        from environments.ILRec import paths

        with tempfile.TemporaryDirectory() as tmp:
            ilrec_root = Path(tmp) / "il-rec"
            datamaps = ilrec_root / "env" / "amazon" / "datamaps.json"
            write_json(datamaps, {"item2id_dict": {"Bad": "not-an-int"}})

            with self.assertRaisesRegex(ValueError, "Invalid item id.*datamaps"):
                paths.load_item2id(ilrec_root, "amazon")

            write_json(datamaps, {"id2item_dict": {"0": "Only Reverse"}})
            with self.assertRaisesRegex(ValueError, "item2id_dict"):
                paths.load_item2id(ilrec_root, "amazon")

    def test_load_reward_matrix_uses_mmap_root_matrix(self):
        from environments.ILRec import paths

        with tempfile.TemporaryDirectory() as tmp:
            ilrec_root = Path(tmp) / "il-rec"
            matrix_path = ilrec_root / "env" / "steam" / "steam_train.npy"
            matrix_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(matrix_path, np.array([[4.0, 5.0]], dtype=np.float32))

            matrix = paths.load_reward_matrix(ilrec_root, "steam", "train")

        self.assertEqual(matrix.shape, (1, 2))
        self.assertEqual(float(matrix[0, 1]), 5.0)
        self.assertIsInstance(matrix, np.memmap)


if __name__ == "__main__":
    unittest.main()
