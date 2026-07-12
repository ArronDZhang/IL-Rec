import json
import os
import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class TestILRecWorldModelArtifacts(unittest.TestCase):
    def setUp(self):
        self.dataset = os.environ.get("ILREC_SMOKE_DATASET", "amazon")
        self.data_root = Path(
            os.environ.get(
                "ILREC_SMOKE_DATA_ROOT",
                str(ROOT / "environments" / "ILRec" / "data_smoke" / self.dataset),
            )
        )

    def test_smoke_tables_are_committed_and_readable(self):
        required = {
            "train.csv": {"user_id", "item_id", "rating"},
            "test.csv": {"user_id", "item_id", "rating"},
            "user.csv": {"user_id"},
            "item.csv": {"item_id"},
        }

        for filename, expected_columns in required.items():
            path = self.data_root / filename
            self.assertTrue(path.exists(), f"missing smoke table: {path}")
            table = pd.read_csv(path)
            self.assertFalse(table.empty, f"empty smoke table: {path}")
            self.assertTrue(expected_columns.issubset(table.columns))

    def test_manifest_documents_external_large_artifacts(self):
        manifest_path = ROOT / "data_manifest" / f"ILRec_{self.dataset}.json"

        manifest = json.loads(manifest_path.read_text())

        required_files = set(manifest["required_external_files"])
        self.assertIn(f"env/{self.dataset}/{self.dataset}_train.npy", required_files)
        self.assertIn(f"env/{self.dataset}/{self.dataset}_test.npy", required_files)
        self.assertIn(f"env/{self.dataset}/test_distance_mat.pickle", required_files)
        self.assertIn(f"env/{self.dataset}/{self.dataset}_embedding_task.pt", required_files)


if __name__ == "__main__":
    unittest.main()
