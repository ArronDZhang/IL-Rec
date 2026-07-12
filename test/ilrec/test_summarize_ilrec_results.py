import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import summarize_ilrec_results


def write_summary(path, seed, avg_length, avg_reward, avg_return, total_return, reasons):
    payload = {
        "avg_length": avg_length,
        "avg_reward": avg_reward,
        "avg_return": avg_return,
        "eval_seed": seed,
        "user_count": 2,
        "total_return": total_return,
        "termination_reasons": json.dumps(reasons),
        "config": {
            "env": "AmazonEnv-v0",
            "setting": "FB",
            "train_episodes": 100000,
            "discount": 0.5,
            "train_action_selection": "sample",
            "eval_action_selection": "sample",
            "mixed_replay_sampling": "global_priority",
            "mixed_replay_env_priority_scale": 0.05,
            "train_normalize_advantages": True,
            "train_advantage_clip": 5.0,
            "train_actor_row_norm_project": 10.0,
            "train_actor_bias_clamp": 5.0,
            "policy_logit_clamp": 1.0,
            "policy_logit_clamp_mode": "tanh",
            "eval_logit_clamp": 1.0,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestSummarizeILRecResults(unittest.TestCase):
    def test_aggregates_seed_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary0 = root / "seed0.json"
            summary1 = root / "seed1.json"
            output_json = root / "summary.json"
            output_csv = root / "summary.csv"
            write_summary(summary0, 0, 10.0, 4.0, 40.0, 80.0, {"distance": 1})
            write_summary(summary1, 1, 20.0, 5.0, 100.0, 200.0, {"low_reward": 2})

            summarize_ilrec_results.main(
                [
                    "--dataset",
                    "amazon",
                    "--env",
                    "AmazonEnv-v0",
                    "--summary",
                    str(summary0),
                    "--summary",
                    str(summary1),
                    "--output-json",
                    str(output_json),
                    "--output-csv",
                    str(output_csv),
                ]
            )

            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["setting"], "FB")
            self.assertEqual(payload["seed_count"], 2)
            self.assertEqual(payload["mean_avg_length"], 15.0)
            self.assertEqual(payload["pooled_total_return"], 280.0)
            self.assertEqual(payload["combined_termination_reasons"], {"distance": 1, "low_reward": 2})

            with output_csv.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["dataset"], "amazon")
            self.assertNotIn("eval_" + "action_" + "mask", rows[0])


if __name__ == "__main__":
    unittest.main()
