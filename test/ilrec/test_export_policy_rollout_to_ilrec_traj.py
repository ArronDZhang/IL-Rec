import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "export_policy_rollout_to_ilrec_traj.py"
ILREC_ROOT = Path("/home/hehui/il-rec")


def load_calculate_module():
    path = ILREC_ROOT / "calculate_traj_results.py"
    spec = importlib.util.spec_from_file_location("calculate_traj_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


class TestExportPolicyRolloutToILRecTraj(unittest.TestCase):
    def test_exports_toy_rollout_and_metric_script_can_parse_it(self):
        calculate = load_calculate_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rollout_path = tmp_path / "rollout.json"
            output_path = tmp_path / "amazon_ilrec_gpt35_seed0.json"
            write_json(
                rollout_path,
                [
                    {
                        "trajectory_id": "toy-user",
                        "userid": 42,
                        "steps": [
                            {"item": "Known Item", "action_id": 7, "reward": 3.5, "done": False},
                            {
                                "item": "Terminal Item",
                                "action_id": 8,
                                "reward": 1.0,
                                "done": True,
                                "terminal_failure": True,
                            },
                        ],
                    }
                ],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--rollout-json",
                    str(rollout_path),
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            summary, users = calculate.calculate_file(output_path)

        self.assertEqual(summary["user_count"], 1)
        self.assertEqual(summary["total_length"], 2)
        self.assertEqual(summary["total_return"], 3.5)
        self.assertEqual(summary["avg_reward"], 1.75)
        self.assertEqual(users[0]["raw_rewards"], [3.5, -1000.0])
        self.assertEqual(users[0]["adjusted_rewards"], [3.5, 0.0])

    def test_output_contains_traj_and_traj_by_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rollout_path = tmp_path / "rollout.json"
            output_path = tmp_path / "exported.json"
            write_json(
                rollout_path,
                {
                    "u1": {
                        "userid": 1,
                        "steps": [{"item": "A", "reward": 2.0, "done": True}],
                    }
                },
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--rollout-json",
                    str(rollout_path),
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            exported = json.loads(output_path.read_text())

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("traj", exported["u1"])
        self.assertIn("traj_by_line", exported["u1"])
        self.assertIn("reward=2.000000", exported["u1"]["traj"])


if __name__ == "__main__":
    unittest.main()
