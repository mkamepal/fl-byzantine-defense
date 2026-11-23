"""Detection counts use actual component scores and saved assignment ground truth."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image

from fl_defense.utils.defense_confusion import DEFENSES, build_reports, save_defense_confusion


def record(rnd, scores):
    return {"round": rnd, "clients": {cid: {name: {"score": values[i]}
            for i, name in enumerate(DEFENSES)} for cid, values in scores.items()}}


class DefenseConfusionTests(unittest.TestCase):
    def setUp(self):
        self.rounds = {
            1: record(1, {"client_0": (1, 0, 1, 0), "client_2": (0, 1, 0, 1),
                          "client_10": (1, 0, 0, 1)}),
            2: record(2, {"client_0": (0, 1, 1, 0), "client_10": (0, 1, 0, 1)}),
        }
        self.assignments = {"client_2": "label_flip", "client_10": "model_replacement"}

    def test_partial_participation_numeric_ids_and_all_four_counts(self):
        before = copy.deepcopy(self.rounds)
        reports = build_reports(self.rounds, self.assignments, 11)
        expected = {"clustering": [[1, 1], [1, 2]], "direction": [[1, 1], [2, 1]],
                    "magnitude": [[2, 0], [0, 3]], "validation": [[0, 2], [3, 0]]}
        for name, matrix in expected.items():
            self.assertEqual(reports[name]["overall"]["matrix"], matrix)
            self.assertEqual(reports[name]["overall"]["observations"], 5)
            self.assertEqual(reports[name]["per_round"]["2"]["observations"], 2)
            self.assertEqual([v["client_id"] for v in reports[name]["decisions"][:3]],
                             ["client_0", "client_2", "client_10"])
        self.assertEqual(self.rounds, before)

    def test_no_malicious_clients_and_undefined_metrics(self):
        reports = build_reports({1: record(1, {"client_0": (1, 1, 1, 1)})}, {}, 1)
        for report in reports.values():
            self.assertEqual(report["overall"]["matrix"], [[1, 0], [0, 0]])
            for metric in ("malicious_precision", "malicious_recall", "malicious_f1"):
                self.assertIsNone(report["overall"][metric])

    def test_missing_scores_bad_ids_and_rounds_fail(self):
        for mutation in (lambda r: r[1]["clients"]["client_0"].pop("direction"),
                         lambda r: r[1]["clients"]["client_0"]["clustering"].update(score=0.5),
                         lambda r: r[1].update(round=2),
                         lambda r: r[1]["clients"].update(client_11={})):
            rounds = copy.deepcopy(self.rounds)
            mutation(rounds)
            with self.assertRaises(ValueError):
                build_reports(rounds, self.assignments, 11)
        with self.assertRaises(ValueError):
            build_reports(self.rounds, {"client_11": "backdoor"}, 11)

    def test_export_saved_assignments_pngs_and_nonoverwriting(self):
        with tempfile.TemporaryDirectory() as root:
            run_dir = Path(root)
            logger = SimpleNamespace(run_dir=run_dir, experiment_id="test_run", config={
                "flower_run_id": 18446744073709551615,
                "run_config": {"num-clients": 11, "attacks-enabled": True}})
            # Missing ground truth must not silently classify all clients as benign.
            with self.assertRaises(FileNotFoundError):
                save_defense_confusion(logger, self.rounds, 2)
            self.assertFalse((run_dir / "confusion_matrix").exists())
            (run_dir / "attacks").mkdir()
            source = run_dir / "attacks" / "assignment_and_poisoning.json"
            source.write_text(json.dumps({"total_clients": 11, "assignments": self.assignments}))
            original = source.read_bytes()
            directory = save_defense_confusion(logger, self.rounds, 2)
            self.assertEqual(len(list(directory.iterdir())), 8)
            for name in DEFENSES:
                data = json.loads((directory / f"{name}.json").read_text())
                self.assertEqual(data["completed_rounds"], [1, 2])
                self.assertEqual(data["flower_run_id"], "18446744073709551615")
                self.assertIn("inactive", data["ground_truth_policy"])
                with Image.open(directory / f"{name}.png") as image:
                    self.assertGreater(image.width, 500)
                    image.verify()
            self.assertEqual(source.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                save_defense_confusion(logger, self.rounds, 2)

    def test_incomplete_run_skips_export(self):
        with tempfile.TemporaryDirectory() as root:
            logger = SimpleNamespace(run_dir=Path(root))
            for rounds in ({}, {1: self.rounds[1]}, {2: self.rounds[2]}):
                self.assertIsNone(save_defense_confusion(logger, rounds, 2))
            self.assertFalse((Path(root) / "confusion_matrix").exists())


if __name__ == "__main__":
    unittest.main()
