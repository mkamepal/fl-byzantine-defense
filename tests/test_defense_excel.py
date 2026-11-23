"""Export fidelity, typed cells, alignment, history and atomic failure behavior."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from openpyxl import load_workbook

from fl_defense.detection.defense_config import DefenseConfig
from fl_defense.detection.direction import measure_direction
from fl_defense.detection.magnitude import measure_magnitude
from fl_defense.detection.trust_score import combine_scores
from fl_defense.detection.validation import score_validation
from fl_defense.utils.defense_excel import COLUMNS, DefenseExcel, headers, round_tables


def example_round(rnd=1):
    direction = measure_direction({"client_0": [1, 0], "client_10": [-1, 0], "client_2": [0, 0]})
    magnitude = measure_magnitude({"client_0": [4], "client_10": [9], "client_2": [3]})
    validation = score_validation({cid: {"accuracy": acc, "loss": .35, "example_count": 100}
                                   for cid, acc in (("client_0", .9), ("client_10", .8), ("client_2", .84))})
    clustering = {cid: {"label": -1, "membership_strength": 0.0, "pca_coordinates": [i * .1, i * .2],
                        "head_distance": i * .3, "noise_distance_threshold": 1.5, "score": int(i == 0)}
                  for i, cid in enumerate(direction)}
    components = dict(direction=direction, magnitude=magnitude, validation=validation, clustering=clustering)
    trust = combine_scores({name: {cid: v["score"] for cid, v in values.items()}
                            for name, values in components.items()}, DefenseConfig().weights)
    return {"round": rnd, "head_client": "client_0", "validation_source": "synthetic holdout",
            "clients": {cid: {**trust[cid], **{name: values[cid] for name, values in components.items()}}
                        for cid in reversed(direction)}}


class DefenseExcelTests(unittest.TestCase):
    def test_numeric_run_ids_remain_exact_across_rounds(self):
        for run_id in (99, 17619004868423387106, 2**64 - 1):
            with self.subTest(run_id=run_id), tempfile.TemporaryDirectory() as folder:
                metadata = {"Run ID": run_id}
                writer = DefenseExcel(folder, metadata)
                writer.save({1: example_round()})
                writer.save({1: example_round(), 2: example_round(2)})
                DefenseExcel(folder, metadata).save({3: example_round(3)})
                book = load_workbook(writer.path)
                try:
                    self.assertEqual(book["Configuration"]["B2"].value, str(run_id))
                    self.assertEqual(book["Configuration"]["B2"].data_type, "s")
                    for sheet in COLUMNS:
                        self.assertEqual({row[0] for row in list(book[sheet].values)[1:]}, {1, 2, 3})
                        self.assertEqual(book[sheet].max_row, 10)
                finally:
                    book.close()
                before = writer.path.read_bytes()
                with self.assertRaisesRegex(RuntimeError, "different effective"):
                    DefenseExcel(folder, {"Run ID": run_id - 1}).save({4: example_round(4)})
                self.assertEqual(writer.path.read_bytes(), before)

    def test_columns_order_types_fidelity_and_idempotent_history(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = DefenseExcel(folder, {"Run ID": "test", "settings": DefenseConfig().effective()})
            first, second = example_round(), example_round(2)
            # Export actual runner selection, even if it differs from recomputation.
            first["clients"]["client_0"]["selected"] = False
            writer.save({1: first})
            self.assertEqual(writer.path, Path(folder) / "defense_scores.xlsx")
            writer.save({2: second, 1: first})
            writer.save({1: first, 2: second})
            # Reconstructed writer cannot erase previous rounds.
            resumed = DefenseExcel(folder, writer.metadata)
            resumed.save({3: example_round(3)})
            book = load_workbook(writer.path)
            self.addCleanup(book.close)
            self.assertEqual(book.sheetnames, [*COLUMNS, "Configuration"])
            for sheet in COLUMNS:
                self.assertEqual([c.value for c in book[sheet][1]], headers(sheet))
                self.assertEqual(book[sheet].max_row, 10)
                self.assertTrue(book[sheet].freeze_panes)
                self.assertTrue(book[sheet].auto_filter.ref)
                for rnd, record in ((1, first), (2, second), (3, example_round(3))):
                    expected = round_tables(record)[sheet]
                    rows = list(book[sheet].values)[1 + (rnd - 1) * 3:1 + rnd * 3]
                    self.assertEqual([r[1] for r in rows], ["client_0", "client_2", "client_10"])
                    for row in rows:
                        for actual, value in zip(row, expected[row[:2]], strict=True):
                            if type(value) is float:
                                self.assertAlmostEqual(actual, value, places=12)
                            else:
                                self.assertEqual(actual, value)
            self.assertEqual(book["Scores"]["A2"].data_type, "n")
            self.assertEqual(book["Scores"]["G2"].data_type, "n")
            self.assertEqual(book["Scores"]["H2"].data_type, "b")
            self.assertFalse(book["Scores"]["H2"].value)
            self.assertIn("%", book["Validation"]["D2"].number_format)
            self.assertEqual(book["Validation"]["D2"].value, .9)
            self.assertEqual(book["Clustering"]["D2"].value, 0)

    def test_small_effective_tolerances_are_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = DefenseExcel(folder, {"direction_epsilon": 1e-12, "magnitude_boundary_atol": 1e-8})
            writer.save({1: example_round()})
            book = load_workbook(writer.path)
            try:
                self.assertEqual(book["Configuration"]["B2"].value, 1e-12)
                self.assertIn("E+00", book["Configuration"]["B2"].number_format)
                self.assertIn("E+00", book["Configuration"]["B3"].number_format)
            finally:
                book.close()

    def test_invalid_results_and_write_failures_preserve_prior_workbook(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = DefenseExcel(folder, {"Run ID": "test"})
            writer.save({1: example_round()})
            before = writer.path.read_bytes()
            for change in ("missing", "nonfinite", "head", "round", "decision"):
                record = example_round(2)
                if change == "missing":
                    del record["clients"]["client_2"]["magnitude"]
                elif change == "nonfinite":
                    record["clients"]["client_2"]["direction"]["angle"] = float("nan")
                elif change == "head":
                    del record["clients"]["client_0"]
                elif change == "round":
                    record["round"] = 3
                else:
                    del record["clients"]["client_2"]["selected"]
                with self.subTest(change=change), self.assertRaisesRegex(RuntimeError, str(writer.path)):
                    writer.save({2: record})
                self.assertEqual(writer.path.read_bytes(), before)
            for target in ("openpyxl.workbook.workbook.Workbook.save", "fl_defense.utils.defense_excel.os.replace"):
                with patch(target, side_effect=OSError("synthetic disk failure")), self.assertRaisesRegex(RuntimeError, "synthetic disk failure"):
                    writer.save({2: example_round(2)})
                self.assertEqual(writer.path.read_bytes(), before)
                self.assertFalse(list(Path(folder).glob(".defense_scores_*.xlsx")))
            conflict = example_round()
            conflict["clients"]["client_2"]["selected"] = False
            with self.assertRaisesRegex(RuntimeError, "Conflicting"):
                writer.save({1: conflict})
            self.assertEqual(writer.path.read_bytes(), before)

    def test_existing_history_alignment_and_metadata_are_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = DefenseExcel(folder, {"Run ID": "test"})
            writer.save({1: example_round()})
            with self.assertRaisesRegex(RuntimeError, "different effective"):
                DefenseExcel(folder, {"Run ID": "another"}).save({2: example_round(2)})
            book = load_workbook(writer.path)
            book["Magnitude"].delete_rows(2)
            book.save(writer.path)
            book.close()
            before = writer.path.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "misaligned"):
                writer.save({2: example_round(2)})
            self.assertEqual(writer.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
