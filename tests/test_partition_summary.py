"""Verify partition coverage, reproducibility, split isolation, and CSV output."""

import contextlib
import csv
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from datasets import Dataset
import numpy as np

from fl_defense.data import cifar10
from fl_defense.data.client_identity import client_id, client_role
from fl_defense.data.partition_plan import ClientPartition, build_partition_plan, validate_partitions
from fl_defense.data.partitioning import PartitionSettings, iid_partitioner
from fl_defense.data.partition_summary import CLASS_NAMES, partition_statistics, prepare_partition_summary


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.tile(np.arange(10), 150)
        cifar10._partition_plan.cache_clear()

    def tearDown(self):
        cifar10._partition_plan.cache_clear()

    def test_identity_is_partition_based(self):
        self.assertEqual([client_id(i) for i in range(15)], [f"client_{i}" for i in range(15)])
        self.assertEqual(client_role(0), "trusted_head")
        self.assertTrue(all(client_role(i) == "client" for i in range(1, 15)))

    def test_iid_preserves_original_assignments_and_split(self):
        plan = build_partition_plan(self.labels, 15, PartitionSettings())
        original = iid_partitioner(15)
        original.dataset = Dataset.from_dict({"index": np.arange(len(self.labels)), "label": self.labels})
        for i, partition in enumerate(plan):
            old_split = original.load_partition(i).train_test_split(test_size=0.2, seed=42)
            np.testing.assert_array_equal(partition.train_indices, old_split["train"]["index"])
            np.testing.assert_array_equal(partition.validation_indices, old_split["test"]["index"])

    def test_methods_reproduce_and_match_observed_counts(self):
        for method in ("iid", "dirichlet"):
            with self.subTest(method=method):
                settings = PartitionSettings(method=method)
                first = build_partition_plan(self.labels, 15, settings)
                second = build_partition_plan(self.labels, 15, settings)
                # Reverse access represents a different client execution order.
                for i in reversed(range(15)):
                    np.testing.assert_array_equal(first[i].indices, second[i].indices)
                    np.testing.assert_array_equal(first[i].train_indices, second[i].train_indices)
                rows = partition_statistics(self.labels, first, settings, 123)
                self.assertEqual(sum(row["total_images"] for row in rows), len(self.labels))
                for i, row in enumerate(rows):
                    self.assertEqual(sum(row[name.lower()] for name in CLASS_NAMES), row["total_images"])
                    self.assertEqual(row["train_total_images"] + row["validation_total_images"], row["total_images"])
                    for label, name in enumerate(CLASS_NAMES):
                        self.assertEqual(row[name.lower()], np.sum(self.labels[first[i].indices] == label))
                        self.assertEqual(row[name.lower()], row["train_" + name.lower()] + row["validation_" + name.lower()])
                if method == "dirichlet":
                    changed = build_partition_plan(self.labels, 15, PartitionSettings(method=method, seed=43))
                    self.assertFalse(np.array_equal(first[0].indices, changed[0].indices))

    def test_invalid_coverage_and_split_are_rejected(self):
        good = build_partition_plan(self.labels, 15, PartitionSettings())
        with self.assertRaisesRegex(ValueError, "overlap|cover"):
            validate_partitions((good[0],) + good[:-1], len(self.labels))
        with self.assertRaisesRegex(ValueError, "cover"):
            validate_partitions(good[:-1], len(self.labels))
        bad = ClientPartition(good[0].indices, good[0].train_indices, good[0].train_indices)
        with self.assertRaises(ValueError):
            validate_partitions((bad,) + good[1:], len(self.labels))
        with self.assertRaises(ValueError):
            PartitionSettings(method="unknown")
        with self.assertRaises(ValueError):
            PartitionSettings(alpha=0)
        with self.assertRaises(ValueError):
            build_partition_plan(self.labels, 15, PartitionSettings(method="dirichlet", min_partition_size=200))

    def test_summary_matches_client_loading_and_csv(self):
        # Encode each source row in its pixel value to check row identity as well as labels.
        images = np.arange(len(self.labels), dtype=np.float32).reshape(-1, 1, 1, 1)
        with tempfile.TemporaryDirectory() as directory:
            config = {"results-dir": directory, "dataset-dir": directory}
            with patch.object(cifar10, "_load_local_training_data", return_value=(images, self.labels)), \
                 patch("fl_defense.data.partition_summary._load_local_training_data", return_value=(images, self.labels)):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    rows = prepare_partition_summary(config, 15, 456, Path(directory) / "partition_summary.csv")
                self.assertEqual(output.getvalue().count("DATASET PARTITION SUMMARY"), 1)
                self.assertEqual(output.getvalue().count("Client ID:"), 15)
                plan = cifar10._partition_plan(directory, 15, PartitionSettings())
                for i in reversed(range(15)):
                    x_train, y_train, x_validation, y_validation = cifar10.load_data(i, 15, directory)
                    np.testing.assert_allclose(x_train[:, 0, 0, 0], plan[i].train_indices / 255.0)
                    np.testing.assert_array_equal(y_train, self.labels[plan[i].train_indices])
                    np.testing.assert_array_equal(y_validation, self.labels[plan[i].validation_indices])
                    self.assertEqual(len(x_validation), rows[i]["validation_total_images"])
                with (Path(directory) / "partition_summary.csv").open() as stream:
                    saved = list(csv.DictReader(stream))
                self.assertEqual(len(saved), 15)
                self.assertEqual(saved[0]["client_id"], "client_0")
                self.assertEqual(saved[0]["role"], "trusted_head")
                self.assertEqual(saved[0]["source_split"], "cifar10/train")
                self.assertEqual(saved[0]["indices_sha256"], rows[0]["indices_sha256"])
                self.assertEqual(cifar10._partition_plan.cache_info().misses, 1)

    def test_server_saves_each_runs_csv_before_training_with_unchanged_schema(self):
        from fl_defense.server_app import main

        images = np.zeros((len(self.labels), 1, 1, 1), dtype=np.uint8)
        model = SimpleNamespace(name="synthetic", get_weights=lambda: [np.zeros(1, np.float32)])
        result = SimpleNamespace(train_metrics_clientapp={}, evaluate_metrics_clientapp={},
                                 evaluate_metrics_serverapp={})
        with tempfile.TemporaryDirectory() as root:
            shared = Path(root) / "partition_summary.csv"
            shared.write_text("existing experiment summary\n")
            saved_files = {}
            for run_id, population, method in ((321, 3, "iid"), (654, 5, "dirichlet")):
                config = {"num-clients": population, "num-server-rounds": 1,
                          "results-dir": root, "dataset-dir": root, "partition-method": method,
                          "fraction-train": 1.0, "clustering-enabled": False,
                          "attacks-enabled": False, "save-client-weights": False, "save-model": False}
                grid = SimpleNamespace(get_node_ids=lambda: list(range(population)))

                def before_training(**kwargs):
                    paths = list((Path(root) / "tensorboard").glob(f"run_{run_id}_*/partition_summary.csv"))
                    self.assertEqual(len(paths), 1)
                    path = paths[0]
                    with path.open(newline="") as stream:
                        reader = csv.DictReader(stream)
                        saved = list(reader)
                        plan = cifar10._partition_plan(root, population, PartitionSettings(method=method))
                        expected = partition_statistics(self.labels, plan, PartitionSettings(method=method), run_id)
                        self.assertEqual(reader.fieldnames, list(expected[0]))
                        self.assertEqual(saved, [{k: str(v) for k, v in row.items()} for row in expected])
                    self.assertEqual(len(saved), population)
                    saved_files[path] = path.read_bytes()
                    return result

                with patch.object(cifar10, "_load_local_training_data", return_value=(images, self.labels)), \
                        patch("fl_defense.data.partition_summary._load_local_training_data", return_value=(images, self.labels)), \
                        patch("fl_defense.server_app.load_model", return_value=model), \
                        patch("fl_defense.server_app.log_dataset"), \
                        patch("fl_defense.server_app.LoggedFedAvg") as strategy, \
                        contextlib.redirect_stdout(io.StringIO()):
                    strategy.return_value.start.side_effect = before_training
                    main(grid, SimpleNamespace(run_id=run_id, run_config=config))
                    strategy.return_value.start.assert_called_once()
                self.assertEqual(shared.read_text(), "existing experiment summary\n")
                for path, content in saved_files.items():
                    self.assertEqual(path.read_bytes(), content)
            self.assertEqual(len(saved_files), 2)

    def test_summary_requires_explicit_destination(self):
        with self.assertRaises(TypeError):
            prepare_partition_summary({"results-dir": "/tmp"}, 3, 123)


if __name__ == "__main__":
    unittest.main()
