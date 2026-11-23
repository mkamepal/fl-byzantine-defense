"""Read back real event files and verify unmodified FedAvg results."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from flwr.app import ArrayRecord, ConfigRecord, Error, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg

from fl_defense.aggregation.logged_fedavg import LoggedFedAvg
from fl_defense.client_app import client_info
from fl_defense.data import cifar10
from fl_defense.data.partition_summary import prepare_partition_summary, CLASS_NAMES
from fl_defense.data.partitioning import PartitionSettings
from fl_defense.utils.tensorboard import ExperimentLogger, log_dataset
from fl_defense.utils.image_previews import log_labeled_previews
from fl_defense.server_app import wait_for_clients
from fl_defense.utils.weight_archive import WeightArchive, load_weights


def events(path):
    return EventAccumulator(str(path), size_guidance={"tensors": 0}).Reload()


def value(reader, tag):
    return tf.make_ndarray(reader.Tensors(tag)[-1].tensor_proto)


class TensorBoardTests(unittest.TestCase):
    def setUp(self):
        # Newer Flower releases normally populate this in the runtime process.
        try:
            from flwr.supercore.task_identity import TaskIdentity
        except ImportError:
            return
        for name, value in (("_run_id", 99), ("_task_id", 1), ("_node_id", 0)):
            fixture = patch.object(TaskIdentity, name, value)
            fixture.start()
            self.addCleanup(fixture.stop)

    def test_startup_waits_for_registration(self):
        from unittest.mock import Mock
        grid = Mock()
        grid.get_node_ids.side_effect = [[], [100], [100, 101]]
        with patch("fl_defense.server_app.time.sleep"):
            wait_for_clients(grid, 2)
        grid.get_node_ids.side_effect = None
        grid.get_node_ids.return_value = [100]
        with self.assertRaisesRegex(ValueError, "Expected 2"):
            wait_for_clients(grid, 2, timeout=0)

    def test_isolation_flush_close_and_custom_summaries(self):
        with tempfile.TemporaryDirectory() as root:
            config = {"num-clients": 15, "num-server-rounds": 2}
            with ExperimentLogger(root, 17, config, "resnet56") as first:
                first.scalar("detection/precision", 0.75, 1)
                first.histogram("detection/update_magnitude", [0.1, 0.3, 0.7], 1)
                first.client_event("client_0", "trust", 1, {"score": 0.9}, "trusted head")
                first.flush()
                self.assertAlmostEqual(float(value(events(first.run_dir), "detection/precision")), 0.75)
            before = {p.name: p.read_bytes() for p in first.run_dir.iterdir()}
            with self.assertRaisesRegex(RuntimeError, "deliberate"):
                with ExperimentLogger(root, 17, config, "resnet56") as second:
                    raise RuntimeError("deliberate")
            self.assertNotEqual(first.run_dir, second.run_dir)
            self.assertTrue(first.closed and second.closed)
            self.assertEqual(before, {p.name: p.read_bytes() for p in first.run_dir.iterdir()})
            self.assertEqual(json.loads((first.run_dir / "status.json").read_text())["status"], "completed")
            self.assertEqual(json.loads((second.run_dir / "status.json").read_text())["status"], "failed")
            self.assertEqual(json.loads((first.run_dir / "config.json").read_text())["model_name"], "resnet56")
            saved_config = json.loads((first.run_dir / "config.json").read_text())
            self.assertTrue(saved_config["client_train_run_eagerly"])
            self.assertFalse(saved_config["client_evaluate_run_eagerly"])
            self.assertFalse(any(p.is_dir() for p in first.run_dir.iterdir()))

    def test_all_clients_images_labels_counts_and_heatmaps(self):
        labels = np.tile(np.arange(10), 30)
        images = np.random.default_rng(42).integers(0, 256, (300, 32, 32, 3), dtype=np.uint8)
        for method in ("iid", "dirichlet"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as root:
                cifar10._partition_plan.cache_clear()
                config = {"num-clients": 15, "num-server-rounds": 1, "results-dir": root,
                          "dataset-dir": root, "partition-method": method}
                with patch.object(cifar10, "_load_local_training_data", return_value=(images, labels)), \
                     patch("fl_defense.data.partition_summary._load_local_training_data", return_value=(images, labels)), \
                     ExperimentLogger(root, 20, config, "resnet56") as logger:
                    with contextlib.redirect_stdout(io.StringIO()):
                        rows = prepare_partition_summary(config, 15, 20, logger.run_dir / "partition_summary.csv")
                    log_dataset(logger, config, rows)
                    reader = events(logger.run_dir)
                    plan = cifar10._partition_plan(root, 15, PartitionSettings(method=method))
                    for i in range(15):
                        indices = plan[i].train_indices[:5]
                        self.assertEqual(len(indices), 5)
                        encoded = value(reader, f"dataset/client_{i}/images")
                        decoded = tf.image.decode_png(encoded[2], channels=3).numpy()
                        directory = logger.run_dir / 'images' / 'clean_training' / f'client_{i}'
                        saved = tf.image.decode_png((directory / 'preview.png').read_bytes(), channels=3).numpy()
                        np.testing.assert_array_equal(decoded, saved)
                        records = json.loads((directory / 'labels.json').read_text())
                        self.assertEqual(len(records), len(indices))
                        for position, index in enumerate(indices):
                            record = records[position]
                            self.assertEqual(record['source_index'], int(index))
                            self.assertEqual(record['label_name'], CLASS_NAMES[labels[index]])
                            self.assertIn(CLASS_NAMES[labels[index]].lower(), record['file'])
                            tile = tf.image.decode_png((directory / record['file']).read_bytes(), channels=3).numpy()
                            np.testing.assert_array_equal(decoded[:, position * 224:(position + 1) * 224], tile)
                            np.testing.assert_array_equal(tile[78:206:4, 48:176:4], images[index])
                            self.assertTrue(np.any(tile[32:54] != 255))  # Visible class-name caption.
                        label_text = value(reader, f"dataset/client_{i}/image_labels").item().decode()
                        summary = value(reader, f"dataset/client_{i}/summary").item().decode()
                        for index in indices:
                            self.assertIn(f"| {index} | {CLASS_NAMES[labels[index]]} |", label_text)
                        self.assertIn(f"**Total Images:** {len(plan[i].indices)}", summary)
                        for label, name in enumerate(CLASS_NAMES):
                            total = np.count_nonzero(labels[plan[i].indices] == label)
                            train = np.count_nonzero(labels[plan[i].train_indices] == label)
                            validation = np.count_nonzero(labels[plan[i].validation_indices] == label)
                            self.assertIn(f"| {name} | {total} | {train} | {validation} |", summary)
                    for name in ("class_distribution", "training_distribution", "validation_distribution"):
                        self.assertGreater(len(value(reader, f"dataset/{name}")[2]), 1000)
                    rows[0]["airplane"] += 1
                    with self.assertRaisesRegex(ValueError, "differs"):
                        log_dataset(logger, config, rows)
        cifar10._partition_plan.cache_clear()

    def test_attack_preview_labels_originals_and_pixels_are_preserved(self):
        images = np.zeros((2, 32, 32, 3), dtype=np.float32)
        images[:, -2:, -2:] = 1.0
        before = images.copy()
        labels = np.array([0, 0])
        originals = np.array([5, 0])
        with tempfile.TemporaryDirectory() as root, \
                ExperimentLogger(root, 1, {'num-clients': 2, 'num-server-rounds': 1}, 'test') as logger:
            log_labeled_previews(logger, 'attacks/client_1/training_images', images, labels,
                                 np.array([17, 33]), 'client_1', 'attack_training',
                                 original_labels=originals, poisoned_mask=np.array([True, True]), attack='backdoor')
            logger.flush()
            directory = logger.run_dir / 'images/attack_training/client_1'
            records = json.loads((directory / 'labels.json').read_text())
            self.assertEqual([r['file'] for r in records], ['01_row_17_airplane_from_dog.png', '02_row_33_airplane.png'])
            self.assertEqual([r['original_label_name'] for r in records], ['Dog', 'Airplane'])
            self.assertTrue(all(r['poisoned'] for r in records))
            encoded = value(events(logger.run_dir), 'attacks/client_1/training_images')[2]
            decoded = tf.image.decode_png(encoded, channels=3).numpy()
            saved = tf.image.decode_png((directory / 'preview.png').read_bytes(), channels=3).numpy()
            np.testing.assert_array_equal(decoded, saved)
            for i in range(2):
                np.testing.assert_array_equal(decoded[78:206:4, i*224+48:i*224+176:4],
                                              (images[i] * 255).astype(np.uint8))
            np.testing.assert_array_equal(images, before)
            np.testing.assert_array_equal(labels, [0, 0])
            np.testing.assert_array_equal(originals, [5, 0])
            with self.assertRaisesRegex(ValueError, 'actually poisoned'):
                log_labeled_previews(logger, 'unused', images, labels, np.array([17, 33]),
                                     'client_1', 'after_attack', original_labels=originals,
                                     poisoned_mask=np.array([True, False]), attack='backdoor')
            with self.assertRaises(FileExistsError):
                log_labeled_previews(logger, 'unused', images, labels, np.array([17, 33]),
                                     'client_1', 'attack_training')

    def test_strategy_matches_fedavg_and_logs_real_metrics(self):
        class Grid:
            def get_node_ids(self):
                return [100, 101, 102]

        def replies(training):
            result = []
            for partition, (weight, loss) in enumerate(((10, 1.0), (30, 3.0))):
                metrics = {"num-examples": weight,
                           "train_loss" if training else "eval_loss": loss,
                           "train_acc" if training else "eval_acc": loss / 10}
                request = Message(RecordDict(), dst_node_id=100 + partition,
                                  message_type="train" if training else "evaluate")
                content = RecordDict({"metrics": MetricRecord(metrics),
                                      "client-info": client_info(partition, 2)})
                if training:
                    content["arrays"] = ArrayRecord([np.array([loss], dtype=np.float32)])
                result.append(Message(content, reply_to=request))
            failed = Message(RecordDict(), dst_node_id=102, message_type="train" if training else "evaluate")
            result.append(Message(error=Error(1, "test failure"), reply_to=failed))
            return result

        with tempfile.TemporaryDirectory() as root:
            with ExperimentLogger(root, 99, {"num-clients": 3, "num-server-rounds": 1}, "resnet56") as logger:
                strategy = LoggedFedAvg(logger, 3, weight_archive=WeightArchive(logger.run_dir, 3))
                arrays = ArrayRecord([np.zeros(1, dtype=np.float32)])
                strategy.configure_train(1, arrays, ConfigRecord(), Grid())
                expected_arrays, expected_metrics = FedAvg().aggregate_train(1, replies(True))
                actual_arrays, actual_metrics = strategy.aggregate_train(1, replies(True))
                np.testing.assert_array_equal(actual_arrays.to_numpy_ndarrays()[0], expected_arrays.to_numpy_ndarrays()[0])
                self.assertEqual(dict(actual_metrics), dict(expected_metrics))
                directory = logger.run_dir / "models"
                np.testing.assert_array_equal(load_weights(directory / "round_001_global_before.npz")[0], [0])
                np.testing.assert_array_equal(load_weights(directory / "round_001_client_0.npz")[0], [1])
                np.testing.assert_array_equal(load_weights(directory / "round_001_client_1.npz")[0], [3])
                np.testing.assert_array_equal(load_weights(directory / "round_001_global_after.npz")[0], [2.5])
                manifest = json.loads((directory / "round_001_weight_shifts.json").read_text())
                self.assertEqual(len(manifest["clients"]), 2)
                self.assertEqual(len(manifest["errors"]), 1)
                self.assertEqual(manifest["clients"][0]["shift"]["l2"], 1)
                self.assertFalse((directory / "round_001_client_2.npz").exists())
                strategy.configure_evaluate(1, arrays, ConfigRecord(), Grid())
                expected = FedAvg().aggregate_evaluate(1, replies(False))
                actual = strategy.aggregate_evaluate(1, replies(False))
                self.assertEqual(dict(actual), dict(expected))
                reader = events(logger.run_dir)  # Round summaries are already flushed.
                self.assertAlmostEqual(float(value(reader, "server/global_accuracy")), 0.25)
                self.assertEqual(float(value(reader, "server/global_loss")), 2.5)
                self.assertEqual(float(value(reader, "server/participating_clients")), 3)
                self.assertEqual(float(value(reader, "server/successful_updates")), 2)
                self.assertEqual(float(value(reader, "training/client_0/num_samples")), 10)
                self.assertEqual(float(value(reader, "training/client_0/local_epochs")), 2)
                self.assertEqual(float(value(reader, "validation/client_1/loss")), 3)
                self.assertEqual(float(value(reader, "weight_shift/client_1/l2")), 3)
                self.assertEqual(reader.Tensors("validation/client_1/loss")[0].step, 1)
                self.assertNotIn("training/client_2/loss", reader.Tags()["tensors"])
                strategy.aggregate_train(2, [])
                self.assertEqual(float(value(events(logger.run_dir), "server/successful_updates")), 0)


if __name__ == "__main__":
    unittest.main()
