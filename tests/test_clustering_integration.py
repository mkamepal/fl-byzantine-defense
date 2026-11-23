"""Synthetic Flower message/strategy lifecycle, without a costly simulation."""

import json
import tempfile
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg

from fl_defense.aggregation.logged_fedavg import LoggedFedAvg
from fl_defense.client_app import client_info
from fl_defense.detection.clustering import ClusteringConfig, ClusteringDefense
from fl_defense.detection.parameters import model_schema, schema_json
from fl_defense.utils.tensorboard import ExperimentLogger
from fl_defense.utils.weight_archive import WeightArchive, load_weights
from test_clustering import SCHEMA, baseline, synthetic_clients
from test_tensorboard import events, value


class SyntheticGrid:
    """Test doubles local optimization with declared synthetic model deltas."""
    def __init__(self):
        self.clients = synthetic_clients()
        self.expected = {}

    def get_node_ids(self):
        return [100 + int(cid.split("_")[1]) for cid in self.clients]

    def push_messages(self, messages):
        messages = list(messages)
        for message in messages:
            # Simulate the transport assigning its otherwise read-only ID.
            message.metadata.__dict__["_message_id"] = uuid.uuid4().hex
        self.pending_replies = self.send_and_receive(messages)
        return [message.metadata.message_id for message in messages]

    def pull_messages(self, message_ids):
        replies = self.pending_replies
        self.pending_replies = []
        return replies

    def send_and_receive(self, messages, **kwargs):
        replies = []
        for message in messages:
            partition = message.metadata.dst_node_id - 100
            cid = f"client_{partition}"
            server_round = message.content["config"]["server-round"]
            training = message.metadata.message_type == "train"
            info = client_info(partition, 1)
            metrics = {"num-examples": partition + 1,
                       "train_loss" if training else "eval_loss": .5,
                       "train_acc" if training else "eval_acc": .8}
            content = RecordDict({"metrics": MetricRecord(metrics), "client-info": info})
            if training:
                base = message.content["arrays"].to_numpy_ndarrays()
                synthetic, _ = self.clients[cid]
                weights = [base[i] + synthetic[i] - baseline()[i] for i in range(len(SCHEMA))]
                self.expected.setdefault(server_round, {})[cid] = weights
                content["arrays"] = ArrayRecord(weights)
                info["weight-schema"] = schema_json(SCHEMA)
                info["weight-convention"] = "full-model"
                info["server-round"] = server_round
            replies.append(Message(content, reply_to=message))
        return replies


class ClusteringIntegrationTests(unittest.TestCase):
    def setUp(self):
        try:
            from flwr.supercore.task_identity import TaskIdentity
        except ImportError:
            return
        for name, setting in (("_run_id", 99), ("_task_id", 1), ("_node_id", 0)):
            fixture = patch.object(TaskIdentity, name, setting)
            fixture.start()
            self.addCleanup(fixture.stop)

    def test_small_population_rejected_before_data_loading_or_training(self):
        from fl_defense.server_app import main
        context = SimpleNamespace(run_config={"num-server-rounds": 1, "fraction-train": 1.0,
                                              "num-clients": 2, "clustering-enabled": True})
        with patch("fl_defense.server_app.load_model") as load, patch("fl_defense.server_app.wait_for_clients") as wait:
            with self.assertRaisesRegex(ValueError, "at least 3"):
                main(Mock(), context)
        load.assert_not_called()
        wait.assert_not_called()
        # A valid total population can still yield an invalid partial sample.
        with tempfile.TemporaryDirectory() as root, ExperimentLogger(
                root, 12, {"num-clients": 15, "num-server-rounds": 1}, "synthetic") as logger:
            defense = ClusteringDefense(SCHEMA)
            strategy = LoggedFedAvg(logger, 15, clustering_defense=defense, fraction_train=.25)
            grid = SyntheticGrid()
            with patch.object(grid, "push_messages") as send, self.assertRaisesRegex(ValueError, "N=2"):
                strategy.start(grid=grid, initial_arrays=ArrayRecord(baseline()), num_rounds=1)
            send.assert_not_called()
            self.assertIsNone(defense.reference)
            self.assertTrue((logger.run_dir / "clustering" / "round_001_failure.json").exists())

    def test_synthetic_two_round_flower_lifecycle_and_original_weight_aggregation(self):
        grid = SyntheticGrid()
        with tempfile.TemporaryDirectory() as root, ExperimentLogger(
                root, 99, {"num-clients": 15, "num-server-rounds": 2, "clustering-enabled": True}, "synthetic") as logger:
            archive = WeightArchive(logger.run_dir, 15)
            defense = ClusteringDefense(SCHEMA, ClusteringConfig(save_update_vectors=True, save_pca_loadings=True))
            strategy = LoggedFedAvg(logger, 15, archive, defense)
            result = strategy.start(grid=grid, initial_arrays=ArrayRecord(baseline()), num_rounds=2)
            expected_previous = baseline()
            for rnd in (1, 2):
                record = json.loads((logger.run_dir / "clustering" / f"round_{rnd:03d}.json").read_text())
                selected = [cid for cid, score in zip(record["client_ids"], record["clustering_scores"]) if score]
                self.assertEqual(selected, ["client_0", "client_1", "client_2", "client_3"])
                expected = [np.average([grid.expected[rnd][cid][i] for cid in selected], axis=0,
                                       weights=[int(cid.split("_")[1]) + 1 for cid in selected]) for i in range(len(SCHEMA))]
                before = load_weights(archive.run_dir / f"round_{rnd:03d}_global_before.npz")
                after = load_weights(archive.run_dir / f"round_{rnd:03d}_global_after.npz")
                for i in range(len(SCHEMA)):
                    np.testing.assert_allclose(before[i], expected_previous[i], rtol=1e-6)
                    np.testing.assert_allclose(after[i], expected[i], rtol=1e-6)
                expected_previous = expected
                for cid in grid.clients:  # Rejected models remain available for research.
                    self.assertTrue((archive.run_dir / f"round_{rnd:03d}_{cid}.npz").exists())
                with np.load(logger.run_dir / "clustering" / f"round_{rnd:03d}_analysis.npz") as arrays:
                    self.assertEqual(arrays["update_vectors"].shape, (8, 5))
                    expected_delta = grid.expected[rnd]["client_0"][0] - before[0]
                    np.testing.assert_array_equal(arrays["update_vectors"][0, :3], expected_delta)
                self.assertGreater((logger.run_dir / "clustering" / f"round_{rnd:03d}.png").stat().st_size, 1000)
            self.assertIsNone(defense.reference)
            for i, weights in enumerate(result.arrays.to_numpy_ndarrays()):
                np.testing.assert_allclose(weights, expected_previous[i], rtol=1e-6)
            reader = events(logger.run_dir)
            self.assertEqual(float(value(reader, "detection/clustering/selected_clients")), 4)
            self.assertEqual(float(value(reader, "detection/clustering/client_14/score")), 0)
            self.assertEqual(float(value(reader, "server/successful_updates")), 8)
            self.assertAlmostEqual(float(value(reader, "server/global_accuracy")), .8, places=6)

    def test_different_populations_and_round_counts_across_experiments(self):
        for count, rounds, components in ((1, 2, None), (2, 2, 1), (3, 1, 2), (5, 3, 2), (15, 2, 2)):
            with self.subTest(clients=count, rounds=rounds), tempfile.TemporaryDirectory() as root:
                grid = SyntheticGrid()
                grid.clients = {}
                for i in range(count):
                    weights = baseline()
                    sign = 1 if i < (count + 1) // 2 else -1
                    weights[0] += np.array([sign, (i + 1) * .03, (i + 1)**2 * .001], np.float32)
                    grid.clients[f'client_{i}'] = weights, schema_json(SCHEMA)
                config = {'num-clients': count, 'num-server-rounds': rounds,
                          'clustering-enabled': components is not None}
                with ExperimentLogger(root, 33, config, 'synthetic') as logger:
                    defense = ClusteringDefense(SCHEMA, ClusteringConfig(n_components=components)) if components else None
                    strategy = LoggedFedAvg(logger, count, clustering_defense=defense)
                    result = strategy.start(grid=grid, initial_arrays=ArrayRecord(baseline()), num_rounds=rounds)
                    self.assertEqual(len(result.train_metrics_clientapp), rounds)
                    self.assertEqual(len(result.evaluate_metrics_clientapp), rounds)
                    reports = sorted((logger.run_dir / 'clustering').glob('round_*.json'))
                    self.assertEqual(len(reports), rounds if components else 0)
                    for report in reports:
                        data = json.loads(report.read_text())
                        self.assertEqual(data['client_ids'], [f'client_{i}' for i in range(count)])
                        self.assertEqual(np.shape(data['pca_coordinates']), (count, components))

    def test_optional_snapshots_do_not_affect_reference_or_compact_outputs(self):
        with tempfile.TemporaryDirectory() as root, ExperimentLogger(
                root, 10, {"num-clients": 15, "num-server-rounds": 1}, "synthetic") as logger:
            defense = ClusteringDefense(SCHEMA)
            strategy = LoggedFedAvg(logger, 15, clustering_defense=defense)
            grid = SyntheticGrid()
            strategy.start(grid=grid, initial_arrays=ArrayRecord(baseline()), num_rounds=1)
            self.assertFalse((logger.run_dir / "models").exists())
            self.assertTrue((logger.run_dir / "clustering" / "round_001.json").exists())
            self.assertFalse(list(logger.run_dir.rglob("*.npz")))

    def test_invalid_replies_abort_before_fedavg_and_persist_failure(self):
        for problem in ("missing_head", "schema", "round", "convention", "keys", "duplicate", "degenerate"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as root, ExperimentLogger(
                    root, 12, {"num-clients": 15, "num-server-rounds": 1}, "synthetic") as logger:
                defense = ClusteringDefense(SCHEMA)
                strategy = LoggedFedAvg(logger, 15, clustering_defense=defense)
                grid = SyntheticGrid()
                messages = strategy.configure_train(1, ArrayRecord(baseline()), ConfigRecord(), grid)
                replies = grid.send_and_receive(messages)
                if problem == "missing_head":
                    replies = [m for m in replies if m.content["client-info"]["client-id"] != "client_0"]
                elif problem == "duplicate":
                    replies.append(replies[0])
                elif problem == "degenerate":
                    for message in replies:
                        message.content["arrays"] = ArrayRecord(baseline())
                elif problem == "keys":
                    arrays = replies[0].content["arrays"]
                    arrays["wrong"] = arrays.pop("0")
                else:
                    field = {"schema": "weight-schema", "round": "server-round", "convention": "weight-convention"}[problem]
                    replies[0].content["client-info"][field] = "invalid"
                with patch.object(FedAvg, "aggregate_train") as aggregate, self.assertRaises((ValueError, KeyError)):
                    strategy.aggregate_train(1, replies)
                aggregate.assert_not_called()
                self.assertIsNone(defense.reference)
                report = json.loads((logger.run_dir / "clustering" / "round_001_failure.json").read_text())
                self.assertEqual(report["status"], "aborted")

    def test_real_keras_metadata_trainable_biases_and_bn_parameters(self):
        import keras
        from fl_defense.models.resnet56 import load_model
        keras.backend.clear_session()
        server = load_model()
        schema = model_schema(server)
        expected_dimension = sum(int(np.prod(v.shape)) for v in server.trainable_weights)
        self.assertEqual(sum(int(np.prod(p.shape)) for p in schema if p.trainable), expected_dimension)
        self.assertTrue(any(p.name.endswith("bias") and p.trainable for p in schema))
        self.assertTrue(any(p.name.endswith("gamma") and p.trainable for p in schema))
        self.assertTrue(any(p.name.endswith("beta") and p.trainable for p in schema))
        self.assertTrue(any(p.name.endswith("moving_mean") and not p.trainable for p in schema))
        keras.backend.clear_session()
        client = load_model()
        info = client_info(10, 2, client, 3)
        self.assertEqual(info["weight-schema"], schema_json(schema))
        self.assertEqual(info["server-round"], 3)
        self.assertEqual(info["weight-convention"], "full-model")
        # Frozen layer weights must not leak into the trainable vector.
        client.layers[-1].trainable = False
        self.assertLess(sum(int(np.prod(p.shape)) for p in model_schema(client) if p.trainable), expected_dimension)
        del server, client
        keras.backend.clear_session(free_memory=True)

    def test_archive_validation_failure_is_also_recorded(self):
        with tempfile.TemporaryDirectory() as root, ExperimentLogger(
                root, 12, {"num-clients": 15, "num-server-rounds": 1}, "synthetic") as logger:
            archive = WeightArchive(logger.run_dir, 15)
            defense = ClusteringDefense(SCHEMA)
            strategy = LoggedFedAvg(logger, 15, archive, defense)
            grid = SyntheticGrid()
            messages = strategy.configure_train(1, ArrayRecord(baseline()), ConfigRecord(), grid)
            replies = grid.send_and_receive(messages)
            replies[0].content["arrays"] = ArrayRecord(baseline()[:-1])
            with patch.object(FedAvg, "aggregate_train") as aggregate, self.assertRaises(ValueError):
                strategy.aggregate_train(1, replies)
            aggregate.assert_not_called()
            self.assertTrue((logger.run_dir / "clustering" / "round_001_failure.json").exists())
            self.assertIsNone(defense.reference)
            self.assertIsNone(archive.baseline)


if __name__ == "__main__":
    unittest.main()
