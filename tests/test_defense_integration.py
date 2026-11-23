"""Normal launcher -> ServerApp -> Flower rounds -> automatic Excel snapshot.

Only data/model size and transport/local optimization are synthetic. PCA,
HDBSCAN, all four decisions, Keras trusted inference, FedAvg, and exports run.
This is integration verification, not evidence of attack detection.
"""

import json
import os
from pathlib import Path
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.common.config import parse_config_args
from openpyxl import load_workbook

import run
from fl_defense.aggregation.logged_fedavg import LoggedFedAvg
from fl_defense.client_app import client_info
from fl_defense.detection.clustering import ClusteringConfig, ClusteringDefense
from fl_defense.detection.combined import CombinedDefense
from fl_defense.detection.defense_config import DefenseConfig
from fl_defense.detection.trust_score import combine_scores
from fl_defense.detection.parameters import model_schema, schema_json
from fl_defense.detection.validation import TrustedEvaluator
from fl_defense.utils.defense_excel import COLUMNS, round_tables
from fl_defense.utils.tensorboard import ExperimentLogger
from test_clustering_integration import SyntheticGrid
from test_defense import tiny_model


class ModelGrid(SyntheticGrid):
    def __init__(self, model, ids=(0, 1, 2), fail_evaluation_round=None):
        self.schema = model_schema(model)
        self.ids = ids
        self.expected, self.before = {}, {}
        self.fail_evaluation_round = fail_evaluation_round

    def get_node_ids(self):
        return [100 + i for i in self.ids]

    def send_and_receive(self, messages, **kwargs):
        replies = []
        for message in messages:
            partition = message.metadata.dst_node_id - 100
            cid = f"client_{partition}"
            rnd = message.content["config"]["server-round"]
            training = message.metadata.message_type == "train"
            if not training and rnd == self.fail_evaluation_round:
                raise RuntimeError("synthetic evaluation failure")
            info = client_info(partition, 1)
            metrics = {"num-examples": partition + 1,
                       "train_loss" if training else "eval_loss": .5,
                       "train_acc" if training else "eval_acc": .8}
            content = RecordDict({"metrics": MetricRecord(metrics), "client-info": info})
            if training:
                base = message.content["arrays"].to_numpy_ndarrays()
                self.before[rnd] = base
                weights = [w.copy() for w in base]
                index = self.ids.index(partition)
                sign = 1 if index < 2 else -1
                for j, p in enumerate(self.schema):
                    if p.trainable:
                        # Noncollinear updates, varied magnitudes, and unequal
                        # sample counts distinguish analysis and aggregation.
                        weights[j] += np.full(p.shape, sign * (.05 + index * .003), np.float32)
                        weights[j].flat[0] += .007 * index
                    elif p.name.endswith("moving_mean"):
                        weights[j] += .01 * index
                self.expected.setdefault(rnd, {})[cid] = weights
                content["arrays"] = ArrayRecord(weights)
                info.update({"weight-schema": schema_json(self.schema), "weight-convention": "full-model", "server-round": rnd})
            replies.append(Message(content, reply_to=message))
        return replies


class CombinedIntegrationTests(unittest.TestCase):
    def setUp(self):
        try:
            from flwr.supercore.task_identity import TaskIdentity
        except ImportError:
            return
        for name, setting in (("_run_id", 99), ("_task_id", 1), ("_node_id", 0)):
            fixture = patch.object(TaskIdentity, name, setting)
            fixture.start()
            self.addCleanup(fixture.stop)

    def test_trust_cutoff_filters_full_models_and_prints_actual_client_lists(self):
        for ids in ((0, 1, 2), (0, 2, 10)):
            with self.subTest(ids=ids), tempfile.TemporaryDirectory() as root:
                model = tiny_model()
                grid = ModelGrid(model, ids=ids)
                names = ('clustering', 'direction', 'magnitude', 'validation')
                scores = {f'client_{ids[0]}': (1, 1, 1, 1), f'client_{ids[1]}': (1, 1, 1, 0),
                          f'client_{ids[2]}': (1, 0, 1, 0)}
                components = {name: {cid: values[i] for cid, values in scores.items()}
                              for i, name in enumerate(names)}
                cfg = DefenseConfig(trust_threshold=.51)
                trust = combine_scores(components, cfg.weights, cfg.trust_threshold)
                record = {'round': 1, 'clients': {cid: {**trust[cid],
                    **{name: {'score': components[name][cid]} for name in names}} for cid in reversed(scores)}}
                evaluator = SimpleNamespace(dataset_identifier='synthetic', source='synthetic')
                combined = CombinedDefense(evaluator, cfg)
                with ExperimentLogger(root, 99, {'num-clients': max(ids)+1, 'num-server-rounds': 1}, model.name) as logger:
                    strategy = LoggedFedAvg(logger, max(ids)+1, combined_defense=combined,
                                            clustering_defense=ClusteringDefense(model_schema(model),
                                                ClusteringConfig(n_components=2, min_cluster_size=2)))
                    requests = strategy.configure_train(1, ArrayRecord(model.get_weights()), ConfigRecord(), grid)
                    replies = grid.send_and_receive(requests)
                    with patch.object(combined, 'analyze', return_value=record), self.assertLogs('flwr', level='INFO') as logs:
                        arrays, _ = strategy.aggregate_train(1, replies)
                    summaries = [r.getMessage().strip() for r in logs.records if 'global aggregation (successful' in r.getMessage()]
                    self.assertEqual(summaries, [f'Round 1 global aggregation (successful client updates; trust >= 0.51)\n'
                        f'Accepted clients (2): client_{ids[0]} (trust=1), client_{ids[1]} (trust=0.75)\n'
                        f'Rejected clients (1): client_{ids[2]} (trust=0.5)'])
                    for index, actual in enumerate(arrays.to_numpy_ndarrays()):
                        expected = np.average([grid.expected[1][f'client_{i}'][index] for i in ids[:2]],
                                              axis=0, weights=[i+1 for i in ids[:2]])
                        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_normal_runner_two_rounds_automatic_workbook_matches_actual_decisions(self):
        from fl_defense.server_app import main
        run_id = 17619004868423387106
        model = tiny_model()
        grid = ModelGrid(model)
        defaults = tomllib.loads(Path("pyproject.toml").read_text())["tool"]["flwr"]["app"]["config"]
        with tempfile.TemporaryDirectory() as directory:
            # Optional retention is for inspecting this test's output. Ordinary
            # experiments need no environment option or post-run export command.
            project = Path(os.environ.get("FL_SYNTHETIC_PROJECT", directory))
            config_file = Path(directory) / "run.json"
            config_file.write_text(json.dumps({"simulation": {"num-supernodes": 3}, "training": {
                "num-server-rounds": 2, "clustering-enabled": True, "save-client-weights": False,
                "clustering-n-components": 2, "clustering-min-cluster-size": 2,
                "save-model": False, "batch-size": 2, "fraction-train": 1.0}}))
            strategies = []

            def strategy(*args, **kwargs):
                result = LoggedFedAvg(*args, **kwargs)
                strategies.append(result)
                return result

            def launch(command, **kwargs):
                training = parse_config_args([command[command.index("--run-config") + 1]])
                context = SimpleNamespace(run_id=run_id, run_config={**defaults, **training})
                main(grid, context)
                return 0

            def evaluator(source, x, y, batch_size, description):
                return TrustedEvaluator(source, x, y, batch_size, "synthetic clean holdout for integration verification",
                                        dataset_identifier="synthetic classification data (integration check only)")

            with patch.object(run, "__file__", str(project / "run.py")), \
                    patch.object(run, "run_command", side_effect=launch), \
                    patch("fl_defense.server_app.load_model", return_value=model), \
                    patch("fl_defense.server_app.prepare_partition_summary", return_value=[]), \
                    patch("fl_defense.server_app.log_dataset"), \
                    patch("fl_defense.server_app.load_client_split", return_value=(
                        np.array([[1, 0], [-1, 0], [1, 0]], np.float32), np.array([0, 1, 1]))) as split, \
                    patch("fl_defense.server_app.TrustedEvaluator", side_effect=evaluator), \
                    patch("fl_defense.server_app.LoggedFedAvg", side_effect=strategy):
                self.assertEqual(run.main(["--config", str(config_file)]), 0)
                self.assertEqual(split.call_args.args[:3], (0, 3, "validation"))
            strategy = strategies[0]
            path = strategy.logger.run_dir / "defense_scores.xlsx"
            self.assertTrue(path.is_file())
            self.assertEqual(path.parent.parent, project / "results" / "tensorboard")
            self.assertEqual(set(strategy.completed_defense_rounds), {1, 2})
            for name in ("clustering", "direction", "magnitude", "validation"):
                directory = path.parent / "confusion_matrix"
                report = json.loads((directory / f"{name}.json").read_text())
                self.assertTrue((directory / f"{name}.png").is_file())
                self.assertEqual(report["completed_rounds"], [1, 2])
                self.assertEqual(report["overall"]["observations"], 6)
                rejected = sum(v[name]["score"] == 0
                               for r in strategy.completed_defense_rounds.values() for v in r["clients"].values())
                self.assertEqual(report["overall"]["matrix"], [[6 - rejected, rejected], [0, 0]])
            book = load_workbook(path)
            try:
                self.assertEqual(dict(list(book["Configuration"].values)[1:])["Run ID"], str(run_id))
                for sheet in COLUMNS:
                    rows = list(book[sheet].values)[1:]
                    self.assertEqual(len(rows), 6)
                    self.assertEqual(len({row[:2] for row in rows}), 6)
                    for row in rows:
                        expected = round_tables(strategy.completed_defense_rounds[row[0]])[sheet][row[:2]]
                        for actual, value in zip(row, expected, strict=True):
                            if type(value) is float:
                                self.assertAlmostEqual(actual, value, places=12)
                            else:
                                self.assertEqual(actual, value)
            finally:
                book.close()
            for rnd in (1, 2):
                records = strategy.completed_defense_rounds[rnd]["clients"]
                for cid, record in records.items():
                    self.assertEqual(record["trust_score"], sum(record[n]["score"] for n in
                                     ("clustering", "direction", "magnitude", "validation")) * .25)
                    self.assertEqual(record["selected"], record["trust_score"] >= .5)
                    self.assertEqual(record["validation"]["example_count"], 3)
                saved = json.loads((path.parent / "defense" / f"round_{rnd:03d}.json").read_text())
                self.assertEqual(saved, strategy.completed_defense_rounds[rnd])
            # The next round receives sample-weighted ORIGINAL full models.
            selected = [cid for cid, v in strategy.completed_defense_rounds[1]["clients"].items() if v["selected"]]
            for i, actual in enumerate(grid.before[2]):
                expected = np.average([grid.expected[1][cid][i] for cid in selected], axis=0,
                                      weights=[int(cid.split("_")[1]) + 1 for cid in selected])
                np.testing.assert_allclose(actual, expected, atol=1e-7)
            self.assertIsNone(strategy.clustering_defense.reference)
            self.assertIsNone(strategy._pending_defense)
            if "FL_SYNTHETIC_PROJECT" in os.environ:
                print(f"SYNTHETIC_WORKBOOK={path}")

    def test_partial_ids_and_later_evaluation_failure_preserve_completed_round(self):
        model = tiny_model()
        grid = ModelGrid(model, ids=(0, 2, 10), fail_evaluation_round=2)
        evaluator = TrustedEvaluator(model, np.array([[1, 0], [-1, 0]], np.float32),
                                     np.array([0, 1]), 2, "synthetic")
        with tempfile.TemporaryDirectory() as root, ExperimentLogger(
                root, 100, {"num-clients": 11, "num-server-rounds": 2}, model.name) as logger:
            strategy = LoggedFedAvg(logger, 11, clustering_defense=ClusteringDefense(
                                        model_schema(model), ClusteringConfig(n_components=2, min_cluster_size=2)),
                                    combined_defense=CombinedDefense(evaluator))
            with self.assertRaisesRegex(RuntimeError, "evaluation failure"):
                strategy.start(grid=grid, initial_arrays=ArrayRecord(model.get_weights()), num_rounds=2)
            self.assertEqual(list(strategy.completed_defense_rounds), [1])
            book = load_workbook(logger.run_dir / "defense_scores.xlsx")
            try:
                rows = list(book["Scores"].values)[1:]
                self.assertEqual([r[:2] for r in rows], [(1, "client_0"), (1, "client_2"), (1, "client_10")])
            finally:
                book.close()
            self.assertFalse((logger.run_dir / "defense" / "round_002.json").exists())
            self.assertFalse((logger.run_dir / "confusion_matrix").exists())

    def test_central_evaluation_failure_does_not_mark_round_complete(self):
        model = tiny_model()
        evaluator = TrustedEvaluator(model, np.array([[1, 0]], np.float32), np.array([0]), 1, "synthetic")
        with tempfile.TemporaryDirectory() as root, ExperimentLogger(
                root, 101, {"num-clients": 3, "num-server-rounds": 1}, model.name) as logger:
            strategy = LoggedFedAvg(logger, 3, clustering_defense=ClusteringDefense(
                                        model_schema(model), ClusteringConfig(n_components=2, min_cluster_size=2)),
                                    combined_defense=CombinedDefense(evaluator))
            def evaluation(rnd, arrays):
                if rnd:
                    raise RuntimeError("central evaluation failed")
            with self.assertRaisesRegex(RuntimeError, "central evaluation"):
                strategy.start(ModelGrid(model), ArrayRecord(model.get_weights()), num_rounds=1, evaluate_fn=evaluation)
            self.assertFalse((logger.run_dir / "defense_scores.xlsx").exists())
            self.assertEqual(strategy.completed_defense_rounds, {})

    def test_disabled_modes_preserve_output_policy_and_skip_trusted_inference(self):
        from fl_defense.server_app import main
        defaults = tomllib.loads(Path("pyproject.toml").read_text())["tool"]["flwr"]["app"]["config"]
        for cluster_enabled, trust_enabled in ((False, True), (True, False)):
            model = tiny_model()
            with self.subTest(clustering=cluster_enabled, trust=trust_enabled), tempfile.TemporaryDirectory() as root:
                context = SimpleNamespace(run_id=102, run_config={**defaults, "num-clients": 3,
                    "num-server-rounds": 1, "results-dir": root, "save-client-weights": False,
                    "clustering-enabled": cluster_enabled, "trust-enabled": trust_enabled})
                with patch("fl_defense.server_app.load_model", return_value=model), \
                        patch("fl_defense.server_app.prepare_partition_summary", return_value=[]), \
                        patch("fl_defense.server_app.log_dataset"), \
                        patch("fl_defense.server_app.load_client_split") as split:
                    main(ModelGrid(model), context)
                split.assert_not_called()
                self.assertFalse(list(Path(root).rglob("defense_scores.xlsx")))
                self.assertFalse(list(Path(root).rglob("defense_config.json")))
                self.assertFalse(list(Path(root).rglob("confusion_matrix")))
                self.assertEqual(len(list(Path(root).rglob("round_001.json"))), int(cluster_enabled))


if __name__ == "__main__":
    unittest.main()
