"""Synthetic partition/training wiring; not evidence that either attack succeeds."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import keras
import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict

from fl_defense import client_app
from fl_defense.aggregation.logged_fedavg import LoggedFedAvg
from fl_defense.attacks.config import AttackConfig, assign_attacks
from fl_defense.attacks.datasets import prepare_attack_datasets, load_training_data
from fl_defense.data import cifar10
from fl_defense.data.partitioning import PartitionSettings
from fl_defense.detection.clustering import ClusteringDefense
from fl_defense.detection.parameters import model_schema
from fl_defense.utils.tensorboard import ExperimentLogger
from test_tensorboard import events, value


def config_for(root, clients=5, **overrides):
    return {'num-clients': clients, 'num-server-rounds': 2, 'dataset-dir': root,
            'results-dir': root, 'local-epochs': 1, 'batch-size': 16, 'learning-rate': .01,
            'attacks-enabled': True, 'attacks-num-malicious': 3, 'partition-method': 'iid',
            **overrides}


def small_model(learning_rate=.01):
    model = keras.Sequential([
        keras.Input((32, 32, 3), name='image'),
        keras.layers.GlobalAveragePooling2D(name='pool'),
        keras.layers.Dense(10, activation='softmax', name='classifier'),
    ], name='synthetic_wiring_model')
    model.compile(optimizer=keras.optimizers.SGD(learning_rate), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'], jit_compile=False)
    return model


class AttackIntegrationTests(unittest.TestCase):
    def setUp(self):
        cifar10._partition_plan.cache_clear()
        self.addCleanup(cifar10._partition_plan.cache_clear)
        rng = np.random.RandomState(5)
        self.images = rng.randint(0, 200, (300, 32, 32, 3)).astype(np.uint8)
        self.labels = np.tile(np.arange(10), 30)
        fixture = patch.object(cifar10, '_load_local_training_data', return_value=(self.images, self.labels))
        fixture.start()
        self.addCleanup(fixture.stop)
        try:
            from flwr.supercore.task_identity import TaskIdentity
        except ImportError:
            return
        for name, setting in (('_run_id', 99), ('_task_id', 1), ('_node_id', 0)):
            fixture = patch.object(TaskIdentity, name, setting)
            fixture.start()
            self.addCleanup(fixture.stop)

    def test_startup_assignments_match_saved_ground_truth(self):
        for clients, requested in ((3, 0), (3, 99), (12, 99)):
            with self.subTest(clients=clients, requested=requested), tempfile.TemporaryDirectory() as root:
                config = config_for(root, clients, **{'attacks-num-malicious': requested, 'verbose': False})
                logger = Mock(run_dir=Path(root))
                with self.assertLogs('flwr', level='INFO') as logs, \
                        patch('fl_defense.attacks.datasets.assign_attacks', wraps=assign_attacks) as assign:
                    prepare_attack_datasets(logger, config)
                assign.assert_called_once()
                report = json.loads((Path(root) / 'attacks/assignment_and_poisoning.json').read_text())
                assignments = report['assignments']
                ordered = [f'client_{i}' for i in range(clients)]
                fake = ', '.join(f'{cid} ({assignments[cid]})' for cid in ordered if cid in assignments) or 'none'
                clean = ', '.join(cid for cid in ordered if cid not in assignments)
                startup = logs.records[0].getMessage()
                self.assertIn(f'Fake clients ({len(assignments)}/{clients}): {fake}', startup)
                self.assertIn(f'Clean clients ({clients - len(assignments)}/{clients}): {clean}', startup)
                self.assertIn('Protected reference clients: client_0', startup)
                self.assertIn('not defense predictions', startup)

    def test_disabled_attacks_print_no_fake_clients_without_preparing_data(self):
        with tempfile.TemporaryDirectory() as root:
            config = config_for(root, clients=5, **{'attacks-enabled': False, 'verbose': False})
            logger = Mock(run_dir=Path(root))
            with self.assertLogs('flwr', level='INFO') as logs, \
                    patch('fl_defense.attacks.datasets.assign_attacks') as assign, \
                    patch('fl_defense.attacks.datasets.load_client_split') as load:
                self.assertIsNone(prepare_attack_datasets(logger, config))
            assign.assert_not_called()
            load.assert_not_called()
            self.assertEqual(len(logs.records), 1)
            startup = logs.records[0].getMessage()
            self.assertIn('Attacks enabled: no', startup)
            self.assertIn('Fake clients (0/5): none', startup)
            self.assertIn('Clients with attacks performed: none (attacks disabled)', startup)
            self.assertIn('Clean clients (5/5): client_0, client_1, client_2, client_3, client_4', startup)
            self.assertFalse((Path(root) / 'attacks').exists())

    def test_performed_attack_list_matches_actual_poisoned_samples(self):
        for flip_prob, poison_prob in ((1.0, 1.0), (0.0, 1.0), (0.0, 0.0)):
            with self.subTest(flip_prob=flip_prob, poison_prob=poison_prob), tempfile.TemporaryDirectory() as root:
                config = config_for(root, clients=12, **{'attacks-num-malicious': 8,
                    'attacks-label-flip-prob': flip_prob, 'attacks-backdoor-poison-prob': poison_prob})
                logger = Mock(run_dir=Path(root))
                with self.assertLogs('flwr', level='INFO') as logs:
                    prepare_attack_datasets(logger, config)
                report = json.loads((Path(root) / 'attacks/assignment_and_poisoning.json').read_text())
                performed = logs.records[-1].getMessage().split('Attacks: requested')[0]
                lines = performed.strip().splitlines()
                self.assertEqual(lines[0], 'Clients with attacks performed:')
                expected = []
                for entry in report['clients']:
                    with np.load(Path(root) / 'attacks/training' / f"{entry['client_id']}.npz") as overlay:
                        count = len(overlay['positions'])
                    preview_dir = Path(root) / 'images/after_attack' / entry['client_id']
                    self.assertEqual(preview_dir.exists(), count > 0)
                    if count:
                        samples = json.loads((preview_dir / 'labels.json').read_text())
                        self.assertEqual(len(samples), min(5, count))
                        self.assertTrue(all(sample['poisoned'] for sample in samples))
                    if count:
                        expected.append(f"  {entry['client_id']} -> {entry['attack']} ({count} poisoned training samples)")
                self.assertEqual(lines[1:], expected or ['  none (no training samples were poisoned)'])

    def test_only_training_poisoned_fixed_overlays_and_metadata_for_iid_and_noniid(self):
        raw = self.images.copy()
        labels = self.labels.copy()
        for method in ('iid', 'dirichlet'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as root:
                config = config_for(root, **{'partition-method': method, 'attacks-save-indices': True})
                settings = PartitionSettings.from_config(config)
                holdouts = [cifar10.load_client_split(i, 5, 'validation', root, settings) for i in range(5)]
                with ExperimentLogger(root, 11, config, 'synthetic') as logger:
                    overlays = prepare_attack_datasets(logger, config)
                    report = json.loads((logger.run_dir / 'attacks/assignment_and_poisoning.json').read_text())
                    self.assertEqual(report['attack_counts'], {'label_flip': 2, 'backdoor': 1})
                    self.assertNotIn('client_0', report['assignments'])
                    self.assertEqual(report['effective_num_malicious'], 3)
                    plan = cifar10._partition_plan(root, 5, settings)
                    before_files = {p.name: p.read_bytes() for p in Path(overlays).iterdir()}
                    with patch('fl_defense.attacks.datasets.apply_attacks_for_client', side_effect=AssertionError('Must not resample')):
                        for i in range(5):
                            clean_x, clean_y = cifar10.load_client_split(i, 5, 'train', root, settings)
                            first = load_training_data(i, 5, root, settings, overlays)
                            second = load_training_data(i, 5, root, settings, overlays)
                            np.testing.assert_array_equal(first[0], second[0])
                            np.testing.assert_array_equal(first[1], second[1])
                            metadata = report['clients'][i]
                            previews = json.loads((logger.run_dir / 'images/attack_training' / f'client_{i}' / 'labels.json').read_text())
                            self.assertEqual([p['label_id'] for p in previews], first[1][:5].tolist())
                            self.assertEqual([p['original_label_id'] for p in previews], clean_y[:5].tolist())
                            self.assertEqual([p['source_index'] for p in previews], plan[i].train_indices[:5].tolist())
                            with np.load(Path(overlays) / f'client_{i}.npz') as saved:
                                positions = set(saved['positions'].tolist())
                            self.assertEqual([p['poisoned'] for p in previews], [j in positions for j in range(len(previews))])
                            after_dir = logger.run_dir / 'images/after_attack' / f'client_{i}'
                            if positions:
                                selected = np.array(sorted(positions)[:5])
                                after = json.loads((after_dir / 'labels.json').read_text())
                                self.assertEqual([p['source_index'] for p in after], plan[i].train_indices[selected].tolist())
                                self.assertEqual([p['label_id'] for p in after], first[1][selected].tolist())
                                self.assertEqual([p['original_label_id'] for p in after], clean_y[selected].tolist())
                                self.assertTrue(all(p['poisoned'] for p in after))
                                from PIL import Image
                                for record, position in zip(after, selected, strict=True):
                                    with Image.open(after_dir / record['file']) as preview_image:
                                        rendered = np.asarray(preview_image)
                                    np.testing.assert_array_equal(rendered[78:206:4, 48:176:4],
                                                                  np.rint(first[0][position] * 255).astype(np.uint8))
                            else:
                                self.assertFalse(after_dir.exists())
                            self.assertEqual(metadata['labels_changed'], np.count_nonzero(first[1] != clean_y))
                            self.assertEqual(sum(metadata['original_class_histogram'].values()), len(clean_y))
                            self.assertEqual(sum(metadata['attacked_class_histogram'].values()), len(clean_y))
                            if metadata['attack'] is None:
                                np.testing.assert_array_equal(first[0], clean_x)
                                np.testing.assert_array_equal(first[1], clean_y)
                            elif metadata['attack'] == 'label_flip':
                                np.testing.assert_array_equal(first[0], clean_x)
                            else:
                                self.assertEqual(np.count_nonzero(np.any(first[0] != clean_x, axis=(1, 2, 3))), metadata['trigger_stamped_samples'])
                                self.assertEqual(metadata['patch_dimensions'], [2, 2])
                            with np.load(logger.run_dir / 'attacks' / metadata['poisoned_indices_file']) as ids:
                                self.assertFalse(np.intersect1d(ids['source_indices'], plan[i].validation_indices).size)
                                np.testing.assert_array_equal(ids['source_indices'], plan[i].train_indices[ids['training_positions']])
                            actual_holdout = cifar10.load_client_split(i, 5, 'validation', root, settings)
                            for actual, expected in zip(actual_holdout, holdouts[i]):
                                np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(before_files, {p.name: p.read_bytes() for p in Path(overlays).iterdir()})
                    with self.assertRaisesRegex(ValueError, 'partition'):
                        load_training_data(0, 5, root, PartitionSettings(method=method, seed=43), overlays)
                np.testing.assert_array_equal(self.images, raw)
                np.testing.assert_array_equal(self.labels, labels)

    def test_disabled_zero_malicious_and_missing_overlay(self):
        with tempfile.TemporaryDirectory() as root:
            config = config_for(root, **{'attacks-enabled': False})
            with ExperimentLogger(root, 12, config, 'synthetic') as logger:
                self.assertIsNone(prepare_attack_datasets(logger, config))
                self.assertFalse((logger.run_dir / 'attacks').exists())
            config.update({'attacks-enabled': True, 'attacks-num-malicious': 0})
            with ExperimentLogger(root, 13, config, 'synthetic') as logger:
                overlays = prepare_attack_datasets(logger, config)
                report = json.loads((logger.run_dir / 'attacks/assignment_and_poisoning.json').read_text())
                self.assertEqual(report['effective_num_malicious'], 0)
                for i in range(5):
                    expected = cifar10.load_client_split(i, 5, 'train', root)
                    actual = load_training_data(i, 5, root, PartitionSettings(), overlays)
                    for a, b in zip(actual, expected):
                        np.testing.assert_array_equal(a, b)
                with self.assertRaises(FileNotFoundError):
                    load_training_data(0, 5, root, PartitionSettings(), str(Path(root) / 'missing'))

    def test_synthetic_real_client_training_two_rounds_and_no_defense_oracle(self):
        """Real tiny Keras fit/evaluate and Flower strategy, with synthetic CIFAR-shaped data."""
        with tempfile.TemporaryDirectory() as root:
            config = config_for(root, clients=3)
            grid = SimpleNamespace(get_node_ids=lambda: [100, 101, 102])
            with ExperimentLogger(root, 14, config, 'synthetic_wiring_model') as logger:
                overlays = prepare_attack_datasets(logger, config)
                initial = small_model()
                defense = ClusteringDefense(model_schema(initial))
                arrays = ArrayRecord(initial.get_weights())
                del initial
                strategy = LoggedFedAvg(logger, 3, clustering_defense=defense, poisoning_dir=overlays)
                with patch.object(client_app, 'load_model', side_effect=small_model), \
                     patch.object(defense, 'analyze', wraps=defense.analyze) as analyze:
                    for rnd in (1, 2):
                        replies = []
                        requests = strategy.configure_train(rnd, arrays, ConfigRecord(), grid)
                        for request in requests:
                            i = request.metadata.dst_node_id - 100
                            context = SimpleNamespace(run_config=config, node_config={'partition-id': i, 'num-partitions': 3})
                            reply = client_app.train(request, context)
                            self.assertEqual(set(reply.content), {'arrays', 'metrics', 'client-info'})
                            self.assertEqual(set(reply.content['client-info']), {'client-id', 'partition-id', 'role',
                                             'local-epochs', 'weight-schema', 'weight-convention', 'server-round'})
                            self.assertTrue(np.isfinite(reply.content['metrics']['train_loss']))
                            replies.append(reply)
                        arrays, _ = strategy.aggregate_train(rnd, replies)
                        # Defense only receives original model arrays and canonical schemas, never assignments/labels.
                        supplied = analyze.call_args.args[1]
                        self.assertEqual(set(supplied), {'client_0', 'client_1', 'client_2'})
                        for weights, schema in supplied.values():
                            self.assertTrue(all(isinstance(w, np.ndarray) for w in weights))
                            self.assertIsInstance(schema, str)
                            self.assertNotIn('attack', schema)
                        evaluations = []
                        for request in strategy.configure_evaluate(rnd, arrays, ConfigRecord(), grid):
                            i = request.metadata.dst_node_id - 100
                            context = SimpleNamespace(run_config=config, node_config={'partition-id': i, 'num-partitions': 3})
                            reply = client_app.evaluate(request, context)
                            self.assertEqual(set(reply.content['metrics']), {'eval_acc', 'eval_loss', 'num-examples'})
                            clean_x, clean_y = cifar10.load_client_split(i, 3, 'validation', root)
                            self.assertEqual(reply.content['backdoor-asr']['eligible'], np.count_nonzero(clean_y != 0))
                            evaluations.append(reply)
                        metrics = strategy.aggregate_evaluate(rnd, evaluations)
                        self.assertEqual(set(metrics), {'eval_acc', 'eval_loss'})
                        report = json.loads((logger.run_dir / 'attacks' / f'round_{rnd:03d}_asr.json').read_text())
                        self.assertEqual(report['eligible'], sum(m.content['backdoor-asr']['eligible'] for m in evaluations))
                        self.assertTrue(0 <= report['asr'] <= 1)
                reader = events(logger.run_dir)
                self.assertEqual(len(reader.Tensors('server/backdoor_asr')), 2)
                self.assertEqual(len(reader.Tensors('server/global_accuracy')), 2)
        keras.backend.clear_session(free_memory=True)

    def test_asr_aggregation_uses_eligible_counts_and_unavailable_is_not_zero_accuracy(self):
        with tempfile.TemporaryDirectory() as root, ExperimentLogger(root, 15, config_for(root, 2), 'synthetic') as logger:
            strategy = LoggedFedAvg(logger, 2)
            for rnd, counts in ((1, [(0, 3), (1, 1)]), (2, [(0, 0), (0, 0)])):
                replies = []
                for i, (successes, eligible) in enumerate(counts):
                    request = Message(RecordDict(), dst_node_id=100+i, message_type='evaluate')
                    replies.append(Message(RecordDict({
                        'metrics': MetricRecord({'eval_acc': .2 + i*.6, 'eval_loss': 1., 'num-examples': [10, 90][i]}),
                        'client-info': client_app.client_info(i),
                        'backdoor-asr': ConfigRecord({'successes': successes, 'eligible': eligible}),
                    }), reply_to=request))
                metrics = strategy.aggregate_evaluate(rnd, replies)
                self.assertAlmostEqual(metrics['eval_acc'], .74)
                report = json.loads((logger.run_dir / 'attacks' / f'round_{rnd:03d}_asr.json').read_text())
                if rnd == 1:
                    self.assertEqual(report['asr'], .25)
                else:
                    self.assertIsNone(report['asr'])
                    self.assertEqual(report['eligible'], 0)
            self.assertEqual(len(events(logger.run_dir).Tensors('server/backdoor_asr')), 1)
            self.assertEqual(float(value(events(logger.run_dir), 'server/backdoor_asr_eligible')), 0)


if __name__ == '__main__':
    unittest.main()
