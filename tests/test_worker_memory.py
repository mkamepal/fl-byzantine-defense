"""Worker RSS diagnostics preserve replies, cleanup, and request identities."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, mock_open, patch
import weakref

import keras
import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Message, RecordDict

from fl_defense import client_app
from fl_defense.aggregation.logged_fedavg import LoggedFedAvg
from fl_defense.utils.client_resources import release_client_resources
from fl_defense.utils.worker_memory import current_rss_bytes, record_client_memory, _active_trace


def records(root):
    return [json.loads(line) for path in Path(root).glob('worker_*.jsonl')
            for line in path.read_text().splitlines()]


class WorkerMemoryTests(unittest.TestCase):
    def setUp(self):
        try:
            from flwr.supercore.task_identity import TaskIdentity
        except ImportError:
            return
        for name, setting in (('_run_id', 99), ('_task_id', 1), ('_node_id', 0)):
            fixture = patch.object(TaskIdentity, name, setting)
            fixture.start()
            self.addCleanup(fixture.stop)

    def test_rss_reads_resident_pages_not_virtual_size_or_peak(self):
        with patch('builtins.open', mock_open(read_data='9999 12 5 1 0 7 0\n')), \
                patch('os.sysconf', return_value=4096):
            self.assertEqual(current_rss_bytes(), 12 * 4096)

    def test_samples_are_immediate_and_request_local(self):
        with tempfile.TemporaryDirectory() as root:
            message = SimpleNamespace(content={'config': {'server-round': 7, 'worker-memory-dir': root}})
            context = SimpleNamespace(node_config={'partition-id': 12})
            reply = object()

            @release_client_resources
            def train(msg, context):
                record_client_memory('after_train')
                self.assertEqual(len(records(root)) % 4, 2)
                return reply

            with patch('fl_defense.utils.worker_memory.current_rss_bytes',
                       side_effect=[100, 300, 320, 180, 200, 400, 420, 210]), \
                    patch('fl_defense.utils.client_resources.keras.backend.clear_session') as clear, \
                    self.assertLogs('flwr', level='INFO'):
                self.assertIs(train(message, context), reply)
                context.node_config['partition-id'] = 2
                message.content['config']['server-round'] = 8
                self.assertIs(train(msg=message, context=context), reply)
                self.assertEqual(clear.call_count, 2)
            rows = records(root)
            self.assertEqual([r['stage'] for r in rows[:4]],
                             ['request_start', 'after_train', 'before_cleanup', 'after_cleanup'])
            self.assertEqual(rows[3]['delta_previous_bytes'], -140)
            self.assertEqual(rows[3]['delta_request_bytes'], 80)
            self.assertIsNone(rows[4]['delta_previous_bytes'])
            self.assertEqual(rows[7]['delta_request_bytes'], 10)
            self.assertEqual((rows[0]['client_id'], rows[0]['round']), ('client_12', 7))
            self.assertEqual((rows[4]['client_id'], rows[4]['round']), ('client_2', 8))
            self.assertNotEqual(rows[0]['request_id'], rows[4]['request_id'])
            self.assertEqual(rows[3]['outcome'], 'success')
            self.assertIsNone(_active_trace.get())

    def test_failed_handler_still_records_cleanup_and_preserves_error(self):
        with tempfile.TemporaryDirectory() as root:
            msg = SimpleNamespace(content={'config': {'worker-memory-dir': root}})

            @release_client_resources
            def train(msg):
                raise RuntimeError('original training error')

            with patch('fl_defense.utils.client_resources.keras.backend.clear_session') as clear:
                with self.assertRaisesRegex(RuntimeError, 'original training error'):
                    train(msg)
                clear.assert_called_once_with(free_memory=True)
            self.assertEqual(records(root)[-1]['stage'], 'after_cleanup')
            self.assertEqual(records(root)[-1]['outcome'], 'error')
            self.assertIsNone(_active_trace.get())

    def test_unavailable_rss_and_unwritable_output_do_not_abort(self):
        with tempfile.TemporaryDirectory() as root:
            blocked = Path(root) / 'file'
            blocked.write_text('existing file')
            msg = SimpleNamespace(content={'config': {'worker-memory-dir': str(blocked)}})

            @release_client_resources
            def evaluate(msg):
                return 123

            with patch('fl_defense.utils.worker_memory.current_rss_bytes', side_effect=OSError('unsupported')), \
                    patch('fl_defense.utils.client_resources.keras.backend.clear_session'), \
                    self.assertLogs('flwr', level='INFO') as logs:
                self.assertEqual(evaluate(msg), 123)
            self.assertIn('rss_mib=unavailable', '\n'.join(logs.output))
            self.assertIn('Cannot save', '\n'.join(logs.output))
            self.assertEqual(blocked.read_text(), 'existing file')

    def test_missing_start_sample_is_not_replaced_with_a_later_baseline(self):
        with tempfile.TemporaryDirectory() as root:
            msg = SimpleNamespace(content={'config': {'worker-memory-dir': root}})

            @release_client_resources
            def evaluate(msg):
                return 123

            with patch('fl_defense.utils.worker_memory.current_rss_bytes',
                       side_effect=[OSError('temporarily unavailable'), 300, 200]), \
                    patch('fl_defense.utils.client_resources.keras.backend.clear_session'):
                self.assertEqual(evaluate(msg), 123)
            rows = records(root)
            self.assertIsNone(rows[0]['rss_bytes'])
            self.assertTrue(all(r['delta_request_bytes'] is None for r in rows))
            self.assertEqual(rows[-1]['delta_previous_bytes'], -100)

    def test_strategy_routes_both_phases_to_this_run_without_mutating_config(self):
        with tempfile.TemporaryDirectory() as root:
            logger = Mock(run_dir=Path(root))
            strategy = LoggedFedAvg(logger, 11, fraction_train=1, fraction_evaluate=1)
            grid = SimpleNamespace(get_node_ids=lambda: [102, 110])
            config = ConfigRecord({'custom': 42})
            arrays = ArrayRecord([np.zeros(1, dtype=np.float32)])
            for configure in (strategy.configure_train, strategy.configure_evaluate):
                requests = configure(4, arrays, config, grid)
                self.assertEqual(len(requests), 2)
                for request in requests:
                    sent = request.content['config']
                    self.assertEqual(sent['worker-memory-dir'], str(Path(root).resolve() / 'worker_memory'))
                    self.assertEqual(sent['server-round'], 4)
                    self.assertEqual(sent['custom'], 42)
            self.assertEqual(dict(config), {'custom': 42})

    def test_real_client_handlers_emit_stages_and_keep_reply_schema(self):
        def tiny_model(learning_rate=.01):
            model = keras.Sequential([keras.Input((32, 32, 3)),
                                      keras.layers.GlobalAveragePooling2D(),
                                      keras.layers.Dense(10, activation='softmax')])
            model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'], jit_compile=False)
            return model

        with tempfile.TemporaryDirectory() as root:
            context = SimpleNamespace(node_config={'partition-id': 10, 'num-partitions': 11},
                run_config={'num-clients': 11, 'local-epochs': 1, 'batch-size': 2,
                            'learning-rate': .01, 'verbose': False, 'attacks-enabled': False})
            model = tiny_model()
            arrays = ArrayRecord(model.get_weights())
            del model
            data = (np.ones((2, 32, 32, 3), dtype=np.float32), np.array([1, 2]))
            with patch.object(client_app, 'load_model', side_effect=tiny_model), \
                    patch.object(client_app, 'load_training_data', return_value=data), \
                    patch.object(client_app, 'load_client_split', return_value=data):
                for phase in ('train', 'evaluate'):
                    context.run_config['attacks-enabled'] = phase == 'evaluate'
                    request = Message(RecordDict({'arrays': arrays, 'config': ConfigRecord({
                        'server-round': 3, 'worker-memory-dir': root})}), dst_node_id=110, message_type=phase)
                    reply = getattr(client_app, phase)(request, context)
                    self.assertEqual(set(reply.content), {'arrays', 'metrics', 'client-info'} if phase == 'train'
                                     else {'metrics', 'client-info', 'backdoor-asr'})
                    rows = [r for r in records(root) if r['phase'] == phase]
                    expected = ['request_start', 'after_initial_cleanup', 'after_data_load',
                                'after_model_load', f'before_{phase}', f'after_{phase}']
                    if phase == 'evaluate':
                        expected += ['before_asr', 'after_asr']
                    expected += ['before_cleanup', 'after_cleanup']
                    self.assertEqual([r['stage'] for r in rows], expected)
                    self.assertTrue(all(r['client_id'] == 'client_10' and r['round'] == 3 for r in rows))
                    self.assertTrue(all(r['rss_bytes'] > 0 for r in rows))

    def test_repeated_training_does_not_accumulate_custom_gradient_graphs(self):
        # Regression for TF 2.20's registry retaining discarded client graphs.
        # Check live registrations rather than noisy allocator/RSS thresholds.
        from tensorflow.python.framework import ops

        refs = []
        def tiny_model(learning_rate=.001):
            model = keras.Sequential([keras.Input((32, 32, 3)),
                                      keras.layers.GlobalAveragePooling2D(),
                                      keras.layers.Dense(10, activation='softmax')])
            model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                          loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            refs.append(weakref.ref(model))
            return model

        initial = tiny_model()
        arrays = ArrayRecord(initial.get_weights())
        del initial
        keras.backend.clear_session(free_memory=True)
        registry = ops._gradient_registry._registry
        original_keys = set(registry)
        context = SimpleNamespace(node_config={'num-partitions': 11}, run_config={
            'num-clients': 11, 'local-epochs': 2, 'batch-size': 2,
            'learning-rate': .001, 'verbose': False, 'attacks-enabled': False})
        data = (np.ones((2, 32, 32, 3), dtype=np.float32), np.array([1, 2]))
        with patch.object(client_app, 'load_model', side_effect=tiny_model), \
                patch.object(client_app, 'load_training_data', return_value=data):
            for rnd, partition in enumerate((0, 2, 10), 1):
                context.node_config['partition-id'] = partition
                request = Message(RecordDict({'arrays': arrays, 'config': ConfigRecord({
                    'server-round': rnd})}), dst_node_id=100 + partition, message_type='train')
                reply = client_app.train(request, context)
                self.assertEqual(reply.content['client-info']['client-id'], f'client_{partition}')
                self.assertEqual(reply.content['client-info']['local-epochs'], 2)
                self.assertTrue(np.isfinite(reply.content['metrics']['train_loss']))
                self.assertTrue(all(np.all(np.isfinite(w))
                                    for w in reply.content['arrays'].to_numpy_ndarrays()))
                self.assertFalse([key for key in set(registry) - original_keys
                                  if key.startswith('CustomGradient-')])
                self.assertTrue(all(ref() is None for ref in refs))


if __name__ == '__main__':
    unittest.main()
