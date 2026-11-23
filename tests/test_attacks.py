"""Exact data-poisoning semantics; synthetic fixtures are not effectiveness results."""

from dataclasses import replace
import unittest
import numpy as np

from fl_defense.attacks.backdoor import backdoor_attack
from fl_defense.attacks.label_flipping import label_flip_attack
from fl_defense.attacks.config import AttackConfig, assign_attacks
from fl_defense.attacks.dispatch import apply_attacks_for_client
from fl_defense.attacks.evaluation import backdoor_success_counts


class LabelFlipTests(unittest.TestCase):
    def test_zero_one_binary_noncontiguous_and_no_mutation(self):
        labels = np.array([0, 1, 0, 1], np.int64)
        np.testing.assert_array_equal(label_flip_attack(labels, 0), labels)
        np.testing.assert_array_equal(label_flip_attack(labels, 1), 1 - labels)
        np.testing.assert_array_equal(labels, [0, 1, 0, 1])
        labels = np.tile([2, 5, 9], 100).astype(np.int32)
        result = label_flip_attack(labels, 1)
        self.assertTrue(np.all(result != labels))
        self.assertTrue(set(result) <= {2, 5, 9})
        self.assertEqual(result.dtype, labels.dtype)
        np.testing.assert_array_equal(labels, np.tile([2, 5, 9], 100))

    def test_reference_rng_mask_reproducibility_and_shape(self):
        labels = np.tile([2, 5, 9], 30).astype(np.int32)
        rng = np.random.RandomState(42)
        mask = rng.rand(len(labels)) < .4
        offsets = rng.randint(1, 3, size=len(labels))
        expected = labels.copy()
        classes = np.unique(labels)
        expected[mask] = classes[(np.searchsorted(classes, labels) + offsets) % 3][mask]
        result, actual_mask = label_flip_attack(labels, .4, return_mask=True)
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(actual_mask, mask)
        np.testing.assert_array_equal(label_flip_attack(labels, .4), result)
        np.testing.assert_array_equal(label_flip_attack(labels[:, None], .4), result[:, None])
        np.testing.assert_array_equal(result[~mask], labels[~mask])
        self.assertTrue(np.all(result[mask] != labels[mask]))

    def test_empty_single_class_and_invalid_parameters(self):
        self.assertEqual(label_flip_attack([]).dtype, np.int32)
        for labels in ([], [5], [5, 5]):
            np.testing.assert_array_equal(label_flip_attack(labels, 1), labels)
        for value in (-.1, 1.1, float('nan'), float('inf'), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                label_flip_attack([0, 1], value)
        with self.assertRaises(ValueError):
            label_flip_attack([.1, 1.9])
        with self.assertRaises(ValueError):
            label_flip_attack([0, 1], seed=-1)


class BackdoorTests(unittest.TestCase):
    def test_zero_and_one_patch_geometry_layout_channels(self):
        for size, patch in ((28, 2), (32, 2), (64, 5)):
            for layout in ('HWC', 'CHW', 'HW'):
                shape = {'HWC': (4, size, size, 3), 'CHW': (4, 3, size, size), 'HW': (4, size, size)}[layout]
                images = np.full(shape, .25, np.float32)
                labels = np.array([0, 1, 2, 0], np.int64)
                with self.subTest(size=size, layout=layout):
                    clean_x, clean_y = backdoor_attack(images, labels, poison_prob=0, layout=layout)
                    np.testing.assert_array_equal(clean_x, images)
                    np.testing.assert_array_equal(clean_y, labels)
                    result, poisoned, mask = backdoor_attack(images, labels, poison_prob=1, layout=layout, return_mask=True)
                    expected = images.copy()
                    if layout == 'HWC':
                        expected[:, -patch:, -patch:, :] = 1
                    elif layout == 'CHW':
                        expected[:, :, -patch:, -patch:] = 1
                    else:
                        expected[:, -patch:, -patch:] = 1
                    np.testing.assert_array_equal(result, expected)
                    np.testing.assert_array_equal(poisoned, np.zeros_like(labels))
                    self.assertTrue(mask.all())  # Original target-class examples are stamped too.
                    np.testing.assert_array_equal(images, np.full(shape, .25, np.float32))
                    np.testing.assert_array_equal(labels, [0, 1, 2, 0])

    def test_mask_reproducibility_and_uint8_white(self):
        images = np.zeros((40, 32, 32, 3), np.uint8)
        labels = np.arange(40) % 10
        first = backdoor_attack(images, labels, return_mask=True)
        second = backdoor_attack(images, labels, return_mask=True)
        for actual, expected in zip(first, second):
            np.testing.assert_array_equal(actual, expected)
        mask = np.random.RandomState(42).rand(40) < .4
        np.testing.assert_array_equal(first[2], mask)
        self.assertEqual(first[0].dtype, np.uint8)
        np.testing.assert_array_equal(first[0][~mask], images[~mask])
        self.assertTrue((first[0][mask, -2:, -2:, :] == 255).all())
        self.assertEqual(np.count_nonzero(first[0]), int(mask.sum()) * 2 * 2 * 3)

    def test_invalid_inputs(self):
        images = np.zeros((3, 32, 32, 3), np.float32)
        for kwargs in ({'poison_prob': -1}, {'trigger_value': 2}, {'target_label': 11},
                       {'layout': 'CHW', 'poison_prob': float('nan')}, {'seed': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                backdoor_attack(images, [0, 1, 2], **kwargs)
        for source in (np.zeros((3, 1, 1, 3), np.float32), images + 2, images.astype(np.int64)):
            with self.assertRaises(ValueError):
                backdoor_attack(source, [0, 1, 2])
        with self.assertRaises(ValueError):
            backdoor_attack(images, [0, 1])
        with self.assertRaises(ValueError):
            backdoor_attack(images, [0, 1, 99])

    def test_global_target_absent_locally_and_single_channel(self):
        images = np.zeros((3, 32, 32, 1), np.float64)
        result, labels = backdoor_attack(images, np.array([2, 5, 9], np.int32), poison_prob=1)
        np.testing.assert_array_equal(labels, [0, 0, 0])
        self.assertEqual(result.dtype, np.float64)
        self.assertTrue((result[:, -2:, -2:, :] == 1).all())
        self.assertEqual(np.count_nonzero(result), 3 * 2 * 2)

    def test_local_rng_does_not_change_global_state(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(17)
        state = np.random.get_state()
        label_flip_attack([0, 1, 2], .4)
        backdoor_attack(np.zeros((3, 32, 32, 3), np.float32), [0, 1, 2])
        assign_attacks(15, AttackConfig(enabled=True))
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])


class AssignmentDispatchTests(unittest.TestCase):
    def test_distribution_rounding_reproducibility_and_protected_heads(self):
        config = AttackConfig(enabled=True)
        assignments, report = assign_attacks(15, config)
        self.assertEqual(report['requested_num_malicious'], 3)
        self.assertEqual(report['effective_num_malicious'], 3)
        self.assertEqual(report['attack_counts'], {'label_flip': 2, 'backdoor': 1})
        self.assertNotIn('client_0', assignments)
        self.assertEqual(assignments, assign_attacks(15, config)[0])
        assignments, report = assign_attacks(5, replace(config, num_malicious=100, head_client='client_2'), 'client_1')
        self.assertEqual(report['effective_num_malicious'], 2)
        self.assertEqual(set(assignments), {'client_3', 'client_4'})
        _, tied = assign_attacks(5, replace(config, num_malicious=1, distribution={'label_flip': 1, 'backdoor': 1}))
        self.assertEqual(tied['attack_counts'], {'backdoor': 1, 'label_flip': 0})

    def test_caps_zero_disabled_uniform_and_single_type(self):
        config = AttackConfig(enabled=True, num_malicious=99)
        for count in (1, 2, 3, 5, 15):
            mapping, report = assign_attacks(count, config)
            self.assertEqual(len(mapping), min(count-1, int(count * .7)))
            self.assertEqual(len(mapping), len(set(mapping)))
        for config in (AttackConfig(), AttackConfig(enabled=True, num_malicious=0), AttackConfig(enabled=True, max_fake_ratio=0)):
            self.assertEqual(assign_attacks(15, config)[0], {})
        for name in ('label_flip', 'backdoor'):
            cfg = AttackConfig(enabled=True, enabled_types=(name,))
            self.assertEqual(set(assign_attacks(15, cfg)[0].values()), {name})
        cfg = AttackConfig(enabled=True, distribution=None)
        self.assertEqual(assign_attacks(15, cfg), assign_attacks(15, cfg))
        self.assertEqual(AttackConfig.from_config({'partition-seed': 19}).seed, 19)
        self.assertIsNone(AttackConfig.from_config({'attacks-distribution': ''}).distribution)

    def test_invalid_configs(self):
        for kwargs in ({'num_malicious': -1}, {'num_malicious': True}, {'max_fake_ratio': 2},
                       {'exclude_head': False}, {'seed': 2**32}, {'enabled_types': ('unknown',)},
                       {'distribution': {'label_flip': -1}}, {'distribution': {'label_flip': float('nan')}},
                       {'distribution': {}}, {'distribution': {'unknown': 1}}, {'target_label': 10}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AttackConfig(**kwargs)
        with self.assertRaises(ValueError):
            AttackConfig.from_config({'attacks-enabled-types': 'not json'})
        with self.assertRaises(ValueError):
            assign_attacks(3, AttackConfig(enabled=True, head_client='client_5'))

    def test_dispatch_counts_benign_head_and_unknown(self):
        images = np.zeros((4, 32, 32, 3), np.float32)
        labels = np.array([0, 1, 0, 1], np.int64)
        config = AttackConfig(enabled=True, poison_prob=1, flip_prob=1)
        clean = apply_attacks_for_client(images, labels, 'client_0', {}, config, 42)
        self.assertIs(clean.images, images)
        self.assertIs(clean.labels, labels)
        flip = apply_attacks_for_client(images, labels, 'client_1', {'client_1': 'label_flip'}, config, 42)
        self.assertIs(flip.images, images)
        self.assertEqual(flip.metadata['labels_changed'], 4)
        poison = apply_attacks_for_client(images, labels, 'client_1', {'client_1': 'backdoor'}, config, 42)
        self.assertEqual(poison.metadata['trigger_stamped_samples'], 4)
        self.assertEqual(poison.metadata['labels_changed'], 2)
        self.assertEqual(sum(poison.metadata['attacked_class_histogram'].values()), 4)
        for mapping in ({'client_0': 'backdoor'}, {'client_0': 'unknown'}):
            with self.assertRaises(ValueError):
                apply_attacks_for_client(images, labels, 'client_0', mapping, config, 42)
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            apply_attacks_for_client(images, labels, 'client_1', {'client_1': 'unknown'}, config, 42)

    def test_asr_excludes_target_triggers_all_and_empty(self):
        images = np.zeros((5, 32, 32, 3), np.float32)
        labels = np.array([0, 1, 0, 2, 3])
        seen = []
        def model(x, training=False):
            self.assertFalse(training)
            self.assertTrue((x[:, -2:, -2:, :] == 1).all())
            seen.append(len(x))
            return np.tile(np.eye(10)[0], (len(x), 1))
        self.assertEqual(backdoor_success_counts(model, images, labels, batch_size=2), {'successes': 3, 'eligible': 3})
        self.assertEqual(seen, [2, 1])
        self.assertEqual(backdoor_success_counts(model, images, np.zeros(5, int)), {'successes': 0, 'eligible': 0})
        np.testing.assert_array_equal(images, np.zeros_like(images))
        self.assertEqual(seen, [2, 1])


if __name__ == '__main__':
    unittest.main()
