"""Check cleanup and exact split equivalence for reused workers."""

import tempfile
import unittest
from unittest.mock import patch
import weakref

import keras
import numpy as np

from fl_defense.data import cifar10
from fl_defense.data.partitioning import PartitionSettings
from fl_defense.utils.client_resources import release_client_resources


class ClientResourceTests(unittest.TestCase):
    def test_model_history_cycles_released_after_training(self):
        refs = []

        @release_client_resources
        def request():
            model = keras.Sequential([keras.Input((2,)), keras.layers.Dense(1)])
            model.compile(optimizer="sgd", loss="mse", jit_compile=False)
            history = model.fit(np.ones((2, 2)), np.zeros((2, 1)), epochs=1, verbose=0)
            refs.extend((weakref.ref(model), weakref.ref(history)))
            return model.get_weights()

        for _ in range(3):
            weights = request()
            self.assertTrue(all(ref() is None for ref in refs))
            self.assertEqual(weights[0].shape, (2, 1))

    def test_cleanup_runs_on_failure(self):
        @release_client_resources
        def request():
            raise RuntimeError("training failed")
        with patch("fl_defense.utils.client_resources.keras.backend.clear_session") as clear:
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                request()
            clear.assert_called_once_with(free_memory=True)

    def test_selected_splits_match_old_loader(self):
        rng = np.random.default_rng(42)
        images = rng.integers(0, 256, (300, 32, 32, 3), dtype=np.uint8)
        labels = np.tile(np.arange(10), 30)
        for method in ("iid", "dirichlet"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as root:
                cifar10._partition_plan.cache_clear()
                settings = PartitionSettings(method=method)
                with patch.object(cifar10, "_load_local_training_data", return_value=(images, labels)):
                    for partition_id in range(15):
                        expected = cifar10.load_data(partition_id, 15, root, settings)
                        for split, offset in (("train", 0), ("validation", 2)):
                            actual = cifar10.load_client_split(partition_id, 15, split, root, settings)
                            np.testing.assert_array_equal(actual[0], expected[offset])
                            np.testing.assert_array_equal(actual[1], expected[offset + 1])
        cifar10._partition_plan.cache_clear()


if __name__ == "__main__":
    unittest.main()
