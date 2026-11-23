"""Exact checkpoint round trips and update measurements."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fl_defense.utils.weight_archive import WeightArchive, load_model_snapshot, load_weights, save_json, save_weights, weight_shift


class WeightArchiveTests(unittest.TestCase):
    def test_restores_model_without_source_factory(self):
        import keras
        from flwr.app import ArrayRecord
        from fl_defense.models.resnet56 import load_model
        model = load_model()
        images = np.random.default_rng(12).random((1, 32, 32, 3), dtype=np.float32)
        expected = model(images, training=False).numpy()
        with tempfile.TemporaryDirectory() as directory:
            archive = WeightArchive(directory, 15, model)
            archive.begin_round(1, ArrayRecord(model.get_weights()), [])
            path = Path(directory) / "models" / "round_001_global_before.npz"
            restored = load_model_snapshot(path)
            for left, right in zip(model.get_weights(), restored.get_weights()):
                np.testing.assert_array_equal(left, right)
            np.testing.assert_allclose(restored(images, training=False).numpy(), expected, atol=1e-6)
            schema = json.loads((path.parent / "weight_schema.json").read_text())
            self.assertEqual(len(schema["tensors"]), len(model.get_weights()))
        keras.backend.clear_session()

    def test_exact_numeric_round_trip_and_no_overwrite(self):
        weights = [np.arange(12, dtype=np.float32).reshape(3, 4), np.array([1, 2], dtype=np.int64)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "client.npz"
            save_weights(path, weights)
            original = path.read_bytes()
            restored = load_weights(path)
            for expected, actual in zip(weights, restored):
                self.assertEqual(expected.dtype, actual.dtype)
                np.testing.assert_array_equal(expected, actual)
            with self.assertRaises(FileExistsError):
                save_weights(path, [np.zeros(1)])
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_known_shift_and_trainable_only_measurement(self):
        reference = [np.array([3., 4.]), np.array([0.])]
        weights = [np.array([6., 8.]), np.array([2.])]
        result = weight_shift(weights, reference, {0})
        self.assertAlmostEqual(result["l2"], np.sqrt(29))
        self.assertEqual(result["trainable_l2"], 5)
        self.assertEqual(result["max_abs"], 4)
        self.assertAlmostEqual(result["relative_l2"], np.sqrt(29) / 5)
        self.assertEqual(result["per_tensor"][1]["l2"], 2)
        self.assertEqual(weight_shift(weights, weights)["l2"], 0)
        with self.assertRaises(ValueError):
            weight_shift(weights, reference[:1])
        with self.assertRaises(ValueError):
            weight_shift([np.zeros((2,))], [np.zeros((3,))])

    def test_zero_reference_and_nonfinite_weights(self):
        self.assertIsNone(weight_shift([np.ones(2)], [np.zeros(2)])["relative_l2"])
        bad = [np.array([np.nan, np.inf], dtype=np.float32)]
        result = weight_shift(bad, [np.zeros(2)])
        self.assertEqual(result["nonfinite_elements"], 2)
        self.assertIsNone(result["l2"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            save_weights(path, bad)
            np.testing.assert_array_equal(load_weights(path)[0], bad[0])
            report = Path(directory) / "report.json"
            save_json(report, result)
            self.assertIsNone(json.loads(report.read_text())["l2"])


if __name__ == "__main__":
    unittest.main()
