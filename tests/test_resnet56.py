"""Model architecture and federated weight/save compatibility checks."""

from pathlib import Path
import tempfile
import unittest

import keras
import numpy as np

from fl_defense.models.resnet56 import load_model


class ResNet56Tests(unittest.TestCase):
    def tearDown(self):
        keras.backend.clear_session()

    def test_architecture(self):
        model = load_model(learning_rate=0.005)
        self.assertEqual(model.input_shape, (None, 32, 32, 3))
        self.assertEqual(model.output_shape, (None, 10))
        self.assertEqual(sum(isinstance(layer, keras.layers.Add) for layer in model.layers), 27)
        self.assertEqual(sum(isinstance(layer, keras.layers.Conv2D) for layer in model.layers), 57)
        for stage, size, channels in ((1, 32, 16), (2, 16, 32), (3, 8, 64)):
            self.assertEqual(
                tuple(model.get_layer(f"stage{stage}_block9_out").output.shape[1:]),
                (size, size, channels),
            )
        self.assertAlmostEqual(float(keras.ops.convert_to_numpy(model.optimizer.learning_rate)), 0.005)

    def test_training_weight_transfer_and_serialization(self):
        model = load_model()
        images = np.random.default_rng(42).random((2, 32, 32, 3), dtype=np.float32)
        labels = np.array([0, 9], dtype=np.int32)
        metrics = model.train_on_batch(images, labels)
        self.assertTrue(np.isfinite(metrics).all())
        expected = keras.ops.convert_to_numpy(model(images, training=False))
        np.testing.assert_allclose(expected.sum(axis=1), 1.0, atol=1e-6)

        replica = load_model()
        replica.set_weights(model.get_weights())
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(replica(images, training=False)), expected, atol=1e-6
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resnet56.keras"
            model.save(path)
            restored = keras.models.load_model(path)
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(restored(images, training=False)), expected, atol=1e-6
            )


if __name__ == "__main__":
    unittest.main()
