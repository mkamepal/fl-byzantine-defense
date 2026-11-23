"""Check that actual Keras training emits identifiable complete epoch lines."""

import unittest

import keras
import numpy as np

from fl_defense.utils.training_logs import ClientTrainingLogger


class TrainingLogTests(unittest.TestCase):
    def test_fit_logs_round_client_and_each_epoch(self):
        model = keras.Sequential([keras.Input((2,)), keras.layers.Dense(1)])
        model.compile(optimizer="sgd", loss="mse", metrics=["mae"])
        callback = ClientTrainingLogger(partition_id=3, server_round=7, epochs=2)
        with self.assertLogs("flwr", level="INFO") as captured:
            model.fit(
                np.ones((2, 2), dtype=np.float32),
                np.zeros((2, 1), dtype=np.float32),
                epochs=2,
                batch_size=2,
                verbose=0,
                callbacks=[callback],
            )
        lines = captured.output
        self.assertEqual(len(lines), 4)
        self.assertTrue(all("[round=7 client=client_3 role=client]" in line for line in lines))
        self.assertIn("Training started", lines[0])
        self.assertIn("Epoch 1/2", lines[1])
        self.assertIn("Epoch 2/2", lines[2])
        self.assertIn("loss=", lines[1])
        self.assertIn("mae=", lines[1])
        self.assertIn("Training finished", lines[3])
        self.assertTrue(all("\r" not in line and "\n" not in line for line in lines))
        keras.backend.clear_session()


if __name__ == "__main__":
    unittest.main()
