"""Line-based training output suitable for streamed simulation worker logs."""

import logging

import keras
from fl_defense.data.client_identity import client_id, client_role


class ClientTrainingLogger(keras.callbacks.Callback):
    """Emit complete log records instead of terminal progress-bar updates."""

    def __init__(self, partition_id, server_round, epochs):
        super().__init__()
        self.prefix = f"[round={server_round} client={client_id(partition_id)} role={client_role(partition_id)}]"
        self.epochs = epochs
        self.logger = logging.getLogger("flwr")

    def on_train_begin(self, logs=None):
        self.logger.info("%s Training started: %s epochs", self.prefix, self.epochs)

    def on_epoch_end(self, epoch, logs=None):
        metrics = " ".join(
            f"{name}={float(value):.4f}" for name, value in (logs or {}).items()
        )
        self.logger.info(
            "%s Epoch %s/%s %s", self.prefix, epoch + 1, self.epochs, metrics
        )

    def on_train_end(self, logs=None):
        self.logger.info("%s Training finished", self.prefix)
