"""Clean shared-holdout inference of complete client models."""

import numpy as np
from decimal import Decimal

from .defense_config import finite_range
from .parameters import client_number, model_schema, schema_json, validate_weights


class TrustedEvaluator:
    """Own an isolated Keras model, with all weights/buffers replaced per client.

    Uses the application's sparse categorical cross-entropy plus model
    regularization losses, like Model.evaluate. Stateless example sums avoid
    batch-size averaging errors and metric leakage between clients.
    """

    def __init__(self, source_model, x, y, batch_size, source, dataset_identifier="CIFAR-10"):
        import keras
        import tensorflow as tf

        self.x, self.y = np.asarray(x), np.asarray(y).reshape(-1)
        if len(self.x) != len(self.y) or not len(self.y):
            raise ValueError("Trusted validation dataset is empty or misaligned; accuracy unavailable")
        if not np.isfinite(self.x).all() or not np.isfinite(self.y).all():
            raise ValueError("Non-finite trusted validation data")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Trusted validation batch size must be positive")
        if source_model.loss != "sparse_categorical_crossentropy":
            raise ValueError("Trusted evaluation requires the application's sparse categorical cross-entropy")
        self.batch_size, self.source = batch_size, source
        self.dataset_identifier = dataset_identifier
        self.schema = model_schema(source_model)
        with tf.device("/CPU:0"):
            self.model = keras.models.clone_model(source_model)
        if schema_json(model_schema(self.model)) != schema_json(self.schema):
            raise ValueError("Trusted evaluator model schema differs from server")

    def evaluate(self, clients, head_client):
        import keras
        import tensorflow as tf

        if head_client not in clients:
            raise ValueError(f"Trusted reference {head_client} is absent")
        results = {}
        for cid in sorted(clients, key=client_number):
            weights, received_schema = clients[cid]
            if received_schema != schema_json(self.schema):
                raise ValueError(f"Parameter names/order/schema mismatch for {cid}")
            validate_weights(weights, self.schema)
            with tf.device("/CPU:0"):
                self.model.set_weights(weights)  # includes BN moving statistics
                self.model.reset_metrics()
                loss_sum, correct, count = 0.0, 0, 0
                for start in range(0, len(self.y), self.batch_size):
                    labels = self.y[start:start + self.batch_size]
                    prediction = self.model(self.x[start:start + self.batch_size], training=False)
                    losses = np.asarray(keras.losses.sparse_categorical_crossentropy(labels, prediction),
                                        dtype=np.float64)
                    if not np.isfinite(prediction.numpy()).all() or not np.isfinite(losses).all():
                        raise ValueError(f"Non-finite trusted validation output for {cid}")
                    regularization = sum(float(v) for v in self.model.losses)
                    loss_sum += float(losses.sum()) + len(labels) * regularization
                    correct += int(np.count_nonzero(np.argmax(prediction.numpy(), axis=-1) == labels))
                    count += len(labels)
            results[cid] = {"loss": loss_sum / count, "accuracy": correct / count, "example_count": count}
        return results


def score_validation(measurements, head_client="client_0", accuracy_tolerance=0.06):
    finite_range("validation accuracy tolerance", accuracy_tolerance, 0, 1)
    if head_client not in measurements:
        raise ValueError(f"Trusted reference {head_client} is absent")
    counts = set()
    for cid, value in measurements.items():
        client_number(cid)
        finite_range("validation accuracy", value["accuracy"], 0, 1)
        finite_range("validation loss", value["loss"], 0, float("inf"))
        if type(value["example_count"]) is not int or value["example_count"] < 1:
            raise ValueError("Trusted validation is unavailable: empty dataset")
        counts.add(value["example_count"])
    if len(counts) != 1:
        raise ValueError("Clients must use the same complete trusted validation dataset")
    head_accuracy = measurements[head_client]["accuracy"]
    # Subtract decimal accuracies/tolerances without moving an inclusive boundary
    # up one binary ULP (for example, 0.90 - 0.06 must accept exactly 0.84).
    minimum = float(Decimal(str(head_accuracy)) - Decimal(str(accuracy_tolerance)))
    return {cid: {**value, "head_accuracy": head_accuracy, "accuracy_tolerance": accuracy_tolerance,
                  "minimum_accuracy": minimum, "score": int(value["accuracy"] >= minimum)}
            for cid, value in measurements.items()}
