"""Persist exact client weights and compare updates without changing FedAvg."""

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from fl_defense.data.client_identity import client_id, client_role


def load_weights(path):
    """Load numeric arrays in Keras get_weights() order, without pickle."""
    with np.load(path, allow_pickle=False) as archive:
        return [archive[f"arr_{i}"] for i in range(len(archive.files))]


def load_model_snapshot(path):
    """Restore an uncompiled model from the saved architecture and exact weights.

    Optimizer state is not sent by Flower clients, so this restores model state
    for inspection/inference, not a client's optimizer trajectory.
    """
    import keras
    path = Path(path)
    model = keras.models.model_from_json((path.parent / "model_architecture.json").read_text())
    model.set_weights(load_weights(path))
    return model


def weight_shift(weights, reference, trainable_indices=None):
    """L2 and maximum absolute differences, including per-tensor measurements."""
    if len(weights) != len(reference):
        raise ValueError("Weight tensor counts differ")
    squared, base_squared, maximum, invalid = 0.0, 0.0, 0.0, 0
    trainable_squared = 0.0
    per_tensor = []
    for i, (value, base) in enumerate(zip(weights, reference)):
        if value.shape != base.shape:
            raise ValueError(f"Weight shape differs at tensor {i}")
        value, base = np.asarray(value, dtype=np.float64), np.asarray(base, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            delta = value - base
            bad = int(np.count_nonzero(~np.isfinite(delta)))
            norm_squared = float(np.sum(delta * delta))
            local_max = float(np.max(np.abs(delta))) if delta.size else 0.0
            base_squared += float(np.sum(base * base))
        invalid += bad
        squared += norm_squared
        maximum = max(maximum, local_max) if math.isfinite(local_max) else float("nan")
        if trainable_indices is not None and i in trainable_indices:
            trainable_squared += norm_squared
        per_tensor.append({"index": i, "l2": _finite(math.sqrt(norm_squared)),
                           "max_abs": _finite(local_max), "nonfinite_elements": bad})
    l2 = math.sqrt(squared)
    reference_l2 = math.sqrt(base_squared)
    return {
        "l2": _finite(l2), "max_abs": _finite(maximum),
        "reference_l2": _finite(reference_l2),
        "relative_l2": _finite(l2 / reference_l2) if reference_l2 > 0 else None,
        "trainable_l2": _finite(math.sqrt(trainable_squared)) if trainable_indices is not None else None,
        "nonfinite_elements": invalid,
        "per_tensor": per_tensor,
    }


def _finite(value):
    return value if math.isfinite(value) else None


def _publish(path, write):
    """Write atomically and refuse to replace an existing checkpoint."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, mode="wb", delete=False) as stream:
            temporary = Path(stream.name)
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic create; fails if destination already exists.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_json(path, value):
    _publish(path, lambda stream: stream.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode()))


def save_weights(path, weights):
    if any(np.asarray(w).dtype.kind not in "biuf" for w in weights):
        raise ValueError("Only numeric weight tensors can be archived")
    _publish(path, lambda stream: np.savez_compressed(stream, *weights))


class WeightArchive:
    """Server-side writer. Holds only one round baseline, not all clients' arrays."""

    def __init__(self, run_dir, num_clients, model=None):
        self.run_dir = Path(run_dir) / "models"
        self.run_dir.mkdir(exist_ok=False)
        self.num_clients = num_clients
        self.round = None
        self.baseline = None
        self.trainable_indices = None
        if model is not None:
            save_json(self.run_dir / "model_architecture.json", json.loads(model.to_json()))
            self.trainable_indices = {i for i, variable in enumerate(model.weights) if variable.trainable}
            save_json(self.run_dir / "weight_schema.json", {
                "model": model.name,
                "ordering": "Keras model.get_weights(); includes BatchNorm moving statistics",
                "tensors": [{"key": f"arr_{i}", "name": getattr(v, "path", v.name),
                             "shape": list(v.shape), "dtype": str(v.dtype), "trainable": bool(v.trainable)}
                            for i, v in enumerate(model.weights)],
            })

    def begin_round(self, server_round, arrays, selected_nodes):
        self.baseline = arrays.to_numpy_ndarrays()
        self.round = server_round
        self.selected_nodes = list(selected_nodes)
        save_weights(self.run_dir / f"round_{server_round:03d}_global_before.npz", self.baseline)

    def capture_clients(self, server_round, replies, array_key, metric_key="metrics"):
        if self.round != server_round or self.baseline is None:
            if not replies:
                return []
            raise ValueError("Cannot archive client updates without this round's global baseline")
        manifest = {"round": server_round, "reference": f"round_{server_round:03d}_global_before.npz",
                    "selected_node_ids": self.selected_nodes, "clients": [], "errors": []}
        seen = set()
        for message in replies:
            node_id = message.metadata.src_node_id
            if message.has_error():
                manifest["errors"].append({"node_id": node_id, "reason": message.error.reason})
                continue
            info = message.content["client-info"]
            partition = info["partition-id"]
            cid = client_id(partition)
            if not 0 <= partition < self.num_clients or info["client-id"] != cid or cid in seen:
                raise ValueError("Invalid or duplicate client identity in weight archive")
            seen.add(cid)
            weights = message.content[array_key].to_numpy_ndarrays()
            filename = f"round_{server_round:03d}_{cid}.npz"
            # Preserve the received weights even if they contain NaN/Inf.
            save_weights(self.run_dir / filename, weights)
            shift = weight_shift(weights, self.baseline, self.trainable_indices)
            manifest["clients"].append({"client_id": cid, "partition_id": partition,
                                        "role": client_role(partition), "node_id": node_id,
                                        "num_examples": message.content[metric_key]["num-examples"],
                                        "weights": filename, "shift": shift})
            del weights
        received = {message.metadata.src_node_id for message in replies}
        manifest["missing_node_ids"] = [n for n in self.selected_nodes if n not in received]
        save_json(self.run_dir / f"round_{server_round:03d}_weight_shifts.json", manifest)
        return manifest["clients"]

    def finish_round(self, server_round, arrays):
        try:
            if arrays is not None:
                save_weights(self.run_dir / f"round_{server_round:03d}_global_after.npz", arrays.to_numpy_ndarrays())
        finally:
            self.baseline = None


def main():
    parser = argparse.ArgumentParser(description="Compare client weights against a global baseline or another client.")
    parser.add_argument("weights")
    parser.add_argument("reference")
    args = parser.parse_args()
    print(json.dumps(weight_shift(load_weights(args.weights), load_weights(args.reference)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
