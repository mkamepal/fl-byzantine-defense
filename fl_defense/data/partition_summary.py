"""Validated CIFAR-10 statistics, printed once by the ServerApp before training."""

import csv
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np

from fl_defense.data.cifar10 import (
    _load_local_training_data, _partition_plan, resolve_dataset_dir,
)
from fl_defense.data.client_identity import client_id, client_role
from fl_defense.data.partition_plan import validate_partitions
from fl_defense.data.partitioning import PartitionSettings


CLASS_NAMES = (
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck",
)


def partition_statistics(labels, partitions, settings, run_id):
    validate_partitions(partitions, len(labels))
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("CIFAR-10 labels must be a one-dimensional integer array")
    if np.any(labels < 0) or np.any(labels >= len(CLASS_NAMES)):
        raise ValueError("CIFAR-10 labels must be in [0, 9]")
    rows = []
    for partition_id, partition in enumerate(partitions):
        row = {
            "run_id": run_id,
            "client_id": client_id(partition_id),
            "partition_id": partition_id,
            "role": client_role(partition_id),
            "partition_method": settings.method,
            "random_seed": settings.seed,
            "validation_fraction": 0.2,
            "dirichlet_alpha": settings.alpha if settings.method == "dirichlet" else "",
            "dirichlet_min_partition_size": settings.min_partition_size if settings.method == "dirichlet" else "",
            "dirichlet_self_balancing": settings.self_balancing if settings.method == "dirichlet" else "",
            "dirichlet_shuffle": settings.shuffle if settings.method == "dirichlet" else "",
            "source_split": "cifar10/train",
        }
        for prefix, indices in (("", partition.indices), ("train_", partition.train_indices),
                                ("validation_", partition.validation_indices)):
            counts = np.bincount(labels[indices], minlength=len(CLASS_NAMES))
            if int(counts.sum()) != len(indices):
                raise ValueError("Class counts do not match partition size")
            row[f"{prefix}total_images"] = len(indices)
            row.update({f"{prefix}{name.lower()}": int(count) for name, count in zip(CLASS_NAMES, counts)})
            row[f"{prefix}indices_sha256"] = hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()
        rows.append(row)
    return rows


def write_summary(rows, output_path):
    """Publish a complete CSV atomically, even when a previous experiment exists."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                         dir=output_path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def print_summary(rows):
    lines = ["========== DATASET PARTITION SUMMARY ==========", "",
             f"Method: {rows[0]['partition_method']} | Seed: {rows[0]['random_seed']}",
             "Totals include local training + local validation (80/20).",
             "Official CIFAR-10 test_batch is reserved for global testing and is not partitioned.", ""]
    for row in rows:
        lines.extend([
            f"Client ID: {row['client_id']} ({row['role']})",
            f"Partition ID: {row['partition_id']}",
            f"Total Images: {row['total_images']}",
            f"Training: {row['train_total_images']} | Local validation: {row['validation_total_images']}",
            "-" * 60,
            f"{'Class':<16}{'Total':>12}{'Training':>14}{'Validation':>14}",
            "-" * 60,
        ])
        for name in CLASS_NAMES:
            key = name.lower()
            lines.append(f"{name:<16}{row[key]:>12}{row['train_' + key]:>14}{row['validation_' + key]:>14}")
        lines.extend(["-" * 60, ""])
    lines.append(f"Validated: {len(rows)} clients, {sum(row['total_images'] for row in rows)} images; no overlapping or missing rows.")
    print("\n".join(lines), flush=True)


def prepare_partition_summary(config, num_partitions, run_id, output_path):
    """Use the clients' cached plan and an explicit run-owned CSV destination.

    The ServerApp supplies its ExperimentLogger directory. Never default to a
    shared results/partition_summary.csv, which loses per-experiment provenance.
    """
    settings = PartitionSettings.from_config(config)
    dataset_dir = resolve_dataset_dir(config.get("dataset-dir"))
    _, labels = _load_local_training_data(dataset_dir)
    partitions = _partition_plan(dataset_dir, num_partitions, settings)
    rows = partition_statistics(labels, partitions, settings, run_id)
    output = Path(output_path)
    write_summary(rows, output)
    print_summary(rows)
    print(f"Partition statistics saved to {output}", flush=True)
    return rows
