"""Build and validate the exact row assignments shared by summary and clients."""

from dataclasses import dataclass

from datasets import Dataset
import numpy as np

from fl_defense.data.partitioning import iid_partitioner, dirichlet_partitioner


@dataclass(frozen=True)
class ClientPartition:
    indices: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray


def validate_partitions(partitions, dataset_size):
    """Reject duplicates, missing rows, out-of-range rows, or split leakage."""
    if not partitions:
        raise ValueError("No client partitions")
    for partition in partitions:
        for indices in (partition.indices, partition.train_indices, partition.validation_indices):
            if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
                raise ValueError("Partition indices must be one-dimensional integers")
            if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= dataset_size):
                raise ValueError("Partition contains empty or out-of-range indices")
        split = np.concatenate((partition.train_indices, partition.validation_indices))
        if not np.array_equal(np.sort(split), np.sort(partition.indices)):
            raise ValueError("Training and validation must exactly cover their partition")
        if len(np.unique(split)) != len(split):
            raise ValueError("Training and validation overlap or contain duplicate rows")
    all_indices = np.concatenate([p.indices for p in partitions])
    if not np.array_equal(np.sort(all_indices), np.arange(dataset_size)):
        raise ValueError("Client partitions overlap or do not cover the training dataset")


def build_partition_plan(labels, num_partitions, settings):
    """Reuse Flower partitioners and the existing Hugging Face 80/20 split."""
    if num_partitions < 1 or len(labels) < 2 * num_partitions:
        raise ValueError("Each client needs at least two images for a train/validation split")
    if settings.method == "dirichlet":
        if settings.min_partition_size * num_partitions > len(labels):
            raise ValueError("Dirichlet minimum partition size exceeds available images")
        partitioner = dirichlet_partitioner(num_partitions, settings)
    else:
        # Preserve the baseline IID row assignments (no additional shuffle).
        partitioner = iid_partitioner(num_partitions)
    partitioner.dataset = Dataset.from_dict({"index": np.arange(len(labels)), "label": labels})
    partitions = []
    for partition_id in range(num_partitions):
        partition = partitioner.load_partition(partition_id)
        split = partition.train_test_split(test_size=0.2, seed=settings.seed)
        arrays = [np.asarray(rows["index"], dtype=np.int64) for rows in
                  (partition, split["train"], split["test"])]
        for array in arrays:
            array.flags.writeable = False
        partitions.append(ClientPartition(*arrays))
    validate_partitions(partitions, len(labels))
    return tuple(partitions)
