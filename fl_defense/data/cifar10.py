"""Download CIFAR-10 once, then partition the local data for clients."""

from functools import lru_cache
import hashlib
import os
from pathlib import Path
import pickle
import tarfile
from urllib.request import urlopen

from filelock import FileLock
import numpy as np

from fl_defense.data.partitioning import PartitionSettings
from fl_defense.data.partition_plan import build_partition_plan


DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[2] / "datasets"
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"


def _download_archive(destination):
    """Download quietly and publish only a complete, verified archive."""
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    try:
        with urlopen(CIFAR10_URL, timeout=60) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != CIFAR10_SHA256:
            raise ValueError("CIFAR-10 download checksum mismatch; please retry")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"CIFAR-10 download complete: {destination}", flush=True)


@lru_cache(maxsize=1)
def _load_local_training_data(dataset_dir):
    """Read the existing archive, downloading it only when missing."""
    root = Path(dataset_dir)
    folder = root / "cifar10"
    folder.mkdir(parents=True, exist_ok=True)
    # Ray workers may request the dataset simultaneously on first startup.
    with FileLock(str(folder / ".download.lock")):
        archive_path = folder / "cifar-10-python.tar.gz"
        if not archive_path.is_file():
            print(f"CIFAR-10 missing; downloading to {archive_path}", flush=True)
            _download_archive(archive_path)

    images, labels = [], []
    # Read only the known batch members; no archive extraction is required.
    with tarfile.open(archive_path, "r:gz") as archive:
        for batch_id in range(1, 6):
            with archive.extractfile(f"cifar-10-batches-py/data_batch_{batch_id}") as stream:
                batch = pickle.load(stream, encoding="bytes")
            images.append(batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
            labels.append(np.asarray(batch[b"labels"], dtype=np.int64))
    return np.concatenate(images), np.concatenate(labels)


def resolve_dataset_dir(dataset_dir=None):
    return str(Path(dataset_dir or os.environ.get("FL_DATASET_DIR", DEFAULT_DATASET_DIR)).expanduser().resolve())


@lru_cache(maxsize=1)
def _partition_plan(dataset_dir, num_partitions, settings):
    _, labels = _load_local_training_data(dataset_dir)
    return build_partition_plan(labels, num_partitions, settings)


def load_data(partition_id, num_partitions, dataset_dir=None, settings=PartitionSettings()):
    """Return local training and validation data; never the official test batch."""
    if not 0 <= partition_id < num_partitions:
        raise ValueError("partition_id is outside the configured client range")
    dataset_dir = resolve_dataset_dir(dataset_dir)
    images, labels = _load_local_training_data(dataset_dir)
    partition = _partition_plan(dataset_dir, num_partitions, settings)[partition_id]
    train_indices = partition.train_indices
    validation_indices = partition.validation_indices
    return (
        images[train_indices].astype("float32") / 255.0,
        labels[train_indices],
        images[validation_indices].astype("float32") / 255.0,
        labels[validation_indices],
    )


def load_client_split(partition_id, num_partitions, split, dataset_dir=None,
                      settings=PartitionSettings()):
    """Materialize only the requested split, retaining the same cached indices."""
    if split not in ("train", "validation"):
        raise ValueError("split must be 'train' or 'validation'")
    if not 0 <= partition_id < num_partitions:
        raise ValueError("partition_id is outside the configured client range")
    dataset_dir = resolve_dataset_dir(dataset_dir)
    images, labels = _load_local_training_data(dataset_dir)
    partition = _partition_plan(dataset_dir, num_partitions, settings)[partition_id]
    indices = partition.train_indices if split == "train" else partition.validation_indices
    selected = images[indices].astype("float32")
    selected /= 255.0
    return selected, labels[indices]
