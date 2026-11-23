"""Dataset partitioning helpers."""

from dataclasses import dataclass
import math

from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner


@dataclass(frozen=True)
class PartitionSettings:
    method: str = "iid"
    seed: int = 42
    alpha: float = 0.5
    min_partition_size: int = 10
    self_balancing: bool = False
    shuffle: bool = True

    def __post_init__(self):
        if self.method not in ("iid", "dirichlet"):
            raise ValueError("partition-method must be 'iid' or 'dirichlet'")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("partition-seed must be a non-negative integer")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("dirichlet-alpha must be finite and positive")
        if not isinstance(self.min_partition_size, int) or self.min_partition_size < 2:
            raise ValueError("dirichlet-min-partition-size must be an integer >= 2")
        if not isinstance(self.self_balancing, bool) or not isinstance(self.shuffle, bool):
            raise ValueError("Dirichlet shuffle and self-balancing settings must be booleans")

    @classmethod
    def from_config(cls, config):
        return cls(
            method=config.get("partition-method", "iid"),
            seed=config.get("partition-seed", 42),
            alpha=config.get("dirichlet-alpha", 0.5),
            min_partition_size=config.get("dirichlet-min-partition-size", 10),
            self_balancing=config.get("dirichlet-self-balancing", False),
            shuffle=config.get("dirichlet-shuffle", True),
        )


def iid_partitioner(num_partitions):
    """Create the IID partitioner used by the baseline."""
    return IidPartitioner(num_partitions=num_partitions)


def dirichlet_partitioner(num_partitions, settings=PartitionSettings(method="dirichlet")):
    return DirichletPartitioner(
        num_partitions=num_partitions,
        partition_by="label",
        alpha=settings.alpha,
        min_partition_size=settings.min_partition_size,
        self_balancing=settings.self_balancing,
        shuffle=settings.shuffle,
        seed=settings.seed,
    )
