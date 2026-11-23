"""Stable logical identities independent of Flower worker scheduling."""


def client_id(partition_id):
    if not isinstance(partition_id, int) or isinstance(partition_id, bool) or partition_id < 0:
        raise ValueError("partition_id must be a non-negative integer")
    return f"client_{partition_id}"


def client_role(partition_id):
    return "trusted_head" if partition_id == 0 else "client"
