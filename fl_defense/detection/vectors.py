"""Float64, unnormalized trainable vectors for direction and magnitude."""

import numpy as np

from .parameters import client_number, schema_json, validate_weights


def aligned_matrix(vectors, head_client):
    ids = sorted(vectors, key=client_number)
    if head_client not in ids:
        raise ValueError(f"Trusted reference {head_client} is absent")
    rows = [np.asarray(vectors[cid], dtype=np.float64) for cid in ids]
    if any(row.ndim != 1 or row.size == 0 or row.shape != rows[0].shape for row in rows):
        raise ValueError("Mismatched vector dimensions")
    matrix = np.stack(rows)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite defense vectors")
    return ids, matrix, ids.index(head_client)


def model_vectors(clients, reference, schema):
    """Return W_i-mean(W_j) and W_i-W_t, each keyed by stable client ID.

    Compute the all-client mean before selection, in float64. Linearity permits
    flattening trainable tensors before taking this unweighted mean; buffers
    never enter either measurement. Clustering retains its existing float32 path.
    """
    validate_weights(reference, schema)
    expected = schema_json(schema)

    def flatten(weights):
        validate_weights(weights, schema)
        pieces = [np.asarray(weights[i], dtype=np.float64).reshape(-1)
                  for i, p in enumerate(schema) if p.trainable]
        if not pieces:
            raise ValueError("No trainable parameters")
        return np.concatenate(pieces)

    baseline = flatten(reference)
    ids = sorted(clients, key=client_number)
    if not ids:
        raise ValueError("No participating client models")
    full = []
    for cid in ids:
        weights, received = clients[cid]
        if received != expected:
            raise ValueError(f"Parameter names/order/schema mismatch for {cid}")
        full.append(flatten(weights))
    matrix = np.stack(full)
    mean = np.mean(matrix, axis=0, dtype=np.float64)
    residuals, updates = matrix - mean, matrix - baseline
    if not np.isfinite(residuals).all() or not np.isfinite(updates).all():
        raise ValueError("Non-finite model differences")
    return ({cid: residuals[i] for i, cid in enumerate(ids)},
            {cid: updates[i] for i, cid in enumerate(ids)})
