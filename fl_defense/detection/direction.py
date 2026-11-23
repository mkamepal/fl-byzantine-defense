"""Head-relative angles of all-client-mean residuals, without normalization."""

import numpy as np

from .defense_config import finite_range
from .vectors import aligned_matrix


def measure_direction(residuals, head_client="client_0", threshold_degrees=90.0, epsilon=1e-12):
    finite_range("direction threshold", threshold_degrees, 0, 180)
    finite_range("direction epsilon", epsilon, 0, float("inf"))
    if epsilon == 0:
        raise ValueError("Direction epsilon must be positive")
    ids, matrix, head = aligned_matrix(residuals, head_client)
    norms = np.linalg.norm(matrix, axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        cosine = (matrix @ matrix[head]) / (norms * norms[head] + epsilon)
    if not np.isfinite(norms).all() or not np.isfinite(cosine).all():
        raise ValueError("Non-finite direction measurement")
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    angles[head] = 0.0
    return {cid: {"angle": float(angles[i]), "threshold_degrees": threshold_degrees,
                  "residual_norm": float(norms[i]), "head_residual_norm": float(norms[head]),
                  "zero_residual": bool(norms[i] == 0), "head_zero_residual": bool(norms[head] == 0),
                  "score": int(angles[i] < threshold_degrees)} for i, cid in enumerate(ids)}
