"""Inclusive head-relative bounds on raw standard-update L2 norms."""

import numpy as np

from .defense_config import finite_range
from .vectors import aligned_matrix


def measure_magnitude(updates, head_client="client_0", threshold=0.25, rtol=1e-5, atol=1e-8):
    finite_range("magnitude threshold", threshold, 0, 1)
    finite_range("boundary rtol", rtol, 0, float("inf"))
    finite_range("boundary atol", atol, 0, float("inf"))
    ids, matrix, head = aligned_matrix(updates, head_client)
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(norms).all():
        raise ValueError("Non-finite update norm")
    lower, upper = float(norms[head] * (1 - threshold)), float(norms[head] * (1 + threshold))
    return {cid: {"norm": float(norms[i]), "head_norm": float(norms[head]), "threshold": threshold,
                  "lower_bound": lower, "upper_bound": upper,
                  "score": int(lower <= norms[i] <= upper or
                               np.isclose(norms[i], lower, rtol=rtol, atol=atol) or
                               np.isclose(norms[i], upper, rtol=rtol, atol=atol))}
            for i, cid in enumerate(ids)}
