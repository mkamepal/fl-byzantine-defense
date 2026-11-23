"""Validation shared by NumPy data-poisoning functions used by Keras."""

import math
import numpy as np

CIFAR10_CLASSES = tuple(range(10))


def probability(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def validate_seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Attack seed must be an integer in [0, 2**32)")


def copy_labels(labels):
    array = np.asarray(labels)
    if array.ndim not in (1, 2) or (array.ndim == 2 and array.shape[1] != 1):
        raise ValueError("Labels must have shape (N,) or (N, 1)")
    if array.size and array.dtype.kind not in 'iu':
        raise ValueError("Labels must contain integer class IDs")
    return array.copy() if array.dtype.kind in 'iu' else array.astype(np.int32)


def validate_target(target_label, global_classes):
    if isinstance(target_label, (bool, np.bool_)) or not isinstance(target_label, (int, np.integer)) or target_label not in global_classes:
        raise ValueError("target_label must exist in the dataset global label space")
