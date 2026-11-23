"""Randomly replace selected labels with a different locally observed class."""

import numpy as np
from fl_defense.attacks.common import copy_labels, probability, validate_seed


def label_flip_attack(labels, flip_prob=0.4, seed=42, *, return_mask=False):
    probability(flip_prob, "flip_prob")
    validate_seed(seed)
    result = copy_labels(labels)
    flat = result.reshape(-1)
    classes = np.unique(flat)
    mask = np.zeros(len(flat), dtype=bool)
    if len(flat) and len(classes) >= 2:
        rng = np.random.RandomState(seed)
        mask = rng.rand(len(flat)) < flip_prob
        if mask.any():
            sources = np.searchsorted(classes, flat)
            offsets = rng.randint(1, len(classes), size=len(flat))
            flat[mask] = classes[(sources + offsets) % len(classes)][mask]
    return (result, mask) if return_mask else result
