"""Combine aligned binary components; membership strength is never an input."""

from .defense_config import finite_range, validate_trust_weights
from .parameters import client_number


def combine_scores(components, weights, threshold=0.50):
    validate_trust_weights(weights)
    finite_range("trust threshold", threshold, 0, 1)
    if set(components) != set(weights):
        raise ValueError("Missing trust component results")
    ids = set(components["clustering"])
    if not ids or any(set(scores) != ids for scores in components.values()):
        raise ValueError("Component client IDs are missing or misaligned")
    result = {}
    for cid in sorted(ids, key=client_number):
        for scores in components.values():
            if type(scores[cid]) is not int or scores[cid] not in (0, 1):
                raise ValueError(f"Invalid binary score for {cid}")
        trust = sum(weights[name] * components[name][cid] for name in weights)
        result[cid] = {"trust_score": trust, "selected": trust >= threshold}
    return result
