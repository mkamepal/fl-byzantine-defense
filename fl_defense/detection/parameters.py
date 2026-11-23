"""Keras model-weight wire schema, including explicit trainable membership."""

from dataclasses import asdict, dataclass
import json
import re

import numpy as np


@dataclass(frozen=True)
class Parameter:
    name: str
    shape: tuple[int, ...]
    trainable: bool
    dtype: str = "float32"


def model_schema(model):
    trainable = {id(v) for v in model.trainable_weights}
    return tuple(Parameter(getattr(v, "path", v.name), tuple(v.shape),
                           id(v) in trainable, str(v.dtype)) for v in model.weights)


def schema_json(schema):
    return json.dumps([asdict(p) for p in schema], sort_keys=True, separators=(",", ":"))


def client_number(cid):
    if not isinstance(cid, str) or re.fullmatch(r"client_(0|[1-9][0-9]*)", cid) is None:
        raise ValueError(f"Invalid stable client ID: {cid!r}")
    return int(cid.removeprefix("client_"))


def validate_weights(weights, schema):
    if len(weights) != len(schema):
        raise ValueError("Parameter count mismatch")
    for i, parameter in enumerate(schema):
        value = np.asarray(weights[i])
        if value.shape != parameter.shape:
            raise ValueError(f"Parameter shape mismatch: {parameter.name}")
        if value.dtype.name != parameter.dtype:
            raise ValueError(f"Parameter dtype mismatch: {parameter.name}")
        if value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError(f"Non-finite or non-floating parameter: {parameter.name}")


def flatten_update(weights, reference, schema):
    """Full models in Keras get_weights order -> W_i - W_t, trainable only."""
    validate_weights(weights, schema)
    validate_weights(reference, schema)
    pieces = []
    with np.errstate(over="ignore", invalid="ignore"):
        for i, parameter in enumerate(schema):
            if parameter.trainable:
                # Subtract before the analysis dtype conversion (Keras uses float32).
                pieces.append((weights[i] - reference[i]).reshape(-1).astype(np.float32))
    if not pieces:
        raise ValueError("No trainable parameters")
    vector = np.concatenate(pieces)
    dimension = sum(int(np.prod(p.shape)) for p in schema if p.trainable)
    if vector.size != dimension or not np.isfinite(vector).all():
        raise ValueError("Invalid update vector dimension or non-finite update")
    return vector
