"""Effective settings for the four-component defense (flat Flower config)."""

from dataclasses import asdict, dataclass, fields
import math


def finite_range(name, value, lower, upper):
    if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")


@dataclass(frozen=True)
class DefenseConfig:
    direction_threshold_degrees: float = 90.0
    direction_epsilon: float = 1e-12
    direction_reference: str = "unweighted_all_client_mean"
    magnitude_threshold: float = 0.25
    magnitude_boundary_rtol: float = 1e-5
    magnitude_boundary_atol: float = 1e-8
    magnitude_reference: str = "round_start_global_model"
    validation_accuracy_tolerance: float = 0.06
    trust_threshold: float = 0.50
    trust_weight_clustering: float = 0.25
    trust_weight_direction: float = 0.25
    trust_weight_magnitude: float = 0.25
    trust_weight_validation: float = 0.25

    def __post_init__(self):
        finite_range("direction_threshold_degrees", self.direction_threshold_degrees, 0, 180)
        finite_range("direction_epsilon", self.direction_epsilon, 0, math.inf)
        if self.direction_epsilon == 0:
            raise ValueError("direction_epsilon must be positive")
        for name in ("magnitude_threshold", "validation_accuracy_tolerance", "trust_threshold"):
            finite_range(name, getattr(self, name), 0, 1)
        for name in ("magnitude_boundary_rtol", "magnitude_boundary_atol"):
            finite_range(name, getattr(self, name), 0, math.inf)
        if self.direction_reference != "unweighted_all_client_mean":
            raise ValueError("Direction requires the unweighted_all_client_mean reference")
        if self.magnitude_reference != "round_start_global_model":
            raise ValueError("Magnitude requires the round_start_global_model reference")
        validate_trust_weights(self.weights)

    @property
    def weights(self):
        return {name: getattr(self, "trust_weight_" + name)
                for name in ("clustering", "direction", "magnitude", "validation")}

    @classmethod
    def from_config(cls, config):
        return cls(**{f.name: config[f.name.replace("_", "-")]
                      for f in fields(cls) if f.name.replace("_", "-") in config})

    def effective(self):
        return asdict(self)


def validate_trust_weights(weights):
    if set(weights) != {"clustering", "direction", "magnitude", "validation"}:
        raise ValueError("Trust requires exactly four component weights")
    for name, value in weights.items():
        finite_range(name, value, 0, 1)
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Trust weights must sum to 1")
