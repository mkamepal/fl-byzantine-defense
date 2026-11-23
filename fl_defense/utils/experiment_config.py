"""Validate run-dependent counts before starting a Flower experiment."""

import math

from fl_defense.detection.clustering import ClusteringConfig
from fl_defense.detection.parameters import client_number
from fl_defense.attacks.config import AttackConfig, assign_attacks
from fl_defense.detection.defense_config import DefenseConfig


def validate_experiment_config(config):
    """Shared launcher/ServerApp checks; never rewrite requested parameters."""
    for key in ("num-clients", "num-server-rounds"):
        value = config.get(key)
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer, received {value!r}")
    count = config["num-clients"]
    if type(config.get("trust-enabled", True)) is not bool:
        raise ValueError("trust-enabled must be a boolean")
    DefenseConfig.from_config(config)
    attacks = AttackConfig.from_config(config)
    if attacks.enabled and "model_replacement" in attacks.enabled_types:
        if attacks.replacement_start_round > config["num-server-rounds"]:
            raise ValueError("Model replacement start round exceeds configured rounds")
        if attacks.replacement_end_round > config["num-server-rounds"]:
            raise ValueError("Model replacement end round exceeds configured rounds")
    if attacks.enabled:
        assign_attacks(count, attacks, config.get('clustering-head-client', 'client_0'))
    fraction = config.get("fraction-train", 1.0)
    if type(fraction) not in (float, int) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("fraction-train must be a finite number in [0, 1]")
    enabled = config.get("clustering-enabled", False)
    if type(enabled) is not bool:
        raise ValueError("clustering-enabled must be a boolean")
    if enabled:
        defense = ClusteringConfig.from_config(config)
        if client_number(defense.head_client) >= count:
            raise ValueError(f"Trusted reference {defense.head_client} is outside this run's {count} clients")
        defense.validate_population(count)
        # Match the application's FedAvg sampling defaults, capped for N=1.
        sampled = max(int(count * fraction), min(2, count)) if fraction > 0 else 0
        try:
            defense.validate_population(sampled)
        except ValueError as error:
            raise ValueError(
                f"This run has {count} clients but fraction-train={fraction} requests {sampled} per round. {error}"
            ) from error
