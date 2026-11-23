"""Scalar Flower settings and deterministic malicious-client assignment."""

from dataclasses import asdict, dataclass, field
import json
import math
import numpy as np

from fl_defense.attacks.common import CIFAR10_CLASSES, probability, validate_seed, validate_target
from fl_defense.detection.parameters import client_number

ATTACK_NAMES = ("label_flip", "backdoor", "model_replacement")
BACKDOOR_TYPES = ("backdoor", "model_replacement")


@dataclass(frozen=True)
class AttackConfig:
    enabled: bool = False
    seed: int = 42
    num_malicious: int = 3
    max_fake_ratio: float = 0.7
    exclude_head: bool = True
    head_client: str = "client_0"
    enabled_types: tuple = ("label_flip", "backdoor")
    distribution: dict | None = field(default_factory=lambda: {"label_flip": .6, "backdoor": .4})
    flip_prob: float = .4
    poison_prob: float = .4
    target_label: int = 0
    trigger_value: float = 1.0
    position: str = "bottom_right"
    patch_size_rule: str = "max_2_or_min_dimension_div_12"
    save_indices: bool = False
    evaluate_asr: bool = True

    replacement_variant: str = "basic"
    replacement_scale_mode: str = "population_weight_estimate"
    replacement_scale: float = 1.0
    replacement_start_round: int = 1
    replacement_end_round: int = 0  # zero means the last configured round
    replacement_interval: int = 1
    replacement_epochs: int = 0  # inherit local-epochs
    replacement_learning_rate: float = 0.0  # inherit learning-rate
    replacement_alpha: float = 0.5

    def replacement_active(self, server_round):
        if type(server_round) is not int or server_round < 1:
            raise ValueError("server_round must be a positive integer")
        return (server_round >= self.replacement_start_round
                and (self.replacement_end_round == 0 or server_round <= self.replacement_end_round)
                and (server_round - self.replacement_start_round) % self.replacement_interval == 0)

    @property
    def has_backdoor(self):
        return any(name in BACKDOOR_TYPES for name in self.enabled_types)

    def __post_init__(self):
        for key in ("enabled", "exclude_head", "save_indices", "evaluate_asr"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"attacks {key} must be boolean")
        if not self.exclude_head:
            raise ValueError("Head exclusion is mandatory: attacks-exclude-head must be true")
        validate_seed(self.seed)
        if type(self.num_malicious) is not int or self.num_malicious < 0:
            raise ValueError("attacks-num-malicious must be a nonnegative integer")
        client_number(self.head_client)
        for key in ("max_fake_ratio", "flip_prob", "poison_prob", "trigger_value"):
            probability(getattr(self, key), key)
        validate_target(self.target_label, CIFAR10_CLASSES)
        if self.position != "bottom_right" or self.patch_size_rule != "max_2_or_min_dimension_div_12":
            raise ValueError("Only bottom_right with max_2_or_min_dimension_div_12 is supported")
        if not isinstance(self.enabled_types, (tuple, list)) or not self.enabled_types or any(not isinstance(x, str) or x not in ATTACK_NAMES for x in self.enabled_types):
            raise ValueError(f"Enabled attacks must be a nonempty list drawn from {ATTACK_NAMES}")
        if len(set(self.enabled_types)) != len(self.enabled_types):
            raise ValueError("Enabled attack names must be unique")
        if self.replacement_variant not in ("basic", "constrain_and_scale"):
            raise ValueError("Invalid model replacement variant")
        if self.replacement_scale_mode not in ("fixed", "population_weight_estimate"):
            raise ValueError("Invalid model replacement scale mode")
        for key, minimum in (("replacement_start_round", 1), ("replacement_end_round", 0),
                             ("replacement_interval", 1), ("replacement_epochs", 0)):
            if type(getattr(self, key)) is not int or getattr(self, key) < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        if self.replacement_end_round and self.replacement_end_round < self.replacement_start_round:
            raise ValueError("Model replacement end round precedes start round")
        for key in ("replacement_scale", "replacement_learning_rate"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {key}")
        if self.replacement_scale == 0:
            raise ValueError("Model replacement scale must be positive")
        probability(self.replacement_alpha, "replacement_alpha")
        if self.replacement_alpha == 0:
            raise ValueError("Model replacement alpha must be positive")
        self.normalized_distribution()

    def normalized_distribution(self):
        if self.distribution is None:
            return None
        if not isinstance(self.distribution, dict) or any(name not in ATTACK_NAMES for name in self.distribution):
            raise ValueError("Unknown attack in distribution or invalid distribution object")
        weights = {name: self.distribution.get(name, 0) for name in sorted(self.enabled_types)}
        if any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in weights.values()):
            raise ValueError("Attack distribution weights must be finite and nonnegative")
        largest = max(weights.values())
        if largest <= 0:
            raise ValueError("Attack distribution must have a positive total over enabled types")
        scaled = {name: value / largest for name, value in weights.items()}
        total = sum(scaled.values())
        return {name: value / total for name, value in scaled.items()}

    @classmethod
    def from_config(cls, config):
        def decoded(key, default):
            value = config.get(key, default)
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a JSON string (Flower scalar setting)")
            try:
                return json.loads(value) if value.strip() else None
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON for {key}") from error
        seed = config.get("attacks-seed", -1)
        if seed == -1 and type(seed) is int:
            seed = config.get("partition-seed", 42)
        return cls(
            enabled=config.get("attacks-enabled", False), seed=seed,
            num_malicious=config.get("attacks-num-malicious", 3),
            max_fake_ratio=config.get("attacks-max-fake-ratio", .7),
            exclude_head=config.get("attacks-exclude-head", True),
            head_client=config.get("attacks-head-client", "client_0"),
            enabled_types=decoded("attacks-enabled-types", '["label_flip", "backdoor"]'),
            distribution=decoded("attacks-distribution", '{"label_flip":0.6,"backdoor":0.4}'),
            flip_prob=config.get("attacks-label-flip-prob", .4),
            poison_prob=config.get("attacks-backdoor-poison-prob", .4),
            target_label=config.get("attacks-backdoor-target-label", 0),
            trigger_value=config.get("attacks-backdoor-trigger-value", 1.0),
            position=config.get("attacks-backdoor-position", "bottom_right"),
            patch_size_rule=config.get("attacks-backdoor-patch-size-rule", "max_2_or_min_dimension_div_12"),
            save_indices=config.get("attacks-save-indices", False),
            evaluate_asr=config.get("attacks-evaluate-asr", True),
            replacement_variant=config.get("attacks-model-replacement-variant", "basic"),
            replacement_scale_mode=config.get("attacks-model-replacement-scale-mode", "population_weight_estimate"),
            replacement_scale=config.get("attacks-model-replacement-scale", 1.0),
            replacement_start_round=config.get("attacks-model-replacement-start-round", 1),
            replacement_end_round=config.get("attacks-model-replacement-end-round", 0),
            replacement_interval=config.get("attacks-model-replacement-interval", 1),
            replacement_epochs=config.get("attacks-model-replacement-epochs", 0),
            replacement_learning_rate=config.get("attacks-model-replacement-learning-rate", 0.0),
            replacement_alpha=config.get("attacks-model-replacement-alpha", 0.5),
        )

    def effective(self):
        return {**asdict(self), "normalized_distribution": self.normalized_distribution(),
                "seed_mode": "shared_RandomState_seed_no_round_offset", "assignment_fixed_across_rounds": True}


def assign_attacks(num_clients, settings, reference_client="client_0"):
    if type(num_clients) is not int or num_clients < 1:
        raise ValueError("num_clients must be a positive integer")
    protected = {"client_0", settings.head_client, reference_client}
    if any(client_number(cid) >= num_clients for cid in protected):
        raise ValueError("An attack/head reference is outside the configured client population")
    candidates = [f"client_{i}" for i in range(num_clients) if f"client_{i}" not in protected]
    count = min(settings.num_malicious, len(candidates), math.floor(num_clients * settings.max_fake_ratio)) if settings.enabled else 0
    rng = np.random.RandomState(settings.seed)
    chosen = sorted(rng.choice(candidates, size=count, replace=False).tolist(), key=client_number)
    distribution = settings.normalized_distribution()
    if distribution is None:
        types = rng.choice(sorted(settings.enabled_types), size=count).tolist()
    else:
        expected = {name: weight * count for name, weight in distribution.items()}
        allocated = {name: math.floor(value) for name, value in expected.items()}
        priority = sorted(expected, key=lambda name: (-(expected[name] - allocated[name]), name))
        for name in priority[:count - sum(allocated.values())]:
            allocated[name] += 1
        types = [name for name in sorted(allocated) for _ in range(allocated[name])]
        rng.shuffle(types)
    assignments = dict(zip(chosen, types))
    return assignments, {"requested_num_malicious": settings.num_malicious, "effective_num_malicious": count,
                         "total_clients": num_clients, "eligible_clients": len(candidates),
                         "malicious_client_fraction": count / num_clients,
                         "protected_clients": sorted(protected, key=client_number),
                         "selected_client_ids": chosen, "assignments": assignments,
                         "attack_counts": {name: types.count(name) for name in sorted(settings.enabled_types)},
                         "parameters": settings.effective()}
