"""Training overlays for one assigned attack family per malicious client."""

from dataclasses import dataclass
import numpy as np

from fl_defense.attacks.backdoor import backdoor_attack, image_geometry
from fl_defense.attacks.common import CIFAR10_CLASSES
from fl_defense.attacks.config import BACKDOOR_TYPES
from fl_defense.attacks.label_flipping import label_flip_attack


@dataclass
class PoisonedData:
    images: np.ndarray
    labels: np.ndarray
    mask: np.ndarray
    metadata: dict


def histogram(labels):
    return {str(label): int(np.count_nonzero(np.asarray(labels) == label)) for label in CIFAR10_CLASSES}


def apply_attacks_for_client(images, labels, client_id, malicious_clients_attack_map, attack_params, seed):
    """Input arrays are never mutated. Unassigned clients reuse clean arrays."""
    attack = malicious_clients_attack_map.get(client_id)
    original = np.asarray(labels)
    images = np.asarray(images)
    if len(images) != len(original):
        raise ValueError("Image and label counts must match")
    if attack is not None and client_id in ("client_0", attack_params.head_client):
        raise ValueError("Trusted head data must never be poisoned")
    mask = np.zeros(len(original), dtype=bool)
    result_images, result_labels = images, original
    patch = None
    if attack == "label_flip":
        result_labels, mask = label_flip_attack(original, attack_params.flip_prob, seed, return_mask=True)
    elif attack in BACKDOOR_TYPES:
        result_images, result_labels, mask = backdoor_attack(
            images, original, attack_params.target_label, attack_params.poison_prob,
            attack_params.trigger_value, seed, return_mask=True,
        )
        size = image_geometry(images, "HWC")[2]
        patch = [size, size]
    elif attack is not None:
        raise ValueError(f"Unknown configured attack: {attack!r}")
    changed = int(np.count_nonzero(original != result_labels))
    selected = int(np.count_nonzero(mask))
    return PoisonedData(result_images, result_labels, mask, {
        "client_id": client_id, "attack": attack, "seed": seed,
        "training_samples": len(original), "poisoned_samples": selected,
        "poisoned_fraction": selected / len(original) if len(original) else 0.0,
        "labels_changed": changed, "labels_changed_fraction": changed / len(original) if len(original) else 0.0,
        "trigger_stamped_samples": selected if attack in BACKDOOR_TYPES else 0,
        "original_class_histogram": histogram(original), "attacked_class_histogram": histogram(result_labels),
        "patch_dimensions": patch,
        "insertion_stage": "after uint8/255 scaling; no spatial augmentation or mean/std normalization" if patch else None,
        "layout": "HWC" if patch else None,
    })
