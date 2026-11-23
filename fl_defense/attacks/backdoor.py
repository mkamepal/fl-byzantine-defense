"""Bottom-right square trigger, with pixel values expressed on a [0, 1] scale."""

import numpy as np
from fl_defense.attacks.common import (
    CIFAR10_CLASSES, copy_labels, probability, validate_seed, validate_target,
)


def image_geometry(images, layout):
    images = np.asarray(images)
    if layout == "HWC" and images.ndim == 4:
        _, height, width, channels = images.shape
    elif layout == "CHW" and images.ndim == 4:
        _, channels, height, width = images.shape
    elif layout == "HW" and images.ndim == 3:
        _, height, width = images.shape
        channels = 1
    else:
        raise ValueError("Expected a batch of HWC, CHW, or HW images matching the explicit layout")
    if height < 2 or width < 2 or channels < 1:
        raise ValueError("Images must have at least 2x2 spatial dimensions and one channel")
    return height, width, max(2, min(height, width) // 12)


def backdoor_attack(images, labels, target_label=0, poison_prob=0.4, trigger_value=1.0,
                    seed=42, *, layout="HWC", global_classes=CIFAR10_CLASSES, return_mask=False):
    """Return copied poisoned images/labels; optionally also the selected-sample mask.

    Supports floating images in [0,1] and raw uint8 images. Other normalized spaces
    must be converted explicitly by their caller. Every channel receives the patch.
    """
    probability(poison_prob, "poison_prob")
    probability(trigger_value, "trigger_value")
    validate_seed(seed)
    validate_target(target_label, global_classes)
    source = np.asarray(images)
    output_labels = copy_labels(labels)
    height, width, size = image_geometry(source, layout)
    if len(source) != len(output_labels):
        raise ValueError("Image and label counts must match")
    if source.dtype == np.uint8:
        value = np.uint8(np.rint(trigger_value * 255))
    elif source.dtype.kind == 'f' and np.isfinite(source).all() and (not source.size or (source.min() >= 0 and source.max() <= 1)):
        value = trigger_value
    else:
        raise ValueError("Images must be uint8 pixels or finite floating pixels scaled to [0, 1]")
    if not np.isin(output_labels, global_classes).all():
        raise ValueError("Training labels are outside the dataset global label space")
    output = source.copy()
    mask = np.random.RandomState(seed).rand(len(source)) < poison_prob
    if layout == "HWC":
        output[mask, height-size:height, width-size:width, :] = value
    elif layout == "CHW":
        output[mask, :, height-size:height, width-size:width] = value
    else:
        output[mask, height-size:height, width-size:width] = value
    output_labels[mask] = target_label
    result = (output, output_labels, mask) if return_mask else (output, output_labels)
    return result
