"""Labeled inspection images, separate from arrays used for training."""

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from fl_defense.data.partition_summary import CLASS_NAMES
from fl_defense.utils.weight_archive import _publish, save_json


def log_labeled_previews(logger, tag, images, labels, source_indices, client_id,
                         stage, original_labels=None, poisoned_mask=None, attack=None):
    """Log and save the supplied preview samples in their existing order.

    Callers pass up to five selected training samples. Labels refer to the actual
    supplied data; original labels and the fixed poisoning mask explain attacks.
    """
    images, labels, source_indices = map(np.asarray, (images, labels, source_indices))
    count = len(images)
    if images.shape != (count, 32, 32, 3) or count > 5:
        raise ValueError("Preview requires at most five CIFAR-10 RGB images")
    if labels.shape != (count,) or labels.dtype.kind not in 'iu' or not np.isin(labels, range(10)).all():
        raise ValueError("Preview labels must match images and be CIFAR-10 class IDs")
    if source_indices.shape != (count,) or source_indices.dtype.kind not in 'iu':
        raise ValueError("Preview source indices must match images")
    if original_labels is not None:
        original_labels = np.asarray(original_labels)
        poisoned_mask = np.asarray(poisoned_mask)
        if (original_labels.shape != (count,) or original_labels.dtype.kind not in 'iu'
                or not np.isin(original_labels, range(10)).all()
                or poisoned_mask.shape != (count,) or poisoned_mask.dtype.kind != 'b'):
            raise ValueError("Preview original labels and poisoning mask must match images")
    if stage not in ('clean_training', 'attack_training', 'after_attack'):
        raise ValueError("Unknown preview stage")
    if stage == 'after_attack' and (original_labels is None or not poisoned_mask.all() or attack is None):
        raise ValueError("After-attack previews require only actually poisoned samples")
    if not count:
        return
    if images.dtype == np.uint8:
        pixels = images
    elif images.dtype.kind == 'f' and np.isfinite(images).all() and np.all((images >= 0) & (images <= 1)):
        pixels = np.rint(images * 255).astype(np.uint8)
    else:
        raise ValueError("Preview images must be uint8 or finite floats in [0, 1]")

    directory = logger.run_dir / 'images' / stage / client_id
    directory.mkdir(parents=True, exist_ok=False)
    font_path = font_manager.findfont('DejaVu Sans')
    title_font = ImageFont.truetype(font_path, 18)
    detail_font = ImageFont.truetype(font_path, 14)
    width, height = 224, 280 if original_labels is not None else 240
    strip = Image.new('RGB', (width * count, height), 'white')
    records = []
    for i in range(count):
        label = int(labels[i])
        name = CLASS_NAMES[label]
        tile = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(tile)
        draw.text((width // 2, 8), client_id, font=title_font, fill='#17365D', anchor='mt')
        draw.text((width // 2, 32), f'{name} ({label})', font=title_font, fill='black', anchor='mt')
        draw.text((width // 2, 56), f'Source row: {int(source_indices[i])}', font=detail_font,
                  fill='#444444', anchor='mt')
        # Nearest-neighbor enlargement keeps the 32x32 pixels and trigger visible.
        tile.paste(Image.fromarray(pixels[i]).resize((128, 128), Image.Resampling.NEAREST), (48, 78))
        record = {'position': i + 1, 'source_index': int(source_indices[i]),
                  'label_id': label, 'label_name': name}
        suffix = ''
        if original_labels is not None:
            original = int(original_labels[i])
            original_name = CLASS_NAMES[original]
            poisoned = bool(poisoned_mask[i])
            draw.text((width // 2, 214), f'Original: {original_name} ({original})',
                      font=detail_font, fill='#444444', anchor='mt')
            draw.text((width // 2, 236), f'Attack: {attack or "none"}',
                      font=detail_font, fill='#444444', anchor='mt')
            draw.text((width // 2, 258), f'Sample: {"poisoned" if poisoned else "clean"}',
                      font=detail_font, fill='#A4262C' if poisoned else '#21683B', anchor='mt')
            record.update(original_label_id=original, original_label_name=original_name,
                          attack=attack, poisoned=poisoned)
            if original != label:
                suffix = f'_from_{original_name.lower()}'
        else:
            draw.text((width // 2, 214), 'Clean training preview', font=detail_font,
                      fill='#444444', anchor='mt')
        filename = f'{i + 1:02d}_row_{int(source_indices[i])}_{name.lower()}{suffix}.png'
        _publish(directory / filename, lambda stream: tile.save(stream, format='PNG'))
        record['file'] = filename
        records.append(record)
        strip.paste(tile, (i * width, 0))
    _publish(directory / 'preview.png', lambda stream: strip.save(stream, format='PNG'))
    save_json(directory / 'labels.json', records)
    logger.images(tag, np.asarray(strip)[None], max_outputs=1)
