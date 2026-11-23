"""Prepare poisoning once per experiment; reuse saved overlays in every worker."""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from fl_defense.attacks.config import AttackConfig, assign_attacks, BACKDOOR_TYPES
from fl_defense.attacks.dispatch import apply_attacks_for_client
from fl_defense.data.cifar10 import load_client_split, resolve_dataset_dir, _partition_plan
from fl_defense.data.partitioning import PartitionSettings
from fl_defense.utils.weight_archive import _publish, save_json


def indices_digest(indices):
    return hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest()


def _log_attack_assignments(num_clients, assignments, enabled, protected=()):
    clients = [f'client_{i}' for i in range(num_clients)]
    fake = [f'{cid} ({assignments[cid]})' for cid in clients if cid in assignments]
    clean = [cid for cid in clients if cid not in assignments]
    rows = ['Simulated attack assignments (ground truth; fixed across rounds)',
            f"Attacks enabled: {'yes' if enabled else 'no'}",
            f"Fake clients ({len(fake)}/{num_clients}): {', '.join(fake) or 'none'}",
            f"Clean clients ({len(clean)}/{num_clients}): {', '.join(clean) or 'none'}"]
    if protected:
        rows.append(f"Protected reference clients: {', '.join(protected)}")
    if not enabled:
        rows.append('Clients with attacks performed: none (attacks disabled)')
    rows.append('These are configured attack assignments, not defense predictions.')
    logging.getLogger('flwr').info('\n%s', '\n'.join(rows))


def prepare_attack_datasets(logger, config):
    """Server-only construction, after the clean partition plan has been established.

    Saved labels and sparse poisoned-image rows are training inputs, not defense
    features. No attack assignments are forwarded in training replies or metrics.
    """
    settings = AttackConfig.from_config(config)
    num_clients = config['num-clients']
    if not settings.enabled:
        _log_attack_assignments(num_clients, {}, enabled=False)
        return None
    from fl_defense.utils.image_previews import log_labeled_previews
    from fl_defense.data.partition_summary import CLASS_NAMES

    assignments, summary = assign_attacks(num_clients, settings, config.get('clustering-head-client', 'client_0'))
    _log_attack_assignments(num_clients, assignments, enabled=True, protected=summary['protected_clients'])
    root = resolve_dataset_dir(config.get('dataset-dir'))
    partition_settings = PartitionSettings.from_config(config)
    plan = _partition_plan(root, num_clients, partition_settings)
    directory = logger.run_dir / 'attacks'
    directory.mkdir(exist_ok=False)
    overlays = directory / 'training'
    overlays.mkdir()
    summary['clients'] = []
    for partition_id, partition in enumerate(plan):
        cid = f'client_{partition_id}'
        images, labels = load_client_split(partition_id, num_clients, 'train', root, partition_settings)
        poisoned = apply_attacks_for_client(images, labels, cid, assignments, settings, settings.seed)
        positions = np.flatnonzero(poisoned.mask)
        stamped = (poisoned.images[positions] if assignments.get(cid) in BACKDOOR_TYPES
                   else np.empty((0, *images.shape[1:]), dtype=images.dtype))
        digest = indices_digest(partition.train_indices)
        # Needed to reuse exact poisoned examples without RNG calls in later rounds.
        _publish(overlays / f'{cid}.npz', lambda stream: np.savez_compressed(
            stream, labels=poisoned.labels, positions=positions, stamped_images=stamped,
            train_indices_sha256=np.array(digest),
        ))
        metadata = {**poisoned.metadata, 'partition_id': partition_id, 'train_indices_sha256': digest,
                    'poisoned_source_indices_sha256': indices_digest(partition.train_indices[positions])}
        if settings.save_indices:
            filename = f'{cid}_poisoned_indices.npz'
            _publish(directory / filename, lambda stream: np.savez_compressed(
                stream, training_positions=positions, source_indices=partition.train_indices[positions]))
            metadata['poisoned_indices_file'] = filename
        summary['clients'].append(metadata)
        logger.text(f'attacks/{cid}/training_data', '```json\n' + json.dumps(metadata, indent=2) + '\n```')
        logger.client_event(cid, 'attacks', 0, {key: metadata[key] for key in
                            ('training_samples', 'poisoned_samples', 'poisoned_fraction', 'labels_changed', 'trigger_stamped_samples')})
        preview = poisoned.images[:5]
        if len(preview):
            log_labeled_previews(logger, f'attacks/{cid}/training_images', preview, poisoned.labels[:5],
                                 partition.train_indices[:5], cid, 'attack_training',
                                 original_labels=labels[:5], poisoned_mask=poisoned.mask[:5],
                                 attack=assignments.get(cid))
            logger.text(f'attacks/{cid}/training_image_labels',
                        ', '.join(f'{CLASS_NAMES[int(label)]} ({int(label)})' for label in poisoned.labels[:5]))
        # Dedicated post-attack examples: select from the already-applied mask,
        # never resample attacks or include unmodified training examples here.
        attacked_preview_positions = positions[:5]
        if len(attacked_preview_positions):
            log_labeled_previews(logger, f'attacks/{cid}/poisoned_images',
                                 poisoned.images[attacked_preview_positions],
                                 poisoned.labels[attacked_preview_positions],
                                 partition.train_indices[attacked_preview_positions], cid, 'after_attack',
                                 original_labels=labels[attacked_preview_positions],
                                 poisoned_mask=poisoned.mask[attacked_preview_positions], attack=assignments[cid])
        else:
            logger.text(f'attacks/{cid}/poisoned_images_status', 'No poisoned training samples for this client.')
    summary['total_training_samples'] = sum(c['training_samples'] for c in summary['clients'])
    summary['total_poisoned_training_samples'] = sum(c['poisoned_samples'] for c in summary['clients'])
    summary['experiment_poisoned_training_fraction'] = (summary['total_poisoned_training_samples'] / summary['total_training_samples']
                                                       if summary['total_training_samples'] else 0.0)
    summary['numpy_version'] = np.__version__
    if 'model_replacement' in settings.enabled_types:
        from fl_defense.attacks.model_replacement import population_scale
        summary['model_replacement'] = population_scale(summary)
        summary['model_replacement']['tensor_policy'] = 'trainable_only; retain locally trained nontrainable buffers'
        summary['model_replacement']['training_overlay_policy'] = 'applied only on scheduled replacement rounds'
        (directory / 'model_replacement').mkdir()

    save_json(directory / 'assignment_and_poisoning.json', summary)
    logger.text('attacks/assignment', '```json\n' + json.dumps({k: v for k, v in summary.items() if k != 'clients'}, indent=2) + '\n```')
    logger.text('dataset/poisoning_note', 'dataset/ summaries show clean pre-poisoning partitions. attacks/ shows actual training previews and poisoning diagnostics. Holdouts remain clean.')
    logger.flush()
    attacked = [entry for entry in summary['clients'] if entry['poisoned_samples'] > 0]
    rows = ['Clients with attacks performed:' if 'model_replacement' not in settings.enabled_types
            else 'Prepared training overlays (model replacement executes on scheduled client rounds):']
    if attacked:
        for entry in attacked:
            rows.append(f"  {entry['client_id']} -> {entry['attack']} ({entry['poisoned_samples']} poisoned training samples)")
    else:
        rows.append('  none (no training samples were poisoned)')
    rows.extend([f"Attacks: requested malicious={summary['requested_num_malicious']}, effective={summary['effective_num_malicious']}/{num_clients}",
                 'Client           Attack          Poisoned training samples    Labels changed'])
    for entry in summary['clients']:
        rows.append(f"{entry['client_id']:<16} {str(entry['attack'] or 'clean'):<15} "
                    f"{entry['poisoned_samples']}/{entry['training_samples']} ({entry['poisoned_fraction']:.2%})"
                    f"    {entry['labels_changed']}")
    logging.getLogger('flwr').info('\n%s', '\n'.join(rows))
    return str(overlays.resolve())


def load_training_data(partition_id, num_clients, dataset_dir, partition_settings, poisoning_dir=None, *, apply_overlay=True):
    images, labels = load_client_split(partition_id, num_clients, 'train', dataset_dir, partition_settings)
    if poisoning_dir is None:
        return images, labels
    partition = _partition_plan(resolve_dataset_dir(dataset_dir), num_clients, partition_settings)[partition_id]
    path = Path(poisoning_dir) / f'client_{partition_id}.npz'
    # A missing/mismatched overlay must fail, not silently turn an attack into a clean run.
    with np.load(path, allow_pickle=False) as saved:
        if saved['train_indices_sha256'].item() != indices_digest(partition.train_indices):
            raise ValueError('Poisoning overlay does not match this client training partition')
        attacked_labels = saved['labels']
        positions = saved['positions']
        stamped = saved['stamped_images']
        if (attacked_labels.shape != labels.shape or attacked_labels.dtype != labels.dtype
                or not np.isin(attacked_labels, range(10)).all()):
            raise ValueError('Invalid poisoning label overlay')
        if (positions.ndim != 1 or positions.dtype.kind not in 'iu' or len(np.unique(positions)) != len(positions)
                or np.any(positions < 0) or np.any(positions >= len(images))):
            raise ValueError('Invalid poisoned training positions')
        if stamped.shape not in ((0, *images.shape[1:]), (len(positions), *images.shape[1:])):
            raise ValueError('Invalid stamped-image overlay dimensions')
        if stamped.dtype != images.dtype or not np.isfinite(stamped).all() or np.any(stamped < 0) or np.any(stamped > 1):
            raise ValueError('Invalid stamped-image overlay values')
        if not apply_overlay:
            return images, labels
        if len(stamped):
            images = images.copy()
            images[positions] = stamped
        return images, attacked_labels
