"""Backdoor attack success on triggered copies of non-target clean holdouts."""

import logging
import numpy as np
from fl_defense.attacks.backdoor import backdoor_attack
from fl_defense.attacks.common import validate_target, CIFAR10_CLASSES
from fl_defense.utils.weight_archive import save_json


def backdoor_success_counts(model, images, labels, target_label=0, trigger_value=1.0, batch_size=32):
    validate_target(target_label, CIFAR10_CLASSES)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('ASR batch_size must be positive')
    if len(images) != len(labels):
        raise ValueError('ASR image/label counts differ')
    indices = np.flatnonzero(np.asarray(labels).reshape(-1) != target_label)
    successes = 0
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        triggered, _ = backdoor_attack(images[selected], np.asarray(labels)[selected], target_label,
                                       poison_prob=1.0, trigger_value=trigger_value)
        predictions = np.asarray(model(triggered, training=False))
        if predictions.ndim != 2 or predictions.shape != (len(selected), len(CIFAR10_CLASSES)) or not np.isfinite(predictions).all():
            raise ValueError('ASR expects finite CIFAR-10 class predictions')
        successes += int(np.count_nonzero(predictions.argmax(axis=1) == target_label))
    return {'successes': successes, 'eligible': len(indices)}


def log_backdoor_evaluation(logger, server_round, replies):
    clients = []
    seen = set()
    for reply in replies:
        if reply.has_error() or 'backdoor-asr' not in reply.content:
            continue
        counts = reply.content['backdoor-asr']
        successes, eligible = counts['successes'], counts['eligible']
        if type(successes) is not int or type(eligible) is not int or not 0 <= successes <= eligible:
            raise ValueError('Invalid backdoor ASR counts')
        cid = reply.content['client-info']['client-id']
        if cid in seen:
            raise ValueError('Duplicate client in backdoor ASR results')
        seen.add(cid)
        rate = successes / eligible if eligible else None
        clients.append({'client_id': cid, 'successes': successes, 'eligible': eligible, 'asr': rate})
        metrics = {'eligible': eligible, 'successes': successes}
        if rate is not None:
            metrics['asr'] = rate
        logger.client_event(cid, 'backdoor', server_round, metrics)
    if not clients:
        return
    eligible = sum(c['eligible'] for c in clients)
    successes = sum(c['successes'] for c in clients)
    report = {'round': server_round, 'source': 'triggered copies of non-target local validation examples',
              'successes': successes, 'eligible': eligible, 'asr': successes / eligible if eligible else None,
              'clients': clients}
    directory = logger.run_dir / 'attacks'
    directory.mkdir(exist_ok=True)
    save_json(directory / f'round_{server_round:03d}_asr.json', report)
    metrics = {'backdoor_asr_eligible': eligible, 'backdoor_asr_successes': successes}
    if eligible:
        metrics['backdoor_asr'] = report['asr']
    else:
        logger.text('server/backdoor_asr_status', 'Unavailable: zero eligible examples', server_round)
    logger.server_event(server_round, metrics)
    logging.getLogger('flwr').info('Round %s backdoor ASR: %s (%s/%s eligible triggered holdout examples)',
                                  server_round, f"{report['asr']:.2%}" if eligible else 'unavailable', successes, eligible)
