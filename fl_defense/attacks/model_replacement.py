"""Model replacement adapted to Keras full-model replies and sample-weighted FedAvg.

Ground truth travels through local simulation artifacts, never training replies
or defense features. No server selection outcome is used to construct an attack.
"""

from contextlib import contextmanager
import hashlib
import json
import logging
import math
from pathlib import Path

import numpy as np

from fl_defense.attacks.config import assign_attacks
from fl_defense.detection.parameters import validate_weights
from fl_defense.utils.weight_archive import save_json


def population_scale(summary):
    """Estimate inverse replacement weight from the FULL configured population.

    This assumes representative participation. It is not the weight of actual
    replies or of accepted clients; other attack types do not enter the denominator.
    """
    clients = summary['clients']
    total = sum(c['training_samples'] for c in clients)
    replacement = sum(c['training_samples'] for c in clients if c['attack'] == 'model_replacement')
    if total <= 0 or replacement < 0 or replacement > total:
        raise ValueError('Invalid training sample totals for model replacement')
    return {'population_training_samples': total, 'replacement_training_samples': replacement,
            'estimated_weight': replacement / total,
            'estimated_scale': total / replacement if replacement else None,
            'scale_assumption': 'full_population_sample_weights; representative participation; server step=1'}


def load_assignment(poisoning_dir, settings, num_clients, reference_client):
    """Validate the server-prepared assignment before choosing a worker branch."""
    if not poisoning_dir:
        raise ValueError('Model replacement requires prepared attack artifacts')
    summary = json.loads((Path(poisoning_dir).parent / 'assignment_and_poisoning.json').read_text())
    expected, _ = assign_attacks(num_clients, settings, reference_client)
    # JSON normalizes tuple fields in the dataclass to lists.
    parameters = json.loads(json.dumps(settings.effective()))
    if (summary['total_clients'] != num_clients or summary['assignments'] != expected
            or summary['parameters'] != parameters):
        raise ValueError('Attack manifest does not match effective configuration')
    clients = summary['clients']
    if len(clients) != num_clients or {c['client_id'] for c in clients} != {f'client_{i}' for i in range(num_clients)}:
        raise ValueError('Attack manifest client coverage mismatch')
    for c in clients:
        if c['attack'] != expected.get(c['client_id']) or type(c['training_samples']) is not int or c['training_samples'] <= 0:
            raise ValueError('Invalid attack manifest client metadata')
    if summary['model_replacement']['estimated_scale'] != population_scale(summary)['estimated_scale']:
        raise ValueError('Model replacement population scale mismatch')
    return summary


def choose_scale(settings, summary):
    value = (settings.replacement_scale if settings.replacement_scale_mode == 'fixed'
             else population_scale(summary)['estimated_scale'])
    if value is None or not math.isfinite(value) or value <= 0:
        raise ValueError('No positive model replacement scale is available')
    return float(value)


def trainable_norm(weights, reference, schema):
    validate_weights(weights, schema)
    validate_weights(reference, schema)
    squared = 0.0
    for i, p in enumerate(schema):
        if p.trainable:
            delta = weights[i].astype(np.float64) - reference[i].astype(np.float64)
            squared += float(np.sum(delta * delta))
    value = math.sqrt(squared)
    if not math.isfinite(value):
        raise ValueError('Non-finite model replacement update norm')
    return value


def replace_model(weights, reference, schema, scale):
    """Return new arrays: W_t + scale*(X-W_t), trainable only; preserve BN buffers."""
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Model replacement scale must be finite and positive')
    before = trainable_norm(weights, reference, schema)
    if not any(p.trainable for p in schema):
        raise ValueError('Model replacement requires trainable parameters')
    result = []
    with np.errstate(over='ignore', invalid='ignore'):
        for i, p in enumerate(schema):
            if p.trainable:
                base = reference[i].astype(np.float64)
                value = (base + scale * (weights[i].astype(np.float64) - base)).astype(p.dtype)
            else:
                value = weights[i].copy()
            result.append(value)
    validate_weights(result, schema)  # Includes overflow on conversion to wire dtype.
    return result, {'pre_scale_trainable_l2': before,
                    'submitted_trainable_l2': trainable_norm(result, reference, schema),
                    'zero_update': before == 0,
                    'tensor_policy': 'trainable_only; retain locally trained nontrainable buffers'}


def weights_digest(weights):
    digest = hashlib.sha256()
    for a in weights:
        digest.update(str((a.shape, a.dtype.str)).encode())
        digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()


@contextmanager
def constrained_loss(model, reference, schema, alpha):
    """Temporary eager loss; no second model or persistent tf.function graphs.

    alpha*classification + (1-alpha)*mean_squared_trainable_distance + existing
    regularizers. Mean squared distance is an explicit smooth anomaly surrogate,
    not a reproduction of the server's four-component decision function.
    """
    import tensorflow as tf
    validate_weights(reference, schema)
    if not 0 < alpha <= 1:
        raise ValueError('Constrain-and-scale alpha must be in (0,1]')
    if not model.run_eagerly:
        raise ValueError('Constrain-and-scale requires eager training')
    pairs = [(model.weights[i], tf.constant(reference[i])) for i, p in enumerate(schema) if p.trainable]
    dimension = sum(int(np.prod(p.shape)) for p in schema if p.trainable)
    if not pairs or len(model.weights) != len(schema):
        raise ValueError('Constrain-and-scale schema mismatch')
    original = model.compute_loss
    had_override = 'compute_loss' in model.__dict__
    override = model.__dict__.get('compute_loss')

    def compute_loss(*args, **kwargs):
        normal = original(*args, **kwargs)
        regularization = (tf.add_n([tf.cast(v, normal.dtype) for v in model.losses])
                          if model.losses else tf.zeros_like(normal))
        anomaly = tf.add_n([tf.reduce_sum(tf.square(v - b)) for v, b in pairs]) / dimension
        return alpha * (normal - regularization) + (1 - alpha) * tf.cast(anomaly, normal.dtype) + regularization

    model.compute_loss = compute_loss
    try:
        yield
    finally:
        if had_override:
            model.compute_loss = override
        else:
            del model.compute_loss
        pairs.clear()


def save_client_diagnostic(poisoning_dir, server_round, cid, record, submitted):
    directory = Path(poisoning_dir).parent / 'model_replacement' / f'round_{server_round:03d}'
    directory.mkdir(parents=True, exist_ok=True)
    save_json(directory / f'{cid}.json', {
        **record, 'round': server_round, 'client_id': cid,
        'submission_state': 'prepared; server receipt tracked in round summary',
        'submitted_weights_sha256': weights_digest(submitted),
    })


def log_round_diagnostics(logger, poisoning_dir, server_round, replies, selected_replies):
    """Server-only, after selection. Diagnostics cannot affect selection/weights."""
    if poisoning_dir is None:
        return
    root = Path(poisoning_dir).parent
    summary = json.loads((root / 'assignment_and_poisoning.json').read_text())
    if 'model_replacement' not in summary.get('parameters', {}).get('enabled_types', []):
        return
    successful = {m.content['client-info']['client-id']: m for m in replies if not m.has_error()}
    selected = {m.content['client-info']['client-id'] for m in selected_replies if not m.has_error()}
    total = sum(m.content['metrics']['num-examples'] for m in successful.values())
    accepted_total = sum(m.content['metrics']['num-examples'] for cid, m in successful.items() if cid in selected)
    records = []
    for cid, attack in summary['assignments'].items():
        if attack != 'model_replacement':
            continue
        row = {'client_id': cid, 'successful_reply': cid in successful, 'selected': cid in selected}
        path = root / 'model_replacement' / f'round_{server_round:03d}' / f'{cid}.json'
        if cid in successful:
            data = json.loads(path.read_text())  # Missing diagnostics are an explicit failure.
            weights = successful[cid].content['arrays'].to_numpy_ndarrays()
            if data['round'] != server_round or data['client_id'] != cid or data['submitted_weights_sha256'] != weights_digest(weights):
                raise ValueError('Model replacement diagnostic does not match submitted weights')
            row.update(data)
            row['submission_state'] = 'received'
            count = successful[cid].content['metrics']['num-examples']
            row['successful_reply_weight'] = count / total if total else None
            row['accepted_weight'] = count / accepted_total if cid in selected and accepted_total else 0.0
            logger.client_event(cid, 'model_replacement', server_round, {
                'active': int(data['active']), 'scale': data['scale'],
                'pre_scale_trainable_l2': data['pre_scale_trainable_l2'],
                'submitted_trainable_l2': data['submitted_trainable_l2'],
                'selected': int(cid in selected),
            })
        else:
            row['submission_state'] = 'no successful reply (not sampled or failed)'
        records.append(row)
    save_json(root / 'model_replacement' / f'round_{server_round:03d}.json', {
        'round': server_round, 'estimated_population_weight': summary['model_replacement']['estimated_weight'],
        'successful_training_samples': total, 'accepted_training_samples': accepted_total,
        'clients': records,
    })
    lines = [f'Round {server_round} model replacement (ground truth; not defense inputs):']
    for row in records:
        lines.append(f"  {row['client_id']}: {row['submission_state']}; active={row.get('active', 'unknown')}; "
                     f"scale={row.get('scale', 'unavailable')}; selected={row['selected']}")
    logging.getLogger('flwr').info('\n%s', '\n'.join(lines))
