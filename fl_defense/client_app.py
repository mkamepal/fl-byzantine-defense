"""fl_defense: A Flower / TensorFlow app."""

from contextlib import nullcontext
import math

import keras
import tensorflow as tf
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl_defense.data.cifar10 import load_client_split
from fl_defense.data.client_identity import client_id, client_role
from fl_defense.data.partitioning import PartitionSettings
from fl_defense.models.resnet56 import load_model
from fl_defense.utils.training_logs import ClientTrainingLogger
from fl_defense.utils.client_resources import release_client_resources
from fl_defense.utils.worker_memory import record_client_memory
from fl_defense.detection.parameters import model_schema, schema_json
from fl_defense.attacks.config import AttackConfig
from fl_defense.attacks.datasets import load_training_data
from fl_defense.attacks.evaluation import backdoor_success_counts
from fl_defense.attacks.model_replacement import (
    load_assignment, choose_scale, replace_model, constrained_loss,
    trainable_norm, save_client_diagnostic,
)

# Flower ClientApp
app = ClientApp()


def client_info(partition_id, local_epochs=None, model=None, server_round=None):
    """Non-aggregated metadata; no images or datasets travel in these records."""
    info = {"client-id": client_id(partition_id), "partition-id": partition_id,
            "role": client_role(partition_id)}
    if local_epochs is not None:
        info["local-epochs"] = local_epochs
    if model is not None:
        info["weight-schema"] = schema_json(model_schema(model))
        info["weight-convention"] = "full-model"
        info["server-round"] = server_round
    return ConfigRecord(info)


@app.train()
@release_client_resources
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Reset local Tensorflow state
    keras.backend.clear_session()
    record_client_memory("after_initial_cleanup")

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    if num_partitions != context.run_config["num-clients"]:
        raise ValueError("Client partition count does not match the server summary")
    attack_settings = AttackConfig.from_config(context.run_config)
    poisoning_dir = msg.content['config'].get('poisoning-data-dir')
    if attack_settings.enabled and not poisoning_dir:
        raise ValueError('Enabled attacks require experiment training overlays from the server')
    if not attack_settings.enabled and poisoning_dir is not None:
        raise ValueError('Disabled attacks must not receive poisoning overlays')
    server_round = msg.content["config"]["server-round"]
    cid = client_id(partition_id)
    replacement_client = False
    replacement_active = False
    summary = None
    if attack_settings.enabled and 'model_replacement' in attack_settings.enabled_types:
        summary = load_assignment(poisoning_dir, attack_settings, num_partitions,
                                  context.run_config.get('clustering-head-client', 'client_0'))
        replacement_client = summary['assignments'].get(cid) == 'model_replacement'
        replacement_active = replacement_client and attack_settings.replacement_active(server_round)
    x_train, y_train = load_training_data(
        partition_id, num_partitions, context.run_config.get("dataset-dir"),
        PartitionSettings.from_config(context.run_config), poisoning_dir,
        apply_overlay=not replacement_client or replacement_active,
    )
    record_client_memory("after_data_load")

    # Load the model
    learning_rate = context.run_config["learning-rate"]
    if replacement_active and attack_settings.replacement_learning_rate:
        learning_rate = attack_settings.replacement_learning_rate
    model = load_model(learning_rate)
    # TF 2.20's custom-gradient registry retains discarded training graphs.
    # Eager training avoids that leak when Ray actors build successive models;
    # disabling XLA alone does not disable ordinary tf.function tracing.
    model.jit_compile = False
    model.run_eagerly = True
    incoming = msg.content["arrays"].to_numpy_ndarrays()
    reference = [a.copy() for a in incoming] if replacement_client else None
    model.set_weights(incoming)
    del incoming
    schema = model_schema(model) if replacement_client else None
    record_client_memory("after_model_load")
    epochs = context.run_config["local-epochs"]
    if replacement_active and attack_settings.replacement_epochs:
        epochs = attack_settings.replacement_epochs
    if summary is not None:
        metadata = next(c for c in summary['clients'] if c['client_id'] == cid)
        if len(x_train) != metadata['training_samples']:
            raise ValueError('Attack manifest training sample count mismatch')
    batch_size = context.run_config["batch-size"]
    callbacks = []
    if context.run_config.get("verbose", False):
        server_round = msg.content["config"]["server-round"]
        callbacks.append(ClientTrainingLogger(partition_id, server_round, epochs))

    # Train the model
    record_client_memory("before_train")
    loss_scope = (constrained_loss(model, reference, schema, attack_settings.replacement_alpha)
                  if replacement_active and attack_settings.replacement_variant == 'constrain_and_scale'
                  else nullcontext())
    with tf.device("/GPU:0"), loss_scope:
        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=callbacks,
        )
    record_client_memory("after_train")

    # Get training metrics
    train_loss = history.history["loss"][-1] if "loss" in history.history else None
    train_acc = (
        history.history["accuracy"][-1] if "accuracy" in history.history else None
    )

    # Pack and send the model weights and metrics as a message
    submitted = model.get_weights()
    if replacement_client:
        if any(value is not None and not math.isfinite(float(value)) for value in (train_loss, train_acc)):
            raise ValueError('Non-finite model replacement training metrics')
        scale = choose_scale(attack_settings, summary) if replacement_active else 1.0
        if replacement_active:
            submitted, diagnostic = replace_model(submitted, reference, schema, scale)
        else:
            norm = trainable_norm(submitted, reference, schema)
            diagnostic = {'pre_scale_trainable_l2': norm, 'submitted_trainable_l2': norm,
                          'zero_update': norm == 0, 'tensor_policy': 'normal clean training; no scaling'}
        save_client_diagnostic(poisoning_dir, server_round, cid, {
            **diagnostic, 'attack': 'model_replacement', 'active': replacement_active,
            'variant': attack_settings.replacement_variant, 'scale': scale,
            'scale_mode': attack_settings.replacement_scale_mode,
            'scale_assumption': summary['model_replacement']['scale_assumption'],
            'population_estimated_scale': summary['model_replacement']['estimated_scale'],
            'learning_rate': learning_rate, 'local_epochs': len(history.epoch),
            'training_samples': len(x_train),
            'poisoned_training_samples': metadata['poisoned_samples'] if replacement_active else 0,
            'training_metrics_source': 'local training before submission scaling',
            'anomaly_penalty': ('mean_squared_trainable_distance_from_round_start'
                                if replacement_active and attack_settings.replacement_variant == 'constrain_and_scale'
                                else None),
            'classification_weight': attack_settings.replacement_alpha,
        }, submitted)
    model_record = ArrayRecord(submitted)
    metrics = {"num-examples": len(x_train)}
    if train_loss is not None:
        metrics["train_loss"] = train_loss
    if train_acc is not None:
        metrics["train_acc"] = train_acc
    content = RecordDict({"arrays": model_record, "metrics": MetricRecord(metrics),
                          "client-info": client_info(partition_id, len(history.epoch), model,
                                                     msg.content["config"]["server-round"])})
    return Message(content=content, reply_to=msg)


@app.evaluate()
@release_client_resources
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Reset local Tensorflow state
    keras.backend.clear_session()
    record_client_memory("after_initial_cleanup")

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    if num_partitions != context.run_config["num-clients"]:
        raise ValueError("Client partition count does not match the server summary")
    x_validation, y_validation = load_client_split(
        partition_id, num_partitions, "validation", context.run_config.get("dataset-dir"),
        PartitionSettings.from_config(context.run_config),
    )
    record_client_memory("after_data_load")

    # Load the model
    model = load_model(context.run_config["learning-rate"])
    model.jit_compile = False
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())
    record_client_memory("after_model_load")

    # Evaluate the model
    record_client_memory("before_evaluate")
    eval_loss, eval_acc = model.evaluate(x_validation, y_validation, verbose=0)
    record_client_memory("after_evaluate")

    # Pack and send the model weights and metrics as a message
    metrics = {
        "eval_acc": eval_acc,
        "eval_loss": eval_loss,
        "num-examples": len(x_validation),
    }
    content = RecordDict({"metrics": MetricRecord(metrics), "client-info": client_info(partition_id)})
    attack_settings = AttackConfig.from_config(context.run_config)
    if attack_settings.enabled and attack_settings.evaluate_asr and attack_settings.has_backdoor:
        record_client_memory("before_asr")
        content['backdoor-asr'] = ConfigRecord(backdoor_success_counts(
            model, x_validation, y_validation, attack_settings.target_label,
            attack_settings.trigger_value, context.run_config['batch-size'],
        ))
        record_client_memory("after_asr")
    return Message(content=content, reply_to=msg)
