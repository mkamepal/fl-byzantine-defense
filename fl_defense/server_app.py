"""fl_defense: A Flower / TensorFlow app."""

import json
import time

import tensorflow as tf

from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from fl_defense.aggregation.logged_fedavg import LoggedFedAvg

from fl_defense.models.resnet56 import load_model
from fl_defense.data.partition_summary import prepare_partition_summary
from fl_defense.utils.tensorboard import ExperimentLogger, log_dataset
from fl_defense.utils.weight_archive import WeightArchive
from fl_defense.detection.clustering import ClusteringConfig, ClusteringDefense
from fl_defense.detection.parameters import model_schema, schema_json
from fl_defense.detection.combined import CombinedDefense
from fl_defense.detection.defense_config import DefenseConfig
from fl_defense.detection.validation import TrustedEvaluator
from fl_defense.data.cifar10 import load_client_split
from fl_defense.data.partitioning import PartitionSettings
from fl_defense.utils.experiment_config import validate_experiment_config
from fl_defense.attacks.datasets import prepare_attack_datasets

# Create the ServerApp
app = ServerApp()


def wait_for_clients(grid, expected, timeout=60):
    """The simulation starts ServerApp before all virtual nodes have registered."""
    deadline = time.monotonic() + timeout
    while True:
        connected = len(list(grid.get_node_ids()))
        if connected == expected:
            return
        if connected > expected or time.monotonic() >= deadline:
            raise ValueError(f"Expected {expected} clients for the partition summary, found {connected}")
        time.sleep(0.1)


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    validate_experiment_config(context.run_config)
    # Load config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]

    num_clients = context.run_config["num-clients"]
    clustering_config = None
    if context.run_config.get("clustering-enabled", False):
        clustering_config = ClusteringConfig.from_config(context.run_config)
    wait_for_clients(grid, num_clients)

    # Load initial model
    with tf.device("/CPU:0"):
        model = load_model()
    arrays = ArrayRecord(model.get_weights())

    # One writer in this server process owns all experiment summaries.
    with ExperimentLogger(context.run_config["results-dir"], context.run_id,
                          context.run_config, model.name) as logger:
        print(f"TensorBoard experiment: {logger.run_dir}", flush=True)
        rows = prepare_partition_summary(
            context.run_config, num_clients, context.run_id,
            output_path=logger.run_dir / "partition_summary.csv",
        )
        log_dataset(logger, context.run_config, rows)
        poisoning_dir = prepare_attack_datasets(logger, context.run_config)
        archive = (WeightArchive(logger.run_dir, num_clients, model)
                   if context.run_config.get("save-client-weights", True) else None)
        defense = None
        combined = None
        if clustering_config is not None:
            defense = ClusteringDefense(model_schema(model), clustering_config)
            logger.save_json("clustering_config.json", defense.config.effective())
            logger.save_json("clustering_parameter_schema.json", json.loads(schema_json(defense.schema)))
            if context.run_config.get("trust-enabled", True):
                # Same cached raw data, partition plan and clean split as clients.
                # Never load the attack training overlays or the final test batch.
                x, y = load_client_split(0, num_clients, "validation", context.run_config.get("dataset-dir"),
                                         PartitionSettings.from_config(context.run_config))
                evaluator = TrustedEvaluator(model, x, y, context.run_config["batch-size"],
                                             "client_0 clean local holdout (cached CIFAR-10 training partition; not final test)")
                combined = CombinedDefense(evaluator, DefenseConfig.from_config(context.run_config))
                logger.save_json("defense_config.json", {**combined.config.effective(),
                                                        "head_client": clustering_config.head_client,
                                                        "validation_source": evaluator.source})
        strategy = LoggedFedAvg(logger, num_clients, weight_archive=archive,
                               clustering_defense=defense, fraction_train=fraction_train,
                               poisoning_dir=poisoning_dir, combined_defense=combined)
        result = strategy.start(grid=grid, initial_arrays=arrays, num_rounds=num_rounds)
        logger.save_json("metrics.json", {
            "training": {r: dict(m) for r, m in result.train_metrics_clientapp.items()},
            "validation": {r: dict(m) for r, m in result.evaluate_metrics_clientapp.items()},
            "server_evaluation": {r: dict(m) for r, m in result.evaluate_metrics_serverapp.items()},
        })
        if context.run_config["save-model"]:
            with tf.device("/CPU:0"):
                model.set_weights(result.arrays.to_numpy_ndarrays())
                model_dir = logger.run_dir / "models"
                model_dir.mkdir(exist_ok=True)
                model.save(model_dir / "final_model.keras")
