"""One server-owned TensorBoard writer and persistent directory per experiment."""

from datetime import datetime, timezone
import io
import json
import subprocess
from pathlib import Path
import uuid

import numpy as np
import tensorflow as tf


def code_revision():
    """Report only an available commit; bundled apps/checkouts may lack history."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True, timeout=5, check=True)
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


class ExperimentLogger:
    """Use as a context manager in ServerApp; never instantiate inside clients."""

    def __init__(self, results_dir, flower_run_id, config, model_name):
        root = Path(results_dir).expanduser()
        if not root.is_absolute():
            raise ValueError("results-dir must be absolute for Flower subprocesses")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.experiment_id = f"run_{flower_run_id}_{stamp}_{uuid.uuid4().hex}"
        self.run_dir = root / "tensorboard" / self.experiment_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.config = {
            "experiment_id": self.experiment_id,
            "flower_run_id": flower_run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "aggregation": ("combined-trust-filtered FedAvg" if config.get("trust-enabled", True)
                            else "clustering-filtered FedAvg") if config.get("clustering-enabled", False) else "FedAvg",
            "num_clients": config["num-clients"],
            "num_rounds": config["num-server-rounds"],
            "partition_seed": config.get("partition-seed", 42),
            "partition_method": config.get("partition-method", "iid"),
            "model_random_seed": None,  # Training has not configured a model RNG seed.
            "global_metric_source": "sample-weighted client local validation",
            "run_config": dict(config),
            "tensorflow_version": tf.__version__,
            "client_jit_compile": False,
            "client_train_run_eagerly": True,
            "client_evaluate_run_eagerly": False,
            "code_revision": code_revision(),
        }
        self.save_json("config.json", self.config)
        self.writer = tf.summary.create_file_writer(str(self.run_dir), flush_millis=1000)
        self.closed = False

    def save_json(self, name, value):
        with (self.run_dir / name).open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")

    def scalar(self, tag, value, step):
        with self.writer.as_default(), tf.device("/CPU:0"):
            tf.summary.scalar(tag, value, step=step)

    def text(self, tag, value, step=0):
        with self.writer.as_default(), tf.device("/CPU:0"):
            tf.summary.text(tag, value, step=step)

    def images(self, tag, images, step=0, max_outputs=5):
        """Accept NHWC uint8 images or float images in [0, 1]."""
        with self.writer.as_default(), tf.device("/CPU:0"):
            tf.summary.image(tag, images, step=step, max_outputs=max_outputs)

    def histogram(self, tag, values, step):
        with self.writer.as_default(), tf.device("/CPU:0"):
            tf.summary.histogram(tag, values, step=step)

    def figure(self, tag, figure, step=0):
        """Encode a Matplotlib Figure; ownership of the figure stays with the caller."""
        with io.BytesIO() as buffer:
            figure.savefig(buffer, format="png", bbox_inches="tight")
            with tf.device("/CPU:0"):
                image = tf.image.decode_png(buffer.getvalue(), channels=4)
                self.images(tag, image[None, ...], step=step, max_outputs=1)

    def client_event(self, client_id, category, step, metrics=None, message=None):
        for name, value in (metrics or {}).items():
            self.scalar(f"{category}/{client_id}/{name}", value, step)
        if message is not None:
            self.text(f"{category}/{client_id}/events", message, step)

    def server_event(self, step, metrics=None, message=None, category="server"):
        for name, value in (metrics or {}).items():
            self.scalar(f"{category}/{name}", value, step)
        if message is not None:
            self.text(f"{category}/events", message, step)

    def flush(self):
        self.writer.flush()

    def close(self):
        if not self.closed:
            try:
                self.flush()
            finally:
                self.writer.close()
                self.closed = True

    def __enter__(self):
        try:
            self.text("experiment/config", "```json\n" + json.dumps(self.config, indent=2) + "\n```")
            self.text("experiment/status", "Running")
            self.text("server/evaluation_source", "Global metrics are sample-weighted local validation results, not official test-set evaluation.")
            self.flush()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        status = "completed" if exc_type is None else "interrupted" if issubclass(exc_type, (KeyboardInterrupt, SystemExit)) else "failed"
        try:
            self.save_json("status.json", {"status": status, "error": str(exc_value) if exc_value else None})
            self.text("experiment/status", status)
        finally:
            self.close()


def log_dataset(logger, config, rows):
    """Reuse the cached raw CIFAR images and the exact cached training indices."""
    from matplotlib.figure import Figure
    from fl_defense.data.cifar10 import _load_local_training_data, _partition_plan, resolve_dataset_dir
    from fl_defense.data.partitioning import PartitionSettings
    from fl_defense.data.partition_summary import CLASS_NAMES, partition_statistics
    from fl_defense.utils.image_previews import log_labeled_previews

    settings = PartitionSettings.from_config(config)
    root = resolve_dataset_dir(config.get("dataset-dir"))
    images, labels = _load_local_training_data(root)
    plan = _partition_plan(root, len(rows), settings)
    # Validate both the supplied report and actual assignment/class counts.
    if rows != partition_statistics(labels, plan, settings, rows[0]["run_id"]):
        raise ValueError("TensorBoard dataset report differs from the actual partition plan")
    for row, partition in zip(rows, plan):
        cid = row["client_id"]
        table = [
            f"**Client ID:** {cid} ({row['role']})  ",
            f"**Partition ID:** {row['partition_id']}  ",
            f"**Total Images:** {row['total_images']}  ",
            f"**Training:** {row['train_total_images']}  ",
            f"**Local validation:** {row['validation_total_images']}", "",
            "| Class | Total | Training | Validation |", "|---|---:|---:|---:|",
        ]
        for name in CLASS_NAMES:
            key = name.lower()
            table.append(f"| {name} | {row[key]} | {row['train_' + key]} | {row['validation_' + key]} |")
        if config.get('attacks-enabled', False):
            table.append('\nThese class counts describe clean partitions before poisoning; see attacks/ for actual training diagnostics.')
        logger.text(f"dataset/{cid}/summary", "\n".join(table))
        indices = partition.train_indices[:5]
        log_labeled_previews(logger, f"dataset/{cid}/images", images[indices], labels[indices],
                             indices, cid, 'clean_training')
        label_table = ["Left-to-right training images:", "", "| Position | Source row | Class |", "|---:|---:|---|"]
        if config.get('attacks-enabled', False):
            label_table[0] = 'Left-to-right clean training images before poisoning; see attacks/ for poisoned training previews:'
        for position, index in enumerate(indices, 1):
            label_table.append(f"| {position} | {index} | {CLASS_NAMES[labels[index]]} |")
        logger.text(f"dataset/{cid}/image_labels", "\n".join(label_table))

    for prefix, tag in (("", "class_distribution"), ("train_", "training_distribution"), ("validation_", "validation_distribution")):
        counts = np.array([[row[prefix + name.lower()] for name in CLASS_NAMES] for row in rows])
        fig = Figure(figsize=(13, max(4, len(rows) * 0.42)), layout="constrained")
        ax = fig.subplots()
        heatmap = ax.imshow(counts, aspect="auto", cmap="Blues")
        ax.set_xticks(range(10), CLASS_NAMES, rotation=35, ha="right")
        ax.set_yticks(range(len(rows)), [row["client_id"] for row in rows])
        ax.set_xlabel("CIFAR-10 class")
        ax.set_ylabel("Client ID")
        ax.set_title(tag.replace("_", " ").title())
        for (i, j), count in np.ndenumerate(counts):
            ax.text(j, i, str(count), ha="center", va="center", fontsize=8,
                    color="white" if count > counts.max() * 0.6 else "black")
        fig.colorbar(heatmap, ax=ax, label="Number of images")
        logger.figure(f"dataset/{tag}", fig)
        fig.clear()
    logger.flush()
