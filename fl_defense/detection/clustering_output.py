"""Compact experiment artifacts and a headless view of the actual clustering Z."""

from importlib.metadata import version
import logging

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from fl_defense.utils.weight_archive import _publish, save_json


def log_cluster_summary(result):
    """Emit one server log record per round so client rows stay together."""
    head = result.effective_config["head_client"]
    lines = [
        f"========== ROUND {result.round_id}: CLIENT CLUSTERS ==========",
        f"{'Client ID':<16} {'Cluster':<16} {'Membership strength':>20} Reference",
        "-" * 74,
    ]
    for i, cid in enumerate(result.client_ids):
        label = result.labels[i]
        cluster = "Noise (-1)" if label == -1 else f"Cluster {label}"
        strength = f"{result.membership_strengths[i]:.2%}"
        reference = "trusted reference" if cid == head else ""
        lines.append(f"{cid:<16} {cluster:<16} {strength:>20} {reference}")
    lines.extend([
        "-" * 74,
        f"Clusters: {result.num_clusters} | Noise clients: {result.num_noise_clients}",
        "Cluster IDs are local to this round; noise does not mean malicious.",
        "Membership strength is HDBSCAN probabilities_, not a benign probability.",
    ])
    logging.getLogger("flwr").info("\n%s", "\n".join(lines))


def plot_clustering(result):
    """Standalone Agg Figure; never registers an interactive pyplot figure."""
    figure = Figure(figsize=(10, 7))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    coordinates = np.asarray(result.pca_coordinates)
    y = coordinates[:, 1] if coordinates.shape[1] >= 2 else np.zeros(len(coordinates))
    labels = np.asarray(result.labels)
    for label in sorted(set(result.labels)):
        mask = labels == label
        axis.scatter(coordinates[mask, 0], y[mask], marker="x" if label == -1 else "o",
                     s=45 + 65 * np.asarray(result.membership_strengths)[mask],
                     label="Noise" if label == -1 else f"Cluster {label}")
    head = result.client_ids.index(result.effective_config["head_client"])
    axis.scatter(coordinates[head, 0], y[head], s=250, facecolors="none", edgecolors="black",
                 linewidths=2, marker="s", label="Trusted reference")
    for i, cid in enumerate(result.client_ids):
        axis.annotate(cid, (coordinates[i, 0], y[i]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2" if coordinates.shape[1] >= 2 else "Display only (zero); clustering uses PC1")
    projection = " | 2D projection of clustering space" if coordinates.shape[1] > 2 else ""
    axis.set_title(f"Round {result.round_id} | clusters={result.num_clusters} | noise={result.num_noise_clients}\n"
                   f"Retained variance ({coordinates.shape[1]} PCs): {result.cumulative_retained_variance[-1]:.2%}{projection}")
    axis.legend()
    figure.tight_layout()
    return figure


def persist_clustering(logger, result, artifacts):
    directory = logger.run_dir / "clustering"
    directory.mkdir(exist_ok=True)
    stem = f"round_{result.round_id:03d}"
    record = result.to_dict()
    record["library_versions"] = {name: version(name) for name in ("numpy", "scikit-learn", "hdbscan")}
    save_json(directory / f"{stem}.json", record)
    if artifacts:
        _publish(directory / f"{stem}_analysis.npz", lambda stream: np.savez_compressed(stream, **artifacts))
    figure = plot_clustering(result)
    try:
        _publish(directory / f"{stem}.png", lambda stream: figure.savefig(stream, format="png"))
        logger.figure("detection/clustering/coordinates", figure, result.round_id)
    finally:
        # Agg Figure is unregistered (no pyplot manager/window to close).
        figure.clear()
    for i, cid in enumerate(result.client_ids):
        logger.client_event(cid, "detection/clustering", result.round_id, {
            "score": result.clustering_scores[i], "label": result.labels[i],
            "membership_strength": result.membership_strengths[i],
            "head_distance": result.head_distances[i], "raw_l2_norm": result.raw_l2_norms[i],
            "zero_update": int(result.zero_updates[i]),
        })
    logger.server_event(result.round_id, {
        "num_clusters": result.num_clusters, "num_noise_clients": result.num_noise_clients,
        "retained_variance": result.cumulative_retained_variance[-1],
        "selected_clients": sum(result.clustering_scores),
        "rejected_clients": len(result.client_ids) - sum(result.clustering_scores),
    }, category="detection/clustering")
    for i, ratio in enumerate(result.explained_variance_ratio):
        logger.scalar(f"detection/clustering/pc{i + 1}_variance_ratio", ratio, result.round_id)
    logger.flush()
    log_cluster_summary(result)
