"""Four-component round analysis; original model replies remain aggregation inputs."""

from .defense_config import DefenseConfig
from .direction import measure_direction
from .magnitude import measure_magnitude
from .trust_score import combine_scores
from .validation import score_validation
from .vectors import model_vectors


class CombinedDefense:
    def __init__(self, evaluator, config=None):
        self.evaluator = evaluator
        self.config = config or DefenseConfig()

    def analyze(self, clients, clustering, reference, schema):
        ids = clustering.client_ids
        if set(ids) != set(clients) or len(ids) != len(clients):
            raise ValueError("Clustering and model client IDs are misaligned")
        head = clustering.effective_config["head_client"]
        cfg = self.config
        residuals, updates = model_vectors(clients, reference, schema)
        direction = measure_direction(residuals, head, cfg.direction_threshold_degrees, cfg.direction_epsilon)
        magnitude = measure_magnitude(updates, head, cfg.magnitude_threshold,
                                      cfg.magnitude_boundary_rtol, cfg.magnitude_boundary_atol)
        validation = score_validation(self.evaluator.evaluate(clients, head), head,
                                      accuracy_tolerance=cfg.validation_accuracy_tolerance)
        fields = (clustering.labels, clustering.membership_strengths, clustering.pca_coordinates,
                  clustering.head_distances, clustering.clustering_scores)
        if any(len(values) != len(ids) for values in fields):
            raise ValueError("Clustering result dimensions are misaligned")
        clusters = {cid: {"label": clustering.labels[i], "membership_strength": clustering.membership_strengths[i],
                          "pca_coordinates": clustering.pca_coordinates[i], "head_distance": clustering.head_distances[i],
                          "noise_distance_threshold": clustering.effective_config["noise_distance_threshold"],
                          "score": clustering.clustering_scores[i]} for i, cid in enumerate(ids)}
        components = {"clustering": clusters, "direction": direction, "magnitude": magnitude, "validation": validation}
        trust = combine_scores({name: {cid: v["score"] for cid, v in values.items()}
                                for name, values in components.items()}, cfg.weights, cfg.trust_threshold)
        if not any(v["selected"] for v in trust.values()):
            raise ValueError("Combined trust selected no clients; aggregation aborted")
        return {"round": clustering.round_id, "head_client": head,
                "validation_source": self.evaluator.source,
                "clients": {cid: {**trust[cid], **{name: values[cid] for name, values in components.items()}}
                            for cid in ids}}
