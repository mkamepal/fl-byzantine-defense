"""Round-local direction clustering of trainable Keras model updates.

Failure policy: raise and abort aggregation. Zero updates remain zero rows,
are explicitly flagged, and use the same scoring rules as other clients.
"""

from dataclasses import asdict, dataclass, fields
import math

import numpy as np

from fl_defense.detection.parameters import client_number, flatten_update, schema_json, validate_weights


@dataclass(frozen=True)
class ClusteringConfig:
    n_components: int = 3
    random_state: int = 42
    metric: str = "euclidean"
    min_cluster_size: int = 1
    min_samples: int = 1
    cluster_selection_method: str = "eom"
    cluster_selection_epsilon: float = 0.2
    allow_single_cluster: bool = True
    head_client: str = "client_0"
    noise_distance_threshold: float = 1.5
    save_update_vectors: bool = False
    save_pca_loadings: bool = False

    def __post_init__(self):
        for key, minimum in (("n_components", 1), ("random_state", 0),
                             ("min_cluster_size", 2), ("min_samples", 1)):
            value = getattr(self, key)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        if self.random_state >= 2**32:
            raise ValueError("random_state must be below 2**32")
        for key in ("allow_single_cluster", "save_update_vectors", "save_pca_loadings"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be a boolean")
        if self.metric != "euclidean":
            raise ValueError("This defense requires the euclidean metric")
        if self.cluster_selection_method not in ("eom", "leaf"):
            raise ValueError("cluster_selection_method must be eom or leaf")
        for key, maximum in (("noise_distance_threshold", 2), ("cluster_selection_epsilon", math.inf)):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= maximum:
                raise ValueError(f"Invalid {key}: expected [0, {maximum}]")
        client_number(self.head_client)

    @classmethod
    def from_config(cls, config):
        return cls(**{f.name: config["clustering-" + f.name.replace("_", "-")]
                      for f in fields(cls) if "clustering-" + f.name.replace("_", "-") in config})

    def effective(self):
        return {**asdict(self), "whiten": False, "update_convention": "client_model_minus_round_start_global",
                "analysis_dtype": "float32", "failure_policy": "abort_experiment",
                "zero_update_policy": "keep_zero_row_and_flag", "selection": "clustering_score == 1"}

    def validate_population(self, num_clients, dimension=None):
        """Check known sample/feature limits before spending time on training."""
        if self.n_components > num_clients - 1 or (dimension is not None and self.n_components > dimension):
            raise ValueError(
                "PCA requires 1 <= n_components <= min(N - 1, D): "
                f"n_components={self.n_components}, N={num_clients}, "
                f"D={dimension if dimension is not None else 'not yet known'}. "
                f"Request at least {self.n_components + 1} participating clients with successful updates "
                f"and at least {self.n_components} trainable scalar parameters. "
                "Check num-supernodes/num-clients and fraction-train; "
                "PCA components will not be reduced automatically."
            )
        if self.min_cluster_size > num_clients or self.min_samples >= num_clients:
            raise ValueError(
                f"Insufficient clients for configured HDBSCAN density parameters: N={num_clients}, "
                f"min_cluster_size={self.min_cluster_size}, min_samples={self.min_samples}"
            )


@dataclass
class ClusteringResult:
    round_id: int
    client_ids: list[str]
    vector_dimension: int
    raw_l2_norms: list[float]
    zero_updates: list[bool]
    pca_coordinates: list[list[float]]
    explained_variance_ratio: list[float]
    cumulative_retained_variance: list[float]
    labels: list[int]
    membership_strengths: list[float]
    head_distances: list[float]
    clustering_scores: list[int]
    num_clusters: int
    num_noise_clients: int
    effective_config: dict

    def to_dict(self):
        return asdict(self)


def normalize_updates(matrix):
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Updates must be a finite N by D matrix")
    # Accumulate in float64 to avoid overflow/underflow in float32 squared norms.
    norms = np.sqrt(np.einsum("ij,ij->i", matrix, matrix, dtype=np.float64))
    zero = norms == 0
    normalized = np.zeros(matrix.shape, dtype=np.float32)
    np.divide(matrix, norms[:, None], out=normalized, where=~zero[:, None])
    return normalized, norms, zero


def binary_scores(client_ids, coordinates, labels, head_client="client_0", threshold=1.5):
    if not math.isfinite(threshold) or not 0 <= threshold <= 2:
        raise ValueError("Noise threshold must be in [0, 2]")
    if head_client not in client_ids:
        raise ValueError(f"Trusted reference {head_client} is absent")
    if len(set(client_ids)) != len(client_ids) or len(labels) != len(client_ids):
        raise ValueError("Duplicate clients or label count mismatch")
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[0] != len(client_ids) or not np.isfinite(coordinates).all():
        raise ValueError("Invalid PCA coordinates")
    head = client_ids.index(head_client)
    distances = np.linalg.norm(coordinates.astype(np.float64) - coordinates[head], axis=1)
    scores = []
    for i, cid in enumerate(client_ids):
        if cid == head_client:
            scores.append(1)
        elif labels[i] == -1 or labels[head] == -1:
            scores.append(int(distances[i] <= threshold))
        else:
            scores.append(int(labels[i] == labels[head]))
    return scores, distances


class ClusteringDefense:
    """One immutable baseline and a fresh PCA/HDBSCAN fit per round.

    Clients supply full model arrays and a canonical schema string. The schema
    checks names, ordering, shapes, trainability and dtypes against the server.
    The metadata is an alignment check, not Byzantine-proof attestation.
    """

    def __init__(self, schema, config=None):
        self.schema = tuple(schema)
        if not self.schema or len({p.name for p in self.schema}) != len(self.schema):
            raise ValueError("Parameter names must be nonempty and unique")
        if not any(p.trainable for p in self.schema):
            raise ValueError("No trainable parameters")
        self.config = config or ClusteringConfig()
        self.reference = None
        self.round_id = None

    def begin_round(self, round_id, weights):
        if self.reference is not None:
            raise ValueError("Previous defense round has not been released")
        validate_weights(weights, self.schema)
        self.reference = tuple(np.array(w, copy=True) for w in weights)
        for value in self.reference:
            value.setflags(write=False)
        self.round_id = round_id

    def release(self):
        self.reference = None
        self.round_id = None

    def analyze(self, round_id, clients):
        """clients: {stable_id: (full_weight_list, canonical_schema_json)}.

        Returns compact result plus opt-in large arrays, never aggregation data.
        """
        if self.reference is None or self.round_id != round_id:
            raise ValueError("Missing matching round-start reference")
        ids = sorted(clients, key=client_number)
        if self.config.head_client not in ids:
            raise ValueError(f"Trusted reference {self.config.head_client} is absent")
        expected = schema_json(self.schema)
        vectors = []
        for cid in ids:
            weights, received_schema = clients[cid]
            if received_schema != expected:
                raise ValueError(f"Parameter names/order/schema mismatch for {cid}")
            vectors.append(flatten_update(weights, self.reference, self.schema))
        matrix = np.stack(vectors).astype(np.float32, copy=False)
        n, dimension = matrix.shape
        self.config.validate_population(n, dimension)
        normalized, norms, zero = normalize_updates(matrix)
        if np.allclose(normalized, normalized[0], rtol=0, atol=1e-7):
            raise ValueError("Degenerate identical normalized updates (within float32 tolerance)")

        # Lazy imports leave disabled-defense runs independent of these imports.
        import hdbscan
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.config.n_components, random_state=self.config.random_state, whiten=False)
        coordinates = pca.fit_transform(normalized)
        ratios = pca.explained_variance_ratio_
        if not np.isfinite(coordinates).all() or not np.isfinite(ratios).all():
            raise ValueError("Non-finite PCA output")
        clusterer = hdbscan.HDBSCAN(
            metric=self.config.metric, min_cluster_size=self.config.min_cluster_size,
            min_samples=self.config.min_samples,
            cluster_selection_method=self.config.cluster_selection_method,
            cluster_selection_epsilon=float(self.config.cluster_selection_epsilon),
            allow_single_cluster=self.config.allow_single_cluster,
        )
        # No post-PCA normalization: this exact Z also drives distances and plots.
        labels = clusterer.fit_predict(coordinates)
        strengths = clusterer.probabilities_
        if (len(labels) != n or len(strengths) != n or not np.isfinite(strengths).all()
                or np.any(strengths < 0) or np.any(strengths > 1) or np.any(labels < -1)):
            raise ValueError("Invalid HDBSCAN output")
        scores, distances = binary_scores(ids, coordinates, labels, self.config.head_client,
                                          self.config.noise_distance_threshold)
        result = ClusteringResult(
            round_id, ids, dimension, norms.tolist(), zero.tolist(), coordinates.tolist(),
            ratios.tolist(), np.cumsum(ratios, dtype=np.float64).tolist(), labels.tolist(),
            strengths.tolist(), distances.tolist(), scores,
            len(set(labels.tolist()) - {-1}), int(np.count_nonzero(labels == -1)), self.config.effective(),
        )
        artifacts = {}
        if self.config.save_update_vectors:
            artifacts["update_vectors"] = matrix
        if self.config.save_pca_loadings:
            artifacts.update(pca_components=pca.components_, pca_mean=pca.mean_,
                             pca_singular_values=pca.singular_values_)
        return result, artifacts
