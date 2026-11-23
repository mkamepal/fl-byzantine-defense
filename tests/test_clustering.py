"""Synthetic checks of the pipeline, not evidence of attack-detection effectiveness."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from fl_defense.detection.clustering import (
    ClusteringConfig, ClusteringDefense, binary_scores, normalize_updates,
)
from fl_defense.detection.parameters import Parameter, flatten_update, model_schema, schema_json
from fl_defense.detection.clustering_output import plot_clustering


SCHEMA = (Parameter("dense/kernel", (3,), True), Parameter("dense/bias", (1,), True),
          Parameter("bn/moving_mean", (1,), False), Parameter("bn/gamma", (1,), True))


def baseline():
    return [np.full(p.shape, 10, np.float32) for p in SCHEMA]


def synthetic_clients():
    """Eight synthetic clients in two opposed directional groups."""
    clients = {}
    for i in (10, 2, 0, 14, 1, 11, 3, 12):
        sign = 1 if i < 10 else -1
        weights = baseline()
        weights[0] += np.array([sign * (1 + i / 20), (i % 4 - 1) * .01, i * .001], np.float32)
        weights[1] += sign * .2
        weights[2] += 1000 + i  # Non-trainable buffer must be excluded.
        weights[3] += sign * .1
        clients[f"client_{i}"] = (weights, schema_json(SCHEMA))
    return clients


class ClusteringTests(unittest.TestCase):
    def defense(self, **kwargs):
        defense = ClusteringDefense(SCHEMA, ClusteringConfig(**kwargs))
        defense.begin_round(1, baseline())
        self.addCleanup(defense.release)
        return defense

    def test_round_reference_immutable_and_trainable_order(self):
        start = baseline()
        defense = ClusteringDefense(SCHEMA)
        defense.begin_round(4, start)
        start[0][:] = 999  # Does not change W_t.
        weights = baseline()
        weights[0] += [1, 2, 3]
        weights[1] += 4
        weights[2] += 9000
        weights[3] += 5
        delta = flatten_update(weights, defense.reference, SCHEMA)
        np.testing.assert_array_equal(delta, [1, 2, 3, 4, 5])
        self.assertEqual(delta.dtype, np.float32)
        self.assertFalse(defense.reference[0].flags.writeable)
        with self.assertRaisesRegex(ValueError, "matching round"):
            defense.analyze(5, synthetic_clients())
        with self.assertRaisesRegex(ValueError, "released"):
            defense.begin_round(5, start)
        defense.release()
        new_start = baseline()
        new_start[0] += 1
        defense.begin_round(5, new_start)
        np.testing.assert_array_equal(flatten_update(weights, defense.reference, SCHEMA), [0, 1, 2, 4, 5])

    def test_parameter_alignment_failures(self):
        for weights, reason in ((baseline()[:-1], "count"),
                                ([np.zeros(2, np.float32), *baseline()[1:]], "shape"),
                                ([np.zeros(3, np.float64), *baseline()[1:]], "dtype")):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                flatten_update(weights, baseline(), SCHEMA)
        for changed in (SCHEMA[::-1], (replace(SCHEMA[0], name="wrong"), *SCHEMA[1:])):
            clients = synthetic_clients()
            weights, _ = clients["client_0"]
            clients["client_0"] = weights, schema_json(changed)
            with self.assertRaisesRegex(ValueError, "names/order"):
                self.defense().analyze(1, clients)
        with self.assertRaisesRegex(ValueError, "unique"):
            ClusteringDefense((SCHEMA[0], SCHEMA[0]))

    def test_row_normalization_zero_and_extreme_norms(self):
        matrix = np.array([[3, 4, 0], [0, 0, 0], [1e30, 1e30, 0], [1e-30, 0, 0]], np.float32)
        normalized, norms, zero = normalize_updates(matrix)
        np.testing.assert_allclose(np.linalg.norm(normalized[~zero], axis=1), 1, atol=1e-7)
        np.testing.assert_array_equal(normalized[1], [0, 0, 0])
        np.testing.assert_array_equal(zero, [False, True, False, False])
        self.assertEqual(norms[0], 5)
        self.assertEqual(normalized.dtype, np.float32)
        clients = synthetic_clients()
        clients["client_2"] = baseline(), schema_json(SCHEMA)
        result, _ = self.defense().analyze(1, clients)
        row = result.client_ids.index("client_2")
        self.assertTrue(result.zero_updates[row])
        self.assertEqual(result.raw_l2_norms[row], 0)

    def test_synthetic_end_to_end_numeric_ids_variance_and_artifacts(self):
        result, artifacts = self.defense(save_update_vectors=True, save_pca_loadings=True).analyze(1, synthetic_clients())
        self.assertEqual(result.client_ids, ["client_0", "client_1", "client_2", "client_3", "client_10", "client_11", "client_12", "client_14"])
        self.assertEqual(result.vector_dimension, 5)
        self.assertEqual(np.shape(result.pca_coordinates), (8, 2))
        np.testing.assert_allclose(result.cumulative_retained_variance, np.cumsum(result.explained_variance_ratio))
        self.assertLessEqual(result.cumulative_retained_variance[-1], 1 + 1e-6)
        self.assertEqual(result.clustering_scores[:4], [1] * 4)
        self.assertEqual(result.clustering_scores[4:], [0] * 4)
        self.assertEqual(result.num_clusters, 2)  # Never assert arbitrary numeric cluster IDs.
        self.assertEqual(artifacts["update_vectors"].shape, (8, 5))
        self.assertEqual(artifacts["pca_components"].shape, (2, 5))
        figure = plot_clustering(result)
        try:
            axis = figure.axes[0]
            points = np.vstack([c.get_offsets() for c in axis.collections[:-1]])
            self.assertEqual(len(points), 8)
            self.assertEqual({t.get_text() for t in axis.texts}, set(result.client_ids))
            self.assertEqual({tuple(p) for p in points}, {tuple(p) for p in result.pca_coordinates})
        finally:
            figure.clear()
        self.assertEqual(self.defense().analyze(1, synthetic_clients())[1], {})

    def test_pca_receives_row_normalized_updates_and_hdbscan_exact_z(self):
        from sklearn.decomposition import PCA
        real_fit = PCA.fit_transform
        captured = {}
        def fit(pca, x):
            captured["input"] = x.copy()
            captured["pca"] = pca
            captured["z"] = real_fit(pca, x)
            return captured["z"]
        clusterer = Mock()
        clusterer.fit_predict.return_value = np.array([7] * 4 + [3] * 4)
        clusterer.probabilities_ = np.ones(8)
        with patch.object(PCA, "fit_transform", fit), patch("hdbscan.HDBSCAN", return_value=clusterer) as factory:
            result, _ = self.defense().analyze(1, synthetic_clients())
        np.testing.assert_allclose(np.linalg.norm(captured["input"], axis=1), 1, atol=1e-7)
        self.assertIs(clusterer.fit_predict.call_args.args[0], captured["z"])
        np.testing.assert_array_equal(result.pca_coordinates, captured["z"])
        self.assertFalse(np.allclose(np.linalg.norm(captured["z"], axis=1), 1))
        self.assertFalse(captured["pca"].whiten)
        self.assertEqual(captured["pca"].random_state, 42)
        factory.assert_called_once_with(metric="euclidean", min_cluster_size=2, min_samples=1,
                                       cluster_selection_method="eom", cluster_selection_epsilon=.2,
                                       allow_single_cluster=True)

    def test_exact_binary_rules_and_noise_head(self):
        ids = [f"client_{i}" for i in range(6)]
        coordinates = np.array([[0, 0], [1.49, 0], [1.5, 0], [1.51, 0], [1.9, 0], [.1, 0]])
        # Same valid cluster selected even when distant; different valid cluster rejected even when close.
        scores, distances = binary_scores(ids, coordinates, [8, -1, -1, -1, 8, 3])
        self.assertEqual(scores, [1, 1, 1, 0, 1, 0])
        self.assertEqual(distances[2], 1.5)
        self.assertEqual(binary_scores(ids, coordinates, [-1, 2, -1, 2, 2, 3])[0], [1, 1, 1, 0, 0, 1])
        self.assertEqual(binary_scores(ids, coordinates, [8, -1, -1, -1, 8, 3], "client_5")[0][-1], 1)
        with self.assertRaisesRegex(ValueError, "absent"):
            binary_scores(ids[1:], coordinates[1:], [1] * 5)

    def test_missing_insufficient_degenerate_and_nonfinite(self):
        clients = synthetic_clients()
        del clients["client_0"]
        with self.assertRaisesRegex(ValueError, "absent"):
            self.defense().analyze(1, clients)
        for clients in ({"client_0": synthetic_clients()["client_0"]},
                        {f"client_{i}": (baseline(), schema_json(SCHEMA)) for i in range(4)}):
            with self.assertRaises(ValueError):
                self.defense().analyze(1, clients)
        for value in (float("nan"), float("inf")):
            clients = synthetic_clients()
            clients["client_0"][0][0][0] = value
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                self.defense().analyze(1, clients)
        with self.assertRaisesRegex(ValueError, "PCA requires"):
            self.defense(n_components=6).analyze(1, synthetic_clients())
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            self.defense(min_cluster_size=9).analyze(1, synthetic_clients())
        # Magnitude-only differences become identical after normalization.
        schema = (Parameter("w", (2,), True),)
        defense = ClusteringDefense(schema)
        defense.begin_round(1, [np.zeros(2, np.float32)])
        with self.assertRaisesRegex(ValueError, "Degenerate"):
            defense.analyze(1, {f"client_{i}": ([np.array([i + 1, 0], np.float32)], schema_json(schema)) for i in range(3)})

    def test_configuration_and_component_counts(self):
        for kwargs in ({"noise_distance_threshold": -1}, {"noise_distance_threshold": 2.01},
                       {"n_components": .95}, {"n_components": True}, {"min_samples": 0},
                       {"cluster_selection_epsilon": float("nan")}, {"metric": "cosine"},
                       {"allow_single_cluster": "true"}, {"head_client": "client_01"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ClusteringConfig(**kwargs)
        config = ClusteringConfig.from_config({"clustering-noise-distance-threshold": 2.0,
                                               "clustering-min-cluster-size": 3})
        self.assertEqual(config.noise_distance_threshold, 2)
        self.assertEqual(config.min_cluster_size, 3)
        for components in (1, 3):
            result, _ = self.defense(n_components=components).analyze(1, synthetic_clients())
            self.assertEqual(np.shape(result.pca_coordinates), (8, components))
            figure = plot_clustering(result)
            if components == 3:
                self.assertIn("2D projection", figure.axes[0].get_title())
            figure.clear()

    def test_population_preflight_and_three_client_pipeline(self):
        config = ClusteringConfig()
        with self.assertRaisesRegex(ValueError, r"n_components=2, N=2.*at least 3"):
            config.validate_population(2)
        config.validate_population(3, 5)
        with self.assertRaisesRegex(ValueError, "D=1"):
            config.validate_population(3, 1)
        ClusteringConfig(n_components=1).validate_population(2, 5)
        clients = synthetic_clients()
        result, _ = self.defense().analyze(1, {cid: clients[cid] for cid in ('client_0', 'client_1', 'client_10')})
        self.assertEqual(np.shape(result.pca_coordinates), (3, 2))


if __name__ == "__main__":
    unittest.main()
