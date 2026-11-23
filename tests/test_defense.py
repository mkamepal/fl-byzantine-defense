"""Arithmetic and isolated inference tests for the four-component defense."""

from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from fl_defense.detection.clustering import ClusteringDefense
from fl_defense.detection.defense_config import DefenseConfig
from fl_defense.detection.direction import measure_direction
from fl_defense.detection.magnitude import measure_magnitude
from fl_defense.detection.parameters import model_schema, schema_json
from fl_defense.detection.trust_score import combine_scores
from fl_defense.detection.validation import TrustedEvaluator, score_validation
from fl_defense.detection.vectors import model_vectors
from test_clustering import SCHEMA, baseline


def tiny_model():
    import keras
    x = keras.Input((2,), name="input")
    z = keras.layers.BatchNormalization(name="bn")(x)
    z = keras.layers.Dropout(.95, name="dropout")(z)
    y = keras.layers.Dense(2, activation="softmax", kernel_regularizer=keras.regularizers.l2(.01), name="classifier")(z)
    model = keras.Model(x, y, name="synthetic_classifier")
    model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    weights = model.get_weights()
    weights[-2][:] = [[1, -1], [0, 0]]
    weights[-1][:] = 0
    model.set_weights(weights)
    return model


class DefenseMeasurementTests(unittest.TestCase):
    def test_trust_cutoff_point_51_is_inclusive_and_rejects_point_50(self):
        weights = dict(clustering=.5, direction=.01, magnitude=.24, validation=.25)
        components = {name: {'client_0': int(name in ('clustering', 'direction')),
                             'client_2': int(name == 'clustering')} for name in weights}
        result = combine_scores(components, weights, threshold=.51)
        self.assertEqual(result['client_0'], {'trust_score': .51, 'selected': True})
        self.assertEqual(result['client_2'], {'trust_score': .5, 'selected': False})

    def test_immutable_baseline_unweighted_mean_and_partial_ids(self):
        before = baseline()
        defense = ClusteringDefense(SCHEMA)
        defense.begin_round(1, before)
        before[0][:] = 1000
        clients = {}
        for cid, change in (("client_10", 9), ("client_0", 1), ("client_2", 2)):
            weights = baseline()
            weights[0] += change
            weights[1] += change * 2
            weights[2] += 9999  # buffer excluded
            weights[3] += change * 3
            clients[cid] = weights, schema_json(SCHEMA)
        residuals, updates = model_vectors(clients, defense.reference, SCHEMA)
        self.assertEqual(list(residuals), ["client_0", "client_2", "client_10"])
        unit = np.array([1, 1, 1, 2, 3])
        np.testing.assert_array_equal(updates["client_0"], unit)
        np.testing.assert_array_equal(residuals["client_0"], -3 * unit)
        np.testing.assert_array_equal(sum(residuals.values()), np.zeros(5))
        self.assertFalse(defense.reference[0].flags.writeable)
        self.assertNotEqual(measure_magnitude(updates)["client_0"]["norm"],
                            measure_direction(residuals)["client_0"]["residual_norm"])
        # Shapes, names/order, counts and dtypes cannot silently truncate.
        for kind in ("name", "shape", "count", "dtype", "finite"):
            broken = deepcopy(clients)
            weights, wire = broken["client_2"]
            if kind == "name":
                wire = schema_json((replace(SCHEMA[0], name="other"), *SCHEMA[1:]))
            elif kind == "shape":
                weights[0] = weights[0][:-1]
            elif kind == "count":
                weights.pop()
            elif kind == "dtype":
                weights[0] = weights[0].astype(np.float64)
            else:
                weights[0][0] = np.nan
            broken["client_2"] = weights, wire
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                model_vectors(broken, defense.reference, SCHEMA)

    def test_direction_strict_angles_zero_convention_and_invalid_inputs(self):
        vectors = {"client_0": [1, 0], "client_2": [2, 0], "client_10": [0, 2],
                   "client_11": [-2, 0], "client_12": [0, 0]}
        result = measure_direction(vectors)
        self.assertEqual(result["client_0"]["angle"], 0)
        self.assertAlmostEqual(result["client_2"]["angle"], 0, places=3)
        self.assertEqual(result["client_10"]["angle"], 90)
        self.assertAlmostEqual(result["client_11"]["angle"], 180, places=3)
        self.assertEqual([result[c]["score"] for c in vectors], [1, 1, 0, 0, 0])
        self.assertTrue(result["client_12"]["zero_residual"])
        self.assertEqual(result["client_12"]["angle"], 90)
        vectors["client_0"] = [0, 0]
        result = measure_direction(vectors)
        self.assertEqual(result["client_0"]["angle"], 0)
        self.assertTrue(all(v["head_zero_residual"] for v in result.values()))
        self.assertEqual(result["client_2"]["angle"], 90)
        for invalid in ({"client_2": [1, 0]}, {"client_0": [np.inf, 0]},
                        {"client_0": [1, 0], "client_2": [1]}):
            with self.assertRaises(ValueError):
                measure_direction(invalid)

    def test_magnitude_bounds_tolerance_and_zero_head(self):
        result = measure_magnitude({"client_0": [0, 4], "client_1": [3, 0], "client_2": [0, 5],
                                    "client_3": [5.00004, 0], "client_4": [5.001, 0],
                                    "client_5": [2.99998, 0], "client_6": [2.99, 0]})
        self.assertEqual([v["score"] for v in result.values()], [1, 1, 1, 1, 0, 1, 0])
        self.assertEqual(result["client_0"]["norm"], 4)
        self.assertEqual(result["client_0"]["lower_bound"], 3)
        self.assertEqual(result["client_0"]["upper_bound"], 5)
        zero = measure_magnitude({"client_0": [0], "client_1": [1e-9], "client_2": [1e-7]})
        self.assertEqual([v["score"] for v in zero.values()], [1, 1, 0])
        for threshold in (-.1, 1.01, np.nan):
            with self.assertRaises(ValueError):
                measure_magnitude({"client_0": [1]}, threshold=threshold)

    def test_validation_six_percentage_points_inclusive_and_invalid(self):
        measurements = {"client_0": {"loss": .2, "accuracy": .90, "example_count": 100},
                        "client_2": {"loss": 9., "accuracy": .84, "example_count": 100},
                        "client_10": {"loss": 0., "accuracy": .83999, "example_count": 100}}
        result = score_validation(measurements, accuracy_tolerance=.06)
        self.assertEqual(result["client_2"]["minimum_accuracy"], .84)
        self.assertEqual([v["score"] for v in result.values()], [1, 1, 0])
        with self.assertRaisesRegex(ValueError, "absent"):
            score_validation({"client_2": measurements["client_2"]})
        for field, invalid in (("accuracy", 1.01), ("loss", np.nan), ("example_count", 0), ("example_count", 99)):
            bad = deepcopy(measurements)
            bad["client_2"][field] = invalid
            with self.subTest(field=field), self.assertRaises(ValueError):
                score_validation(bad)

    def test_combination_requires_every_client_component(self):
        weights = DefenseConfig().weights
        components = {name: {"client_0": int(i < 2), "client_10": int(i == 0)}
                      for i, name in enumerate(weights)}
        result = combine_scores(components, weights)
        self.assertEqual(result["client_0"], {"trust_score": .5, "selected": True})
        self.assertFalse(result["client_10"]["selected"])
        custom = dict(zip(weights, [.1, .2, .3, .4]))
        self.assertAlmostEqual(combine_scores(components, custom)["client_0"]["trust_score"], .3)
        for name in weights:
            bad = deepcopy(components)
            del bad[name]["client_10"]
            with self.assertRaisesRegex(ValueError, "misaligned"):
                combine_scores(bad, weights)
        with self.assertRaises(ValueError):
            combine_scores({"clustering": components["clustering"]}, weights)

    def test_configuration_rejects_invalid_and_exposes_effective_settings(self):
        for change in ({"direction_epsilon": 0}, {"direction_threshold_degrees": np.nan},
                       {"magnitude_threshold": 1.1}, {"validation_accuracy_tolerance": -.1},
                       {"trust_weight_validation": .3}, {"trust_weight_direction": -.1},
                       {"direction_reference": "global"}, {"magnitude_reference": "mean"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                DefenseConfig(**change)
        cfg = DefenseConfig.from_config({"magnitude-threshold": .4, "validation-accuracy-tolerance": .1})
        self.assertEqual(cfg.magnitude_threshold, .4)
        self.assertEqual(cfg.validation_accuracy_tolerance, .1)


class TrustedInferenceTests(unittest.TestCase):
    def test_inference_buffers_metrics_and_example_weighting(self):
        import keras
        model = tiny_model()
        x = np.array([[1, 0], [-1, 0], [1, 0]], np.float32)
        y = np.array([0, 1, 1])
        evaluator = TrustedEvaluator(model, x, y, 2, "synthetic clean holdout")
        before = model.get_weights()
        other = deepcopy(before)
        other[-2] *= -1
        other[2][:] = 2  # BN moving mean must be replaced, too.
        wire = schema_json(model_schema(model))
        clients = {"client_10": (other, wire), "client_0": (before, wire)}
        saved = deepcopy(clients)
        with patch.object(evaluator.model, "reset_metrics", wraps=evaluator.model.reset_metrics) as reset:
            result = evaluator.evaluate(clients, "client_0")
        self.assertEqual(reset.call_count, len(clients))
        self.assertEqual(result, evaluator.evaluate(dict(reversed(list(clients.items()))), "client_0"))
        self.assertEqual(result["client_0"]["example_count"], 3)
        self.assertEqual(result["client_0"]["accuracy"], 2 / 3)
        prediction = model(x, training=False)
        expected_loss = np.asarray(keras.losses.sparse_categorical_crossentropy(y, prediction), dtype=np.float64).mean()
        expected_loss += sum(float(v) for v in model.losses)
        self.assertAlmostEqual(result["client_0"]["loss"], expected_loss, places=6)
        self.assertNotEqual(result["client_0"]["accuracy"], result["client_10"]["accuracy"])
        for actual, expected in zip(model.get_weights(), before, strict=True):
            np.testing.assert_array_equal(actual, expected)
        for cid in clients:
            for actual, expected in zip(clients[cid][0], saved[cid][0], strict=True):
                np.testing.assert_array_equal(actual, expected)
        # Evaluator retains the last client's complete state without BN updates.
        for actual, expected in zip(evaluator.model.get_weights(), other, strict=True):
            np.testing.assert_array_equal(actual, expected)
        with self.assertRaisesRegex(ValueError, "empty"):
            TrustedEvaluator(model, x[:0], y[:0], 2, "empty")
        with self.assertRaisesRegex(ValueError, "absent"):
            evaluator.evaluate({"client_10": clients["client_10"]}, "client_0")


if __name__ == "__main__":
    unittest.main()
