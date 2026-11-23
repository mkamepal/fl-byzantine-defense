"""Exercise launch overrides through Flower's real configuration parser."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flwr.common.config import parse_config_args

import run
from fl_defense.utils.experiment_config import validate_experiment_config


class RunConfigTests(unittest.TestCase):
    def test_preflight_merges_actual_pyproject_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            defaults = {'num-server-rounds': 2, 'trust-weight-clustering': .1,
                        'trust-weight-direction': .2, 'trust-weight-magnitude': .3,
                        'trust-weight-validation': .4, 'magnitude-threshold': .4}
            (project / 'pyproject.toml').write_text('[tool.flwr.app.config]\n' + run.encode_toml(defaults))
            source = {'simulation': {'num-supernodes': 3}, 'training': {'magnitude-threshold': .2}}
            prepared = run.prepare_run_config(source, project)
            self.assertEqual(prepared['training']['trust-weight-clustering'], .1)
            self.assertEqual(prepared['training']['magnitude-threshold'], .2)
            self.assertEqual(prepared['training']['num-server-rounds'], 2)
            self.assertEqual(source['training'], {'magnitude-threshold': .2})

    def test_combined_defense_overrides_and_preflight(self):
        config = {"simulation": {"num-supernodes": 5}, "training": {
            "num-server-rounds": 2, "clustering-enabled": True, "trust-enabled": True,
            "direction-threshold-degrees": 85.0, "magnitude-threshold": .3,
            "validation-accuracy-tolerance": .08, "trust-threshold": .6,
            "trust-weight-clustering": .1, "trust-weight-direction": .2,
            "trust-weight-magnitude": .3, "trust-weight-validation": .4}}
        original = json.dumps(config)
        prepared = run.prepare_run_config(config, Path('/tmp'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'training.toml'
            path.write_text(run.encode_toml(prepared['training']))
            self.assertEqual(parse_config_args([str(path)]), prepared['training'])
        self.assertEqual(json.dumps(config), original)
        for key, invalid in (("magnitude-threshold", 1.1), ("validation-accuracy-tolerance", -.1),
                             ("trust-weight-clustering", .2), ("trust-enabled", "true"),
                             ("direction-epsilon", 0), ("magnitude-reference", "client_mean")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                run.prepare_run_config({**config, 'training': {**config['training'], key: invalid}}, Path('/tmp'))

    def test_dynamic_overrides_are_isolated_and_explicit(self):
        config = {"simulation": {"num-supernodes": 15}, "training": {
            "num-server-rounds": 3, "num-clients": 99, "clustering-enabled": True}}
        original = json.dumps(config)
        for clients, rounds, components in ((2, 1, 1), (3, 2, 2), (5, 10, 2), (15, 100, 2)):
            with self.subTest(clients=clients, rounds=rounds):
                resolved = run.prepare_run_config(config, Path('/tmp'), clients, rounds, components)
                self.assertEqual(resolved['simulation']['num-supernodes'], clients)
                self.assertEqual(resolved['training']['num-clients'], clients)
                self.assertEqual(resolved['training']['num-server-rounds'], rounds)
                self.assertEqual(resolved['training']['clustering-n-components'], components)
        self.assertEqual(json.dumps(config), original)
        with self.assertRaisesRegex(ValueError, "at least 3"):
            run.prepare_run_config(config, Path('/tmp'), clients=2)
        for overrides in ({'clients': 0}, {'rounds': -1}, {'rounds': True}, {'clients': 2.5}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                run.prepare_run_config(config, Path('/tmp'), **overrides)
        validate_experiment_config({'num-clients': 1, 'num-server-rounds': 1, 'clustering-enabled': False})
        with self.assertRaisesRegex(ValueError, 'requests 2 per round'):
            validate_experiment_config({'num-clients': 15, 'num-server-rounds': 1,
                                        'fraction-train': .1, 'clustering-enabled': True})
        with self.assertRaisesRegex(ValueError, 'outside'):
            validate_experiment_config({'num-clients': 3, 'num-server-rounds': 1,
                                        'clustering-head-client': 'client_9', 'clustering-enabled': True})

    def test_cli_overrides_preserve_file_and_validate_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'experiment.json'
            path.write_text(json.dumps({'simulation': {'num-supernodes': 15},
                                        'training': {'num-server-rounds': 3, 'clustering-enabled': True}}))
            before = path.read_bytes()
            def launch(command, **kwargs):
                simulation = parse_config_args([command[command.index('--federation-config') + 1]])
                training = parse_config_args([command[command.index('--run-config') + 1]])
                self.assertEqual(simulation['num-supernodes'], 2)
                self.assertEqual(training['num-clients'], 2)
                self.assertEqual(training['num-server-rounds'], 4)
                self.assertEqual(training['clustering-n-components'], 1)
                return 0
            with patch.object(run, 'run_command', side_effect=launch):
                self.assertEqual(run.main(['--config', str(path), '--clients', '2', '--rounds', '4',
                                           '--pca-components', '1']), 0)
            with patch.object(run, 'run_command') as launch, self.assertRaises(SystemExit):
                run.main(['--config', str(path), '--clients', '2'])
            launch.assert_not_called()
            self.assertEqual(path.read_bytes(), before)

    def test_scalar_and_nested_json_round_trip(self):
        values = {
            "enabled": True, "count": 15, "rate": 0.001,
            "simulation-config-json": json.dumps({"num-supernodes": 15, "label": "Alice's \"run\""}),
            "path": '/tmp/Alice\'s "results"/café',
            "message": "first\nsecond\t\\end", "literal.key": "value",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(run.encode_toml(values), encoding="utf-8")
            self.assertEqual(parse_config_args([str(path)]), values)

    def test_main_passes_readable_files_and_removes_them(self):
        paths = []

        def launch(command, **kwargs):
            for option in ("--federation-config", "--run-config"):
                paths.append(Path(command[command.index(option) + 1]))
            simulation, training = [parse_config_args([str(path)]) for path in paths]
            self.assertEqual(json.loads(training["simulation-config-json"]), simulation)
            self.assertEqual(training["num-clients"], simulation["num-supernodes"])
            self.assertTrue(Path(training["results-dir"]).is_absolute())
            return 7

        with patch.object(run, "run_command", side_effect=launch):
            self.assertEqual(run.main([]), 7)
        self.assertTrue(all(not path.exists() for path in paths))

    def test_rejects_unsupported_values(self):
        for value in (None, [], {}, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                run.encode_toml({"invalid": value})


if __name__ == "__main__":
    unittest.main()
