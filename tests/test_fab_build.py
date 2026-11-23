"""Application bundles remain independent of growing experiment artifacts."""

from io import BytesIO
from pathlib import Path
import unittest
from zipfile import ZipFile

from flwr.cli.build import build_fab_from_files
from flwr.common.constant import FAB_MAX_SIZE


class FabBuildTests(unittest.TestCase):
    def test_bundle_contains_application_without_experiment_artifacts(self):
        root = Path(__file__).resolve().parents[1]
        sources = {path.relative_to(root).as_posix(): path
                   for path in (root / 'fl_defense').rglob('*.py')}
        files = {**sources, 'pyproject.toml': root / 'pyproject.toml', 'LICENSE': root / 'LICENSE',
                 'results/tensorboard/run_example/models/model_architecture.json': b' ' * (FAB_MAX_SIZE + 1),
                 'datasets/cifar10/metadata.json': b'{}',
                 'experiments/example.json': b'{}',
                 'tests/test_example.py': b'pass',
                 'run_config.json': root / 'run_config.json'}
        bundle, _ = build_fab_from_files(files)
        self.assertLess(len(bundle), FAB_MAX_SIZE)
        with ZipFile(BytesIO(bundle)) as archive:
            self.assertEqual(set(archive.namelist()), {*sources, 'LICENSE', 'pyproject.toml', '.info/CONTENT'})


if __name__ == '__main__':
    unittest.main()
