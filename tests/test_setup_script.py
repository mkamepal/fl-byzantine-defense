"""Check installer control flow without apt, downloads, or driver changes."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "setup.sh"


@unittest.skipUnless(shutil.which("bash"), "Requires Bash")
class SetupScriptTests(unittest.TestCase):
    def shell(self, code, *args):
        return subprocess.run(
            ["bash", "-c", 'source "$1"; shift\n' + code, "test-setup", str(SCRIPT), *map(str, args)],
            cwd=ROOT, text=True, capture_output=True, timeout=60,
        )

    def test_help_and_bad_arguments_do_not_install(self):
        result = self.shell('sudo() { echo UNEXPECTED_INSTALL; exit 99; }; main --help')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Usage:", result.stdout)
        self.assertNotIn("UNEXPECTED_INSTALL", result.stdout)
        for arguments in ("--unknown", "--env-name", "--env-name base", "--env-name ../escape", "--conda-prefix relative"):
            with self.subTest(arguments=arguments):
                result = self.shell('sudo() { exit 99; }; main ' + arguments)
                self.assertEqual(result.returncode, 1, result.stderr)

    def test_working_driver_is_retained(self):
        result = self.shell('''
nvidia-smi() { return 0; }
sudo() { exit 99; }
INSTALL_DRIVER=1
ensure_driver
''')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_driver_requires_flag_and_wsl_never_installs_driver(self):
        for wsl, flag, message in ((0, 0, "--install-driver"), (1, 1, "Windows host driver")):
            with self.subTest(wsl=wsl):
                result = self.shell('''
nvidia-smi() { return 1; }
sudo() { exit 99; }
IS_WSL="$1"
INSTALL_DRIVER="$2"
ensure_driver
''', wsl, flag)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn(message, result.stderr)

    def test_driver_install_stops_for_reboot(self):
        result = self.shell('''
nvidia-smi() { return 1; }
lspci() { echo '01:00.0 VGA compatible controller: NVIDIA Corporation'; }
sudo() { printf 'MOCK sudo %s\\n' "$*"; }
INSTALL_DRIVER=1
ensure_driver
echo UNEXPECTED_CONTINUE
''')
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("ubuntu-drivers install", result.stdout)
        self.assertIn("Reboot manually", result.stdout)
        self.assertNotIn("UNEXPECTED_CONTINUE", result.stdout)

    def test_no_gpu_does_not_install_driver(self):
        result = self.shell('''
nvidia-smi() { return 1; }
lspci() { echo 'VGA compatible controller: Intel'; }
sudo() { exit 99; }
INSTALL_DRIVER=1
ensure_driver
''')
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("No NVIDIA GPU", result.stderr)

    def test_bad_installer_checksum_stops_before_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.shell('''
CONDA_ROOT="$1/miniforge"
curl() {
    local destination="${@: -1}"
    if [[ "$destination" == *.sha256 ]]; then
        printf '%064d  %s\\n' 0 "$(basename "${destination%.sha256}")" > "$destination"
    else
        printf 'echo UNEXPECTED_INSTALLER_EXECUTED\\n' > "$destination"
    fi
}
ensure_conda
''', folder)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FAILED", result.stdout)
            self.assertNotIn("UNEXPECTED_INSTALLER_EXECUTED", result.stdout)
            self.assertFalse((Path(folder) / "miniforge").exists())

    def test_local_config_preserves_source_and_rerun_edits(self):
        for clients, rounds in ((2, 2), (5, 3)):
            with self.subTest(clients=clients), tempfile.TemporaryDirectory(prefix="setup with spaces ") as folder:
                root = Path(folder)
                source = root / "run_config.json"
                config = {
                    "simulation": {"num-supernodes": clients, "client-resources-num-gpus": 1.0},
                    "training": {"num-server-rounds": rounds, "clustering-enabled": True,
                                 "clustering-n-components": 1 if clients == 2 else 2,
                                 "attacks-enabled": False, "learning-rate": .005},
                }
                source.write_text(json.dumps(config))
                before = source.read_bytes()
                shutil.copyfile(ROOT / "pyproject.toml", root / "pyproject.toml")
                code = 'PROJECT_DIR="$1"; PYTHON="$2"; DOWNLOAD_DATA=0; prepare_local_config'
                result = self.shell(code, root, sys.executable)
                self.assertEqual(result.returncode, 0, result.stderr)
                local = root / "run_config.local.json"
                actual = json.loads(local.read_text())
                self.assertEqual(actual["training"].pop("dataset-dir"), str(root / "datasets"))
                self.assertEqual(actual, config)
                self.assertEqual(source.read_bytes(), before)
                modified = json.loads(local.read_text())
                modified["training"]["num-server-rounds"] = rounds + 1
                local.write_text(json.dumps(modified))
                local_before = local.read_bytes()
                result = self.shell(code, root, sys.executable)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(local.read_bytes(), local_before)
                self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
