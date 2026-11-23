"""Run Flower using run_config.json."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import tomllib


def cleanup_process_group(process, timeout=5.0):
    """Stop the run's process group, including workers left after CLI exit."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        process.poll()  # Reap the launcher if it has already exited.
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def run_command(command, **kwargs):
    """Run a command in an isolated POSIX process group and always clean it up."""
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    previous = {}
    process = None
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, interrupted)
        process = subprocess.Popen(command, start_new_session=True, **kwargs)
        return process.wait()
    finally:
        # A second interrupt must not bypass worker cleanup.
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        try:
            if process is not None:
                cleanup_process_group(process)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def encode_toml(settings):
    """Encode scalar overrides as TOML without Flower's inline quote parser."""
    lines = []
    for key, value in settings.items():
        if type(value) not in (bool, int, float, str):
            raise ValueError(f"Configuration value for {key!r} must be bool, int, float, or string")
        # JSON string escapes are compatible with TOML basic strings. Quoted
        # keys also preserve literal dots rather than creating nested tables.
        lines.append(f"{json.dumps(key, ensure_ascii=False)} = {json.dumps(value, ensure_ascii=False, allow_nan=False)}")
    return "\n".join(lines) + "\n"


def prepare_run_config(config, project, clients=None, rounds=None, pca_components=None):
    """Resolve one experiment without modifying the input config or its file."""
    # Keep process-cleanup helpers and --help independent of ML dependencies.
    from fl_defense.utils.experiment_config import validate_experiment_config

    config = deepcopy(config)
    simulation = config["simulation"]
    # Validate exactly the defaults Flower will merge with this invocation,
    # including customized defense weights in pyproject.toml.
    defaults_path = project / "pyproject.toml"
    defaults = (tomllib.loads(defaults_path.read_text(encoding="utf-8"))
                .get("tool", {}).get("flwr", {}).get("app", {}).get("config", {})) if defaults_path.is_file() else {}
    training = config["training"] = {**defaults, **config["training"]}
    if clients is not None:
        simulation["num-supernodes"] = clients
    if rounds is not None:
        training["num-server-rounds"] = rounds
    if pca_components is not None:
        training["clustering-n-components"] = pca_components
    # The simulated population is the single source of truth for partition count.
    training["num-clients"] = simulation["num-supernodes"]
    validate_experiment_config(training)
    training["results-dir"] = str(project.resolve() / "results")
    training["simulation-config-json"] = json.dumps(simulation)
    return config


def main(argv=None):
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run Flower with per-experiment configuration overrides.")
    parser.add_argument("--config", type=Path, default=project / "run_config.json", help="Experiment JSON file")
    parser.add_argument("--clients", type=int, help="Number of simulated clients for this run")
    parser.add_argument("--rounds", type=int, help="Number of federated rounds for this run")
    parser.add_argument("--pca-components", type=int, help="Explicit clustering PCA dimension (no automatic reduction)")
    args = parser.parse_args(argv)
    try:
        config = prepare_run_config(json.loads(args.config.expanduser().read_text()), project,
                                    args.clients, args.rounds, args.pca_components)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    training = config["training"]
    clustering = (f"enabled, PCA components={training.get('clustering-n-components', 2)}"
                  if training.get("clustering-enabled", False) else "disabled")
    selection = ("combined trust" if training.get("trust-enabled", True) else "clustering only") if training.get("clustering-enabled", False) else "FedAvg"
    print(f"Experiment: clients={training['num-clients']}, rounds={training['num-server-rounds']}, "
          f"fraction-train={training.get('fraction-train', 1.0)}, clustering={clustering}, selection={selection}", flush=True)
    # Flower accepts TOML files for both options. Files avoid the inline
    # parser's inability to handle escaped quotes inside JSON-valued strings.
    with tempfile.TemporaryDirectory(prefix="flwr-config-") as directory:
        simulation_path = Path(directory) / "simulation.toml"
        training_path = Path(directory) / "training.toml"
        simulation_path.write_text(encode_toml(config["simulation"]), encoding="utf-8")
        training_path.write_text(encode_toml(config["training"]), encoding="utf-8")
        return run_command(
            [
                "flwr", "run", str(project),
                "--federation-config", str(simulation_path),
                "--run-config", str(training_path),
                "--stream",
            ],
            cwd=project,
            env={**os.environ, "TF_FORCE_GPU_ALLOW_GROWTH": "true"},
        )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
