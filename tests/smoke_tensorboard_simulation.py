"""Manual Ray smoke test: two clients, one round, tiny CIFAR-shaped fixture.

Run from the project root with: python -m tests.smoke_tensorboard_simulation
(or PYTHONPATH=. python tests/smoke_tensorboard_simulation.py).
Uses the real app handlers, ResNet-56, FedAvg, and TensorBoard integration.
All test artifacts are stored under /tmp and removed on normal completion.
"""

import io
import copy
import json
import os
from pathlib import Path
import pickle
import signal
import tarfile
import tempfile

import numpy as np
import ray
from flwr.clientapp import ClientApp
from flwr.serverapp import ServerApp
from flwr.simulation import run_simulation
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from fl_defense.client_app import train, evaluate
from fl_defense.server_app import main


def smoke():
    with tempfile.TemporaryDirectory(prefix="fl-tb-smoke-") as directory:
        root = Path(directory)
        folder = root / "datasets" / "cifar10"
        folder.mkdir(parents=True)
        rng = np.random.default_rng(42)
        with tarfile.open(folder / "cifar-10-python.tar.gz", "w:gz") as archive:
            for batch in range(1, 6):
                content = pickle.dumps({b"data": rng.integers(0, 256, (4, 3072), dtype=np.uint8),
                                        b"labels": [(batch * 4 + i) % 10 for i in range(4)]})
                member = tarfile.TarInfo(f"cifar-10-batches-py/data_batch_{batch}")
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
        rounds = int(os.environ.get("FL_SMOKE_ROUNDS", "1"))
        config = {"num-clients": 2, "num-server-rounds": rounds, "local-epochs": 1,
                  "batch-size": 8, "learning-rate": 0.001, "fraction-train": 1.0,
                  "verbose": False, "save-model": False, "partition-method": "iid",
                  "partition-seed": 42, "results-dir": str(root / "results"),
                  "dataset-dir": str(root / "datasets")}
        server, client = ServerApp(), ClientApp()

        def test_context(context):
            # The public simulation helper has no run-config argument. Supply a
            # copy to app handlers without mutating Flower's immutable run config.
            copied = copy.copy(context)
            copied.run_config = dict(config)
            return copied

        @server.main()
        def server_main(grid, context):
            main(grid, test_context(context))

        @client.train()
        def client_train(msg, context):
            return train(msg, test_context(context))

        @client.evaluate()
        def client_evaluate(msg, context):
            return evaluate(msg, test_context(context))

        try:
            run_simulation(server_app=server, client_app=client, num_supernodes=2,
                           backend_config={"init_args": {"num_cpus": 1, "include_dashboard": False,
                                                          "object_store_memory": 100 * 1024 * 1024},
                                           "client_resources": {"num_cpus": 1, "num_gpus": 0}})
        finally:
            ray.shutdown()
        runs = list((root / "results" / "tensorboard").iterdir())
        assert len(runs) == 1, runs
        reader = EventAccumulator(str(runs[0]), size_guidance={"tensors": 0}).Reload()
        for cid in ("client_0", "client_1"):
            for tag in (f"dataset/{cid}/summary", f"dataset/{cid}/images",
                        f"training/{cid}/loss", f"validation/{cid}/accuracy"):
                assert reader.Tensors(tag), tag
        assert reader.Tensors("server/global_accuracy")
        assert json.loads((runs[0] / "status.json").read_text())["status"] == "completed"
        assert len(reader.Tensors("server/global_accuracy")) == rounds
        for round_number in range(1, rounds + 1):
            for cid in ("client_0", "client_1"):
                assert (runs[0] / "models" / f"round_{round_number:03d}_{cid}.npz").exists()
            assert (runs[0] / "models" / f"round_{round_number:03d}_global_before.npz").exists()
            assert (runs[0] / "models" / f"round_{round_number:03d}_global_after.npz").exists()
        assert (runs[0] / "models" / "weight_schema.json").exists()
        assert (runs[0] / "models" / "model_architecture.json").exists()
        print(f"PASS: real Ray simulation, 2 sequential clients, {rounds} rounds, ResNet-56, central TensorBoard events")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(143)))
    smoke()
