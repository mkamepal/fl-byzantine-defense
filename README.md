---
tags: [quickstart, vision, fds]
dataset: [CIFAR-10]
framework: [tensorflow]
---

# fake-clients-detection-flower

This introductory example to Flower uses Tensorflow/Keras but deep knowledge of this frameworks is required to run the example. However, it will help you understand how to adapt Flower to your use case.
This example downloads CIFAR-10 directly from the original University of Toronto source and uses Flower Datasets to partition the local data.

## Set up the project

### Project structure

```text
fl-byzantine-defense/
├── fl_defense/
│   ├── __init__.py
│   ├── aggregation/
│   │   ├── __init__.py
│   │   ├── baselines.py
│   │   └── trust_strategy.py
│   ├── attacks/
│   │   ├── __init__.py
│   │   ├── backdoor.py
│   │   └── label_flipping.py
│   ├── client_app.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── cifar10.py
│   │   └── partitioning.py
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── clustering.py
│   │   ├── direction.py
│   │   ├── magnitude.py
│   │   ├── trust_score.py
│   │   └── validation.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn.py
│   │   └── resnet56.py
│   ├── server_app.py
│   └── utils/
│       ├── __init__.py
│       └── metrics.py
├── experiments/
│   ├── iid/
│   ├── non_iid/
│   ├── ablation/
│   ├── adaptive_attacks/
├── results/
├── tests/
├── pyproject.toml
├── run.py
├── run_config.json
└── README.md
```

The project name is `fake-clients-detection-flower`; the Python package is
`fl_defense`. Training uses ResNet-56, IID CIFAR-10 loading, and FedAvg.
`models/cnn.py` preserves the CNN baseline. Both models provide the same
`load_model(learning_rate)` interface. Server initialization, client training,
and evaluation must use the same model.
Clustering, direction, magnitude, and clean trusted-validation checks are available
as a combined defense. Simulated label-flipping and backdoor data poisoning remain
optional attacks. Completed defense rounds automatically export to Excel inside
the current run directory. See [the combined defense guide](docs/combined-defense.md).

### Install dependencies and project

From the repository root, install the dependencies and the local `fl_defense`
package into your activated Python environment:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` mirrors the dependency constraints in `pyproject.toml` and
includes an editable installation of this project. `python -m pip install -e .`
remains an equivalent alternative. These constraints are not a fully pinned
environment lockfile. The launcher uses Python's built-in `tomllib`, so it needs
Python 3.11 or newer, with a version supported by the dependencies.

On another machine, also check the CPU/GPU resources in `run_config.json`. The
supplied configuration requests a GPU; installing Python dependencies alone does
not install an NVIDIA driver or configure a CUDA-enabled TensorFlow environment.

## Run the project

### Run using one configuration file

Edit `run_config.json` to set the number of simulated clients (`num-supernodes`),
CPU/GPU resources per active client, and all training settings. Then run from
your activated Python environment after installing dependencies:

```bash
python run.py
```

Flower's app bundle includes `fl_defense` Python source, `LICENSE`, and the
automatically included `pyproject.toml`, as configured by `tool.flwr.app.fab-include`.
Saved results, model archives, datasets, and logs stay outside the bundle so they
do not consume its 10 MiB size limit. Runtime data and results use the configured
absolute paths; existing experiments remain on disk.

On Linux/macOS, this launcher cleans up its process group when the run finishes,
fails, or receives Ctrl+C/SIGTERM. Workers get five seconds to stop before a forced
stop. Use `python run.py` for this cleanup; direct `flwr run` commands bypass it.
Separately managed services and processes that create their own sessions are
outside this process group. SIGKILL of the launcher cannot trigger cleanup.

Client and round counts come from the chosen configuration and optional CLI overrides.
The supplied resource settings reserve 4 CPUs and 1 GPU per active client, with 2
local epochs. On a single-GPU machine, clients execute one at a time.
The CPU count is Ray's scheduling reservation; TensorFlow manages its own thread
pools. Increasing this allocation does not by itself resolve worker memory growth.
TensorFlow GPU memory growth is enabled for clients; the server uses CPU.
A usable NVIDIA GPU and CUDA-enabled TensorFlow environment are required.
Reused simulation workers disable automatic XLA/JIT compilation for each client
model and clear Keras state after every training/evaluation request, including
failed requests. Only the requested data split is converted to float32; partition
assignments, normalization, batch size, epochs, and optimizer remain unchanged.
Disabling XLA may affect runtime performance. Ray's memory monitor remains enabled.
These settings apply to this run without changing Flower's defaults.

Client training also uses `model.run_eagerly = True` to avoid the reproduced
TensorFlow 2.20 custom-gradient registry leak when successive clients build new
training graphs in a reused worker. Evaluation retains its existing graph mode.
The model, optimizer, data, attacks, and aggregation settings are unchanged;
execution speed and floating-point results may differ. Eager execution can be
slower, so monitor GPU throughput as well as RAM on the next run. Each new run's
`config.json` records `client_train_run_eagerly: true` and
`client_evaluate_run_eagerly: false`. Start a fresh run with `python run.py` to use
the fix; it cannot repair an already running worker. Existing cleanup and RAM
diagnostics remain enabled. This is an application workaround for the
[upstream TensorFlow registry issue](https://github.com/tensorflow/tensorflow/issues/129615).

Worker RAM diagnostics run automatically, independently of training verbosity.
Flower logs `[worker-memory]` records at request start, after initial cleanup,
after data/model loading, before/after training or evaluation (including optional
backdoor ASR evaluation), and before/after final Keras cleanup. Each record includes
the PID, client ID, round, phase, request ID, current RSS in MiB, and changes since
the previous sample and request start. Samples are also appended immediately to
`results/tensorboard/<experiment>/worker_memory/worker_<pid>_<id>.jsonl`, with UTC
timestamps and exact byte values. Compare `after_cleanup` across requests with the
same PID/file to identify retained memory, and stage deltas to locate growth.
No models or datasets are retained by these diagnostics. They measure Linux process
resident RAM, not GPU VRAM or peak usage between samples; a rising RSS alone does
not identify which objects or allocator hold the memory. Unsupported RSS reads are
marked unavailable; file-write failures warn without aborting training. Abruptly
killed workers cannot emit final cleanup samples. Failed handlers may still have
locals held by their exception traceback during cleanup. Existing runs are unchanged;
start the next run normally with `python run.py` to collect these measurements.

The server's partition-summary preparation downloads the original CIFAR-10 archive to
`datasets/cifar10/cifar-10-python.tar.gz` (about 163 MiB). Later requests reuse this file without contacting Hugging Face or calling the
downloader again.
Internet access is required only if the archive is missing. SHA-256 checks
new downloads; an archive you placed in this folder is used as supplied. Downloads are locked so concurrent simulation clients share one file.
The archive contains the official train/test batches; training uses the 50,000
training images, partitioned across clients with a fixed 80/20 local split.

### Client identities and partition summary

Partition `N` always belongs to `client_N`, independent of worker process, node ID,
or execution order. `client_0` is designated `trusted_head` in logs and exported
metadata. This designation does not change sampling, aggregation, or training.

Before the first training round, the server prints every configured client's actual counts
for each CIFAR-10 class and saves `partition_summary.csv` inside the experiment's
TensorBoard directory:
`results/tensorboard/run_<id>_<timestamp>_<uuid>/partition_summary.csv`.
Each run gets its own CSV with the same column order and structure: one row per
configured client. The summary helper requires an explicit output path; it never
falls back to a shared `results/partition_summary.csv`. Any older shared CSV is
left untouched. It checks exact
coverage of the official training rows, no overlap between clients, no overlap
between local training and validation, and matching class totals. Invalid plans
stop the experiment before training. The summary prints even with `verbose: false`.

CSV totals include both local splits. Separate `train_*` and `validation_*` columns
show their counts; the CSV also includes client/partition IDs, role, run ID, method,
seed, Dirichlet parameters, and hashes of the row assignments. The official 10,000
test images in `test_batch` remain reserved for global testing and are never put
in a client partition. Existing client evaluation still uses local validation;
this change does not introduce a global evaluation step.

Set `partition-method` in `run_config.json` to `iid` (default) or `dirichlet`.
`partition-seed` defaults to 42 and controls the local split and Dirichlet sampling.
IID retains the existing Flower shard assignments without adding a shuffle.
For Dirichlet, configure `dirichlet-alpha` (positive scalar),
`dirichlet-min-partition-size` (at least 2), `dirichlet-self-balancing`, and
`dirichlet-shuffle`. Smaller alpha values generally produce greater class imbalance.
The same immutable settings and cached index plan are used for reporting and data
loading. Archive decoding and plan construction are cached once per process/configuration.

`python run.py` copies `simulation.num-supernodes` into the app's `num-clients`
setting and uses the project's absolute `results/` path, including in Flower's
installed app bundle. Direct `flwr run` commands must configure matching numbers
of supernodes and clients. Each experiment preserves its own CSV and TensorBoard
events, so later runs do not replace earlier partition summaries.

The launcher passes settings from `run_config.json` directly to Flower.
With `verbose: true`, clients log training start/end and one complete line per
epoch, labeled with the server round and client partition ID, for example:
`[round=1 client=client_0 role=trusted_head] Epoch 1/2 accuracy=0.4200 loss=1.6500`.
Keras progress bars are disabled to avoid overwriting/interleaving worker output.
Concurrent clients can finish in any order; the labels identify each message.
With `verbose: false`, these client training messages are disabled; Flower's
own server logs remain enabled.
Dataset path handling belongs in the application code.
For deployment on separate machines, each machine needs its own dataset copy
or access to the same shared directory.

### TensorBoard experiments

Install updated dependencies with `pip install -e .`, then launch normally with
`python run.py`. The server automatically creates a unique directory under
`results/tensorboard/run_<Flower-run-ID>_<UTC-timestamp>_<UUID>/`. All components
share this one experiment directory; clients never create TensorBoard writers.
The absolute `results-dir` keeps output persistent when Flower changes working
directories. Set it to a persistent server path if deploying on another machine.

While training is running, open another terminal in this project and run:

```bash
tensorboard --logdir results/tensorboard --port 6006
```

Open **http://localhost:6006**. Use the run selector to view or compare experiments.
The same command reopens previous results after training has stopped.

- **Text:** `dataset/client_N/summary` contains client role, partition, totals,
  and a class-count table. `dataset/client_N/image_labels` gives the labels and
  source row IDs of the displayed images. `experiment/config` records settings.
- **Images:** `dataset/client_N/images` shows the first five training images,
  in their actual partition order, with class names and source row IDs drawn on
  each preview. Pixels are enlarged without smoothing for readability.
  If a configured non-IID training partition has fewer than five images, only its
  available images are shown. `dataset/class_distribution`,
  `dataset/training_distribution`, and `dataset/validation_distribution` show
  annotated heatmaps for all clients.
  Labeled PNGs are also saved under the run's `images/clean_training/client_N/`:
  `preview.png` contains the strip; individual filenames include the source row
  and class name, such as `01_row_123_cat.png`. `labels.json` records the mapping.
  When attacks are enabled, `attacks/client_N/training_images` and
  `images/attack_training/client_N/` show the same first five samples after applying
  the fixed overlays, with current/original class names, attack type, and whether
  each sample was poisoned. These are training-label previews, not predictions.
  To inspect **only samples on which an attack was performed**, use
  `attacks/client_N/poisoned_images` or the saved
  `images/after_attack/client_N/` folder. These show up to five actually poisoned
  samples in training order, chosen from the fixed attack mask. Clean clients
  and clients with zero poisoned samples have no `after_attack` image folder.
  Label flipping changes the label only; backdoors also stamp the image pixels.
- **Scalars:** `training/client_N/{loss,accuracy,local_epochs,num_samples,round}`
  contains the final local epoch's returned metrics. `validation/client_N/...`
  contains the existing client evaluation results and validation sample counts.
  Steps are federated rounds. `server/global_accuracy` and `server/global_loss`
  are sample-weighted client validation metrics; they are **not** official
  CIFAR-10 test-set metrics. `server/participating_clients` counts selected clients,
  and `server/successful_updates` counts successful replies after validation.

Dataset summaries are written once before round 1. Client metrics are collected
centrally and flushed after each training/evaluation phase; they appear during the
experiment, rather than waiting for its completion. They do not stream per batch
or local epoch because clients currently return their results after local training.

Each run preserves `config.json`, `partition_summary.csv`, event files, client
weight archives (below), and `status.json`. Completed runs also save aggregated histories as `metrics.json`.
When `save-model` is true, `models/final_model.keras` is saved inside that run directory.
Configuration includes the model, Flower run ID, partition seed/method, client and
round counts, and app settings. The launcher also records simulation resource
settings via `simulation-config-json`; direct CLI runs must supply that optional
field if they want a copy of their resource settings. The model's random seed is
recorded as unset because the existing training code does not seed initialization.

The server closes the writer on normal completion or Python exceptions and flushes
after each phase. A forced process kill can leave the latest buffered data or final
status unwritten. TensorBoard does not automatically capture console output, full
datasets, model checkpoints, per-batch metrics, or detection results that the
application does not produce. The server explicitly saves the NumPy weight
checkpoints described below, separately from TensorBoard events.

For future server-side research code, reuse `ExperimentLogger.scalar`, `text`,
`images`, `histogram`, `figure`, `client_event`, and `server_event` on the existing
logger. For example, when those values are actually available:

```python
logger.client_event("client_3", "detection", round_number, {"trust_score": score})
logger.server_event(round_number, {"precision": precision}, category="detection")
logger.histogram("detection/update_magnitudes", magnitudes, round_number)
logger.figure("detection/clusters", figure, round_number)
logger.flush()
```

Optional combined defense is described below; it adds clean trusted inference for selection
while retaining existing global evaluation. Run tests with
`python -m unittest discover -s tests -v`. A separate CPU/Ray smoke test uses two
clients and tiny temporary CIFAR-shaped data with the real app and ResNet-56:

```bash
CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 \
  PYTHONPATH=. python tests/smoke_tensorboard_simulation.py
```

### Client weights and weight shifts

By default (`save-client-weights=true`), every experiment saves each successful client's full returned weights **before
FedAvg aggregation**, even when `save-model` is false. Each experiment has a
`models/` folder for snapshots from the entire run. TensorBoard event files remain
in the experiment directory, with no additional writers:

```text
results/tensorboard/run_.../models/
    model_architecture.json
    weight_schema.json
    round_001_global_before.npz
    round_001_client_0.npz
    round_001_client_1.npz
    ...
    round_001_global_after.npz
    round_001_weight_shifts.json
```

This repeats for each round. The JSON report records IDs, roles, sample counts,
failed/missing node replies, and each client's difference from that round's
starting global model: L2 norm, maximum absolute difference, relative L2 norm,
trainable-only L2 norm, and per-tensor measurements. Relative L2 is null when the
reference norm is zero. Non-finite weights are preserved for inspection and flagged;
non-finite measurements are stored as null. Weight-shift diagnostics alone never
exclude clients. When enabled, the clustering defense below filters original models
before aggregation; rejected models are still archived. Large updates alone do not
establish malicious behavior, particularly with non-IID data.

TensorBoard scalars appear under `weight_shift/client_N/`. Full tensors remain in
compressed NumPy files; `weight_schema.json` maps `arr_0`, `arr_1`, etc. to model
variable names, shapes, dtypes, and trainable flags. These files include BatchNorm
moving statistics, but not optimizer state. They can restore model weights using
`model.set_weights(load_weights(path))` with the matching model architecture.

Compare a client against the starting global model, or against another client:

```bash
RUN_DIR="results/tensorboard/run_..."
python -m fl_defense.utils.weight_archive \
  "$RUN_DIR/models/round_001_client_1.npz" "$RUN_DIR/models/round_001_global_before.npz"
python -m fl_defense.utils.weight_archive \
  "$RUN_DIR/models/round_001_client_1.npz" "$RUN_DIR/models/round_001_client_0.npz"
```

Load a snapshot later using its saved architecture, without depending on the
current model factory:

```python
from fl_defense.utils.weight_archive import load_model_snapshot, load_weights

path = "results/tensorboard/run_.../models/round_001_client_0.npz"
weights = load_weights(path)
model = load_model_snapshot(path)
```

The restored model is uncompiled and contains the saved model weights; compile it
if further training/evaluation requires an optimizer and loss. Client optimizer
state is not available in the existing Flower replies. No automatic deletion or
retention limit is applied; snapshots remain until explicitly deleted by you.

The comparison tool prints overall and per-tensor differences as JSON. Checkpoints
preserve tensor order and dtype and are loaded with `allow_pickle=False`. Writes
are atomic and refuse to overwrite existing checkpoints. The archive writer decodes
one client at a time; optional clustering additionally constructs an `N x D` matrix.
Disk usage grows with clients
and rounds. Clients that fail or are not selected have no checkpoint for that round.
Earlier experiments cannot recover individual client weights that were never saved.

### Simulated data-poisoning attacks

Random label flipping and a bottom-right square backdoor are implemented. They are
disabled by default. To run the default mixture, add/change these settings inside
`training` in `run_config.json`:

```json
"attacks-enabled": true,
"attacks-num-malicious": 3,
"attacks-enabled-types": "[\"label_flip\", \"backdoor\"]",
"attacks-distribution": "{\"label_flip\":0.6,\"backdoor\":0.4}"
```

With 15 clients, this assigns two label-flipping clients and one backdoor client,
excluding the head. The effective count is capped by eligible clients and the
configured malicious ratio. Use just one name in `attacks-enabled-types` for a
single attack, `attacks-num-malicious=0` for zero poisoned clients, or
`attacks-enabled=false` to disable the feature entirely.

Poisoning happens once after the existing local train/holdout split. Saved training
overlays fix examples across rounds; clean validation remains untouched. Defaults
are 0.4 per-sample poisoning probability, target class 0, and a white 2x2 trigger on
CIFAR-10. The server logs requested/effective assignments and realized counts to the
terminal, JSON, and TensorBoard. Backdoor ASR is evaluated separately on triggered
copies of non-target holdout examples and never replaces clean accuracy.

Before preparing attack overlays and starting round 1, the server prints **Fake
clients** with their assigned attack types, **Clean clients**, and protected
reference clients when attacks are enabled. IDs appear in numeric order. This is
the simulation's configured ground truth for comparison with defense decisions.
The summary also prints with `verbose: false`; disabled attacks explicitly show
`Fake clients (0/N): none`. Assignments stay fixed across rounds.

After preparing the overlays, before round 1, **Clients with attacks performed**
lists each client with poisoned training samples as `client_N -> attack_type`,
including its poisoned sample count. Clients assigned an attack that poisoned
zero samples are excluded from this list; if no samples were poisoned, it says
`none`. The detailed table below still shows every client.

Outputs live in `results/tensorboard/run_.../attacks/`. See the
[data-poisoning guide](docs/data-poisoning.md) for complete defaults, seed/mask
semantics, supported layouts, metadata, ASR, tests, and limitations. Assignment
ground truth is not forwarded to the clustering defense.

### Client-update clustering defense

Client and round counts can vary between experiments without code edits:

```bash
python run.py --clients 15 --rounds 20
python run.py --clients 5 --rounds 3
python run.py --clients 2 --rounds 2 --pca-components 1
```

These override `simulation.num-supernodes` and `training.num-server-rounds` for
that invocation only; the launcher keeps the partition count synchronized and
checks the settings before launching Flower. `--config path/to/experiment.json`
selects a different configuration file. Omitting the options uses `run_config.json`.
The default two PCA components require at least three participating clients;
two-client clustering requires an explicit one-component choice. One-client
baseline runs require clustering to be disabled. Client IDs remain stable within
each run, and every run gets its own output directory.

Set `training.clustering-enabled` to `true` in `run_config.json`, install the updated
project dependencies (`python -m pip install -e .`), and run `python run.py` normally.
It is disabled by default to preserve baseline FedAvg runs. The server snapshots the
global model before each round, subtracts it from each returned client model, flattens
trainable parameters in validated order, normalizes each client vector, fits fresh
two-component PCA, and applies contrib HDBSCAN. Its binary score is combined with
direction, magnitude, and trusted-validation scores. With equal weights of 0.25,
the library default trust threshold is 0.50 (two passing checks suffice).
The selected experiment in `run_config.json` overrides `trust-threshold` to **0.51**,
so three passing checks are required with these equal weights. It also sets
`clustering-noise-distance-threshold` to **0.5**. Original
selected models retain Flower's sample-weighted FedAvg aggregation. Set
`training.trust-enabled=false` for historical clustering-only selection.

After each successful combined-defense aggregation, the terminal prints the round,
trust threshold, and **Accepted clients** / **Rejected clients** lists with IDs
and trust scores in numeric client order. These lists describe successful replies
used or excluded by FedAvg; missing/failed replies are reported separately.
The summary prints even with `verbose: false`. Rejection is a defense decision,
not a confirmed malicious-client label.

Keep `fraction-train=1.0` to request the trusted client every round. A missing head,
invalid input, insufficient clients, or identical normalized updates aborts the
experiment; there is no accept-all fallback. Direction uses residuals from the
unweighted mean of all trained client models. Magnitude and clustering use updates
relative to the immutable round-start global model. Trusted validation evaluates
each full client model on the same clean `client_0` holdout.

After each completed round's aggregation and evaluation, the server automatically
saves `results/tensorboard/run_.../defense_scores.xlsx` and compact
`defense/round_NNN.json` records. The workbook contains Scores, Direction, Magnitude,
Validation, Clustering, and Configuration sheets. Atomic saves preserve earlier
rounds on later failure. Editing workbook configuration does not recalculate scores.
No separate export command is required. Baseline and clustering-only modes retain
their previous output policy without inventing missing scores. See
[the combined defense guide](docs/combined-defense.md) for equations, thresholds,
configuration, trusted-validation limitations, and workbook persistence.

Inspect `results/tensorboard/run_.../clustering/round_NNN.json` and `.png` for exact
coordinates, variance, labels, strengths, norms and scores. TensorBoard shows the
same information under `detection/clustering/`. Large matrices and PCA loadings are
optional; existing model snapshots remain enabled by default and are never
automatically deleted. See [the clustering guide](docs/clustering-defense.md) for
all options, the complete mathematical pipeline, participation/failure behavior,
replay artifacts, tests, and research limitations.

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> [!NOTE]
> Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations and how to optimize them.

```bash
flwr run . --federation-config "num-supernodes=15" --stream
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --federation-config "num-supernodes=15" --run-config "num-server-rounds=5 learning-rate=0.05" --stream
```

> [!TIP]
> For a more detailed walk-through check our [quickstart TensorFlow tutorial](https://flower.ai/docs/framework/tutorial-quickstart-tensorflow.html)

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.

