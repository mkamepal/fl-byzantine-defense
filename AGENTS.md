# Project instructions for coding agents

## Scope and working approach

These instructions apply throughout this repository. Follow more specific
`AGENTS.md` files where present and the user's current task instructions.

- Inspect the affected implementation, configuration, documentation, and tests
  before editing. Reuse the existing Flower lifecycle rather than adding a second
  training pipeline.
- Keep changes focused and explain changes to experimental behavior. Preserve
  unrelated user edits and experiment settings.
- Do not change model architecture, training, evaluation, attacks, partitioning,
  or aggregation semantics merely to add logging or diagnostics.
- Make routine implementation decisions without repeated confirmation. Clearly
  distinguish implemented behavior, measured results, and unverified assumptions.
- Check whether Git is actually available before relying on Git commands; a
  `.git` directory alone does not establish that this checkout has usable history.

## Architecture and important files

This is a TensorFlow/Keras application using Flower's message-based `ClientApp`,
`ServerApp`, `ArrayRecord`, `ConfigRecord`, and `MetricRecord` APIs. Do not introduce
PyTorch or replace these interfaces with the legacy NumPyClient API.

| Location | Responsibility |
|---|---|
| `run.py` | JSON configuration, per-run CLI overrides, streamed launch, process-group cleanup |
| `run_config.json` | User's selected simulation and training settings |
| `pyproject.toml` | Dependencies, Flower app entry points, app configuration defaults |
| `fl_defense/client_app.py` | Local training/evaluation and full-model replies with stable client metadata |
| `fl_defense/server_app.py` | Experiment initialization, dataset reporting, strategy lifecycle, final results |
| `fl_defense/aggregation/logged_fedavg.py` | FedAvg hooks, archives, clustering selection, centralized metrics |
| `fl_defense/models/resnet56.py` | Active CIFAR-10 model; `models/cnn.py` preserves the CNN baseline |
| `fl_defense/data/` | Cached CIFAR-10 loading, IID/Dirichlet partition plans, identities, summaries |
| `fl_defense/attacks/` | Deterministic label flipping/backdoors, fixed training overlays, separate ASR evaluation |
| `fl_defense/detection/parameters.py` | Ordered Keras parameter schema and trainable update extraction |
| `fl_defense/detection/clustering.py` | Configuration, normalization, PCA, HDBSCAN, binary scores |
| `fl_defense/detection/clustering_output.py` | Terminal cluster tables, plots, compact diagnostics, TensorBoard events |
| `fl_defense/utils/experiment_config.py` | Shared per-run configuration validation |
| `fl_defense/utils/tensorboard.py` | One server-owned experiment logger and dataset visualizations |
| `fl_defense/utils/weight_archive.py` | Persistent model snapshots and weight-shift diagnostics |
| `fl_defense/utils/client_resources.py` | Keras cleanup for reused simulation workers |
| `tests/` | Unit, integration, and optional simulation smoke checks |
| `docs/clustering-defense.md` | Mathematical pipeline, configuration, failure policy, replay, limitations |

Direction, magnitude, trusted-validation, and combined-trust decisions are
implemented in `fl_defense/detection/` and integrated into `LoggedFedAvg`.
Direction uses the unweighted all-client model mean; magnitude and clustering
use the immutable round-start global model. Archival weight-shift norms remain
diagnostics. See `docs/combined-defense.md`; never invent missing component scores.

## Configuration and dynamic experiments

- Client and round counts vary between runs. Never hardcode 15 clients, 3 clients,
  or a fixed round count in application logic, reports, or artifact handling.
- `simulation.num-supernodes` is the launcher's source of truth for the population;
  it derives `training.num-clients`. `training.num-server-rounds` sets the rounds.
- CLI overrides apply to one invocation and must not rewrite the JSON file:

  ```bash
  python run.py --clients 15 --rounds 20
  python run.py --clients 5 --rounds 3
  python run.py --clients 2 --rounds 2 --pca-components 1
  python run.py --config experiments/my_run.json --clients 10 --rounds 5
  ```

- Check the effective configuration rather than assuming `run_config.json` and
  `pyproject.toml` have the same values. New app settings must be declared in
  `pyproject.toml` and supported by the existing configuration path.
- Preserve TOML-file overrides in `run.py`; nested JSON strings have caused
  failures with Flower's inline configuration parser.
- Validate settings before launching expensive work. Do not silently reduce PCA
  components, increase the population, disable the defense, or change sampling to
  make an invalid experiment run.
- Keep `run.py` process-cleanup helpers and `--help` usable without importing ML
  dependencies; import configuration validation lazily when preparing a run.

## Data and client identity

- Logical IDs are `client_0` through `client_{N-1}`, derived from partition IDs,
  never worker execution order or Flower node IDs. `client_0` is the default
  trusted head. Use numeric ordering and preserve IDs during partial participation.
- Reuse the cached raw CIFAR-10 data and exact partition plan for training,
  statistics, and image previews. Do not independently repartition for logging.
- Keep IID and Dirichlet behavior and seed handling intact. Partitions and local
  train/validation membership must stay stable across epochs and rounds within a
  run. A different population in a new run can produce a different partition plan.
- Validate disjoint coverage and actual per-class counts. Keep local validation
  separate from training and the official test batch. Current server global
  accuracy/loss are aggregated local-validation metrics, not official test accuracy.
- Preserve the existing split-specific loading and worker cleanup. Avoid caching
  client models or repeatedly materializing whole normalized datasets.

## Clustering and aggregation invariants

- Clients return full model weights. Snapshot the global model before local
  training and compute `W_i - W_t` using that immutable reference for every client.
- Flatten only trainable parameters in validated schema order, including biases
  and normalization gamma/beta. Exclude optimizer state and non-trainable buffers.
  Check counts, names/order, shapes, dtypes, and finite values; never truncate with
  unchecked `zip` calls.
- Rows of the float32 `(N, D)` matrix are clients. Retain raw norms and zero-update
  flags. Normalize each whole nonzero row by its L2 norm; leave zero rows zero.
- Fit fresh PCA each round, default two components, seed 42, no whitening. Do not
  standardize columns, normalize layers independently, or renormalize PCA output.
  Use the same PCA coordinates for HDBSCAN, head distances, and plotting.
- Use the contrib `hdbscan` package. Preserve the configurable defaults and exact
  score rules documented in `docs/clustering-defense.md`: head scores 1; if either
  label is noise use inclusive head-distance thresholding; otherwise compare labels.
- Require the configured head. For `k` PCA components require
  `1 <= k <= min(N - 1, D)`. Two-client clustering requires an explicit one-PC
  choice; one-client runs require clustering disabled. Validate sampled and
  successful populations as well as the configured population.
- Invalid inputs, absent heads, and degenerate identical normalized updates abort
  aggregation. Do not introduce an accept-all fallback or choose a replacement head.
- Filter original client model replies, then preserve FedAvg's sample weighting.
  Never aggregate normalized updates or PCA coordinates. Keep unrelated defense
  references and evaluation behavior unchanged.
- Membership strength is HDBSCAN `probabilities_`, not benign probability or a
  binary selection score. Noise is not a maliciousness label; cluster IDs have no
  persistent meaning across rounds.

## Simulated data-poisoning attacks

- Apply attacks only after partitioning and the local train/holdout split. Protect
  `client_0`, the configured attack head, and the clustering reference. Do not
  poison benign training data, validation data, or the official test set.
- Reuse the server-created training overlays under each run's `attacks/training/`
  directory across rounds. Do not resample masks per epoch/round or mutate cached
  source arrays. Missing or mismatched overlays must fail clearly.
- Preserve local `RandomState(seed)` semantics and sample order. Label flips choose
  a different locally observed class; backdoors stamp every selected example,
  including original target-class examples. See `docs/data-poisoning.md`.
- Keep attack assignments and diagnostics out of defense inputs and training
  replies. Normal training produces full model weights without update manipulation.
- Evaluate ASR separately on triggered copies of non-target clean holdouts, using
  all eligible examples and reporting unavailable when the denominator is zero.
  Aggregate success/eligible counts, not sample-weighted client ASR percentages.

## Logging, artifacts, and process safety

- Use one server-owned TensorBoard logger per unique experiment directory under
  `results/tensorboard/`. Use hierarchical tags; do not create client-side writers.
- Print each round's client clusters and membership percentages through Flower's
  terminal logger, marking noise and the trusted reference. Keep each table in one
  log record and independent of Keras training verbosity.
- Save compact clustering JSON and headless plots under the run's `clustering/`
  directory. Preserve coordinates, variance, labels, norms, scores, and effective
  settings. Release figures and round-local analysis state after use.
- Model archives live under the run's `models/` directory and include successful
  rejected clients when enabled. Preserve atomic, non-overwriting writes. Full
  vectors and PCA loadings remain optional. Distinguish compact inspection data
  from artifacts needed to refit or replay an analysis.
- Preserve existing experiments, weights, logs, and cached datasets. Do not add
  automatic deletion or retention limits without a user request. Never imply that
  weights-only checkpoints contain client optimizer state.
- Use persistent absolute output/data paths across Flower subprocesses. Edit source
  here, not generated bundles or runtime environments under `~/.flwr`.
- Do not broadly kill Python, Ray, or Flower processes. Launcher cleanup only owns
  its process group; separately managed services may require run-specific stopping.
  Do not disable Ray's memory monitor to conceal memory growth.

## Verification and completion

Use an activated project environment with dependencies from `pyproject.toml`.
An environment name containing “pytorch” does not change this project's framework.
Do not assume a temporary environment from a previous session still exists.

Run relevant focused checks first, then the applicable regression suite:

```bash
python -m unittest discover -s tests -p 'test_clustering*.py' -v
python -m unittest discover -s tests -p 'test_run_config.py' -v
CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fl-matplotlib \
  python -m unittest discover -s tests -v
```

- Add focused tests for behavioral changes. Check configuration overrides, varying
  populations/rounds, partial participation, failure paths, and unchanged baseline
  aggregation where relevant. Do not assert arbitrary numeric cluster labels.
- Prefer synthetic Flower lifecycle checks over a full training experiment for
  integration validation. Synthetic clustering results do not prove attack detection.
- `tests/smoke_tensorboard_simulation.py` is an optional real CPU/Ray smoke test,
  separate from unittest discovery. Inspect its setup before using it; its default
  two-client baseline is not a default two-PC clustering test.
- Never claim a test or full simulation passed unless it actually ran. Report
  dependency, hardware, or environment limitations and any untested behavior.
- Documentation-only changes normally need document/path checks, not training.
- Update README or the clustering guide for changed user-facing behavior. Finish
  with files changed, how to use the change, checks actually run, output locations,
  and material limitations as appropriate to the task.
