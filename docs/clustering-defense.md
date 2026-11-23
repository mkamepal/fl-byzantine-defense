# Round-local client-update clustering

The optional defense runs inside `LoggedFedAvg`, using the existing TensorFlow/Keras
ClientApp and Flower message-based FedAvg strategy. It is disabled by default.
The defense itself does not replace the training pipeline or change local training,
partitioning, model architecture, or existing global evaluation. Combined trust
adds clean trusted-validation inference for selection; see the
[combined defense guide](combined-defense.md). Optional data-poisoning attacks are
configured separately; see [data poisoning](data-poisoning.md). Enabling clustering changes
which returned models participate in aggregation.

## Enable and configure

Install the project dependencies, including the **scikit-learn-contrib `hdbscan`**
package (not `sklearn.cluster.HDBSCAN`):

```bash
python -m pip install -e .
```

In `run_config.json`, set `training.clustering-enabled` to `true` and run
`python run.py` normally. Additional options go in the same `training` object.
Client and round counts are per-experiment settings, not fixed constants. Set
`simulation.num-supernodes` and `training.num-server-rounds` in that file, or
override them for just one invocation:

```bash
python run.py --clients 15 --rounds 20
python run.py --clients 5 --rounds 3
python run.py --clients 2 --rounds 2 --pca-components 1
python run.py --config experiments/my_run.json --clients 10 --rounds 5
```

The launcher derives `num-clients` from the final `num-supernodes` value, prints
the resolved settings, and validates them before launching Flower. Overrides never
modify the JSON file. Dataset partitions, IDs `client_0` through `client_{N-1}`,
tables, plots, archives, and round counters use the resolved counts. Changing client
count between experiments rebuilds the partition plan; identity and partition
mapping stay stable within each experiment. Each invocation retains its own run
directory. Dataset size and train/validation splitting still constrain valid
partition counts.

For two clients, explicitly request one PCA component as above; the default two-PC
pipeline still needs at least three participating clients. No component count or
client count is changed automatically. One-client baseline runs are supported with
`clustering-enabled=false`; clustering needs at least two clients. Positive integer
round counts are required. The configured sampling fraction must request enough
clients for the chosen PCA and HDBSCAN settings, and the head must return each round.

Alternatively, use Flower's existing configuration overrides:

```bash
flwr run . --run-config 'clustering-enabled=true clustering-noise-distance-threshold=1.5'
```

All options are declared in `pyproject.toml`:

| Key | Default | Constraint / meaning |
|---|---|---|
| `clustering-enabled` | `false` | Enable defense; combined trust is the default selection policy |
| `trust-enabled` | `true` | Set false to retain standalone clustering selection |
| `clustering-n-components` | `2` | Fixed integer, `1 <= k <= min(N-1,D)` |
| `clustering-random-state` | `42` | PCA seed, integer in `[0,2^32)` |
| `clustering-metric` | `"euclidean"` | Only Euclidean is supported by this pipeline |
| `clustering-min-cluster-size` | `2` | Integer `2 <= value <= N` |
| `clustering-min-samples` | `1` | Integer `1 <= value < N` |
| `clustering-cluster-selection-method` | `"eom"` | `"eom"` or `"leaf"` |
| `clustering-cluster-selection-epsilon` | `0.2` | Finite, nonnegative |
| `clustering-allow-single-cluster` | `true` | Permit, but do not force, one cluster |
| `clustering-head-client` | `"client_0"` | Existing stable client ID, required in every analyzed round |
| `clustering-noise-distance-threshold` | `1.5` | Inclusive comparison, finite value in `[0,2]` |
| `clustering-save-update-vectors` | `false` | Save full raw `N x D` update matrix |
| `clustering-save-pca-loadings` | `false` | Save components, mean and singular values |
| `save-client-weights` | `true` | Keep all returned client models and global before/after snapshots |

Whitening is fixed to `False`. Configuration never substitutes variance-based
component selection, column standardization, or a different clustering algorithm.
No fixed number of clusters is requested. HDBSCAN's remaining library defaults
apply; package versions are saved in each successful round report.

## Reference, alignment and shapes

1. Before sending local training instructions, `configure_train` makes a copied,
   read-only snapshot `W_t` of the global model. It is separate from archival
   snapshots, so disabling disk checkpoints does not affect the reference.
2. The existing ClientApp calls `model.set_weights(W_t)`, trains normally, and
   returns **full** `model.get_weights()` values `W_i`, not deltas. Client metadata
   declares the round, full-model convention, and ordered parameter schema.
3. The server derives that schema from Keras `model.weights` and actual membership
   in `model.trainable_weights`. Names, order, shapes, dtypes, counts and trainability
   must match. ArrayRecord keys must also have the expected order. Shape/count
   mismatches are rejected rather than truncated. The metadata verifies alignment,
   not the honesty of a Byzantine client's self-reported identity or computation.
4. For each trainable tensor `l`, form `delta_i,l = W_i,l - W_t,l`. This is the
   pre-training global reference for **every** client. Include kernels, trainable
   biases, and normalization gamma/beta; exclude non-trainable moving statistics
   and optimizer state. Keras weights are already CPU NumPy arrays on the server.
5. Flatten each tensor in C order and concatenate in schema order. The result is
   `x_i in R^D`, where `D = sum(product(shape_l))` over trainable tensors. Stack
   clients in numeric ID order: `client_2` precedes `client_10`. Save the explicit
   row-to-ID mapping. Partial participation never renumbers clients.

The analysis matrix `X` has shape `(N,D)` and dtype `float32`. Non-finite parameters,
non-finite deltas, and incompatible dimensions cause failure. Raw L2 norms use
float64 accumulation to avoid squared-norm overflow or underflow.

## Normalization, PCA, clustering and scoring

For each client, compute `r_i = ||x_i||_2`. For `r_i > 0`, set `u_i = x_i/r_i`.
A zero update remains a zero row and is explicitly flagged in diagnostics. Zero
rows take part in PCA/HDBSCAN and the same scoring rules; they are not unit vectors
and are not automatically malicious or trusted. A zero head is still assigned 1
by the specified reference rule. No layer-wise normalization or column scaling is
performed, and there is no feature centering before row normalization.

Fit a **fresh** `PCA(n_components=k, random_state=42, whiten=False)` each round on
`X_norm`. PCA itself centers the features, then produces `Z` of shape `(N,k)`.
There is **no** normalization after PCA. HDBSCAN, head distances, and plots all use
this exact `Z`. Explained variance ratios and their cumulative sum are recorded;
there is no assumption that two PCs retain 95% of variance.

Fit `hdbscan.HDBSCAN` on `Z` using the configured Euclidean parameters. Let its
labels be `label_i`, membership strengths be `p_i`, and the head row be `h`.
Record `d_i = ||z_i-z_h||_2` for every client. Scores are exactly:

```python
if client_id == head_client:
    score = 1
elif label_i == -1 or label_h == -1:
    score = int(d_i <= noise_distance_threshold)
else:
    score = int(label_i == label_h)
```

Membership strength never enters this binary decision. Nonnegative labels identify
clusters; `-1` is noise. A noise client can pass the distance test. When the head is
noise, every other client uses the distance test. The threshold bound `[0,2]` follows
from whole-row unit normalization (or zero rows) and unwhitened orthogonal projection.

## Participation, failure and aggregation

The trusted reference must return a valid update in each round. Existing Flower
sampling is preserved. `fraction-train=1.0` (the current setting) requests all clients
and is recommended; partial participation works only when it includes the configured
head and enough successful clients. A head training failure or an unlucky partial
sample that omits the head **aborts the experiment**. No replacement head is chosen.
The configured head must belong to the configured client partitions. `client_0`
remains the application's default trusted-head role; overriding the reference is an
explicit research choice, not a reassignment of dataset IDs or roles.

Default two-component PCA requires at least three successful clients and two
trainable scalar parameters. Invalid inputs, missing head, insufficient PCA/HDBSCAN
population, or all identical normalized rows (absolute tolerance `1e-7`, appropriate
for float32) raise an exception **before aggregation**. Collinear but nonidentical
rows are allowed; unused PCs may retain zero variance. No fallback accepts everyone,
reduces the number of PCs, or swaps the algorithm. A compact failure report is saved
when round analysis fails, and the experiment logger marks the experiment failed.
Failed non-head replies are excluded from the analysis population, provided the
remaining population satisfies all checks. The in-memory baseline is then released.

Population constraints are checked at server startup and against the sampled
clients before sending training requests, then checked again on successful replies.
For example, two clients with two PCs is rejected with the actual `N`, `D` (when
known), and requested component count. Use at least three clients for a small
two-PC smoke run, or 15 for the intended experiment. Three is only the minimum
valid population, not evidence that density clustering is reliable at that size.

With `trust-enabled=true` (default), `ClusteringResult.clustering_scores` is one
component of the implemented four-check formula
`0.25*(clustering+direction+magnitude+validation) >= 0.50`. Two passing checks suffice.
Direction uses the all-client mean, while magnitude and clustering retain `W_t`.
See [combined defense](combined-defense.md) for the full mathematical definitions,
clean trusted holdout, configuration, and automatic Excel export. Setting
`trust-enabled=false` restores selection by `clustering_i == 1`. Archival weight-shift
norms remain diagnostics; the magnitude component is separately measured and scored.

Selected **original full model replies** are passed to Flower FedAvg, retaining its
`num-examples` weighted average, including the model's non-trainable state. Normalized
vectors and PCA coordinates never enter aggregation. Client train metrics are logged
for all successful replies; aggregate train metrics use the selected subset. The
`successful_updates` count includes rejected but successfully returned models;
`detection/clustering/selected_clients` and `rejected_clients` count component passes.
In combined mode, `detection/trust/selected_clients` and `rejected_clients` report
actual aggregation selection.
Evaluation behavior remains unchanged (sample-weighted local validation, not a new
official CIFAR-10 test evaluation). The defense resets its baseline every round.

## Results, plots and replay

Everything is kept in the existing unique experiment directory:

```text
results/tensorboard/run_.../
    config.json
    clustering_config.json
    clustering_parameter_schema.json
    defense_config.json              # combined mode
    defense_scores.xlsx              # automatically saved after completed rounds
    defense/round_001.json            # combined results keyed by client ID
    clustering/
        round_001.json
        round_001.png
        round_001_analysis.npz       # only when large-array options are enabled
        round_002_failure.json       # only on failure
    models/                         # save-client-weights=true (default)
        model_architecture.json
        weight_schema.json
        round_001_global_before.npz
        round_001_client_0.npz
        ...                         # includes rejected successful client models
        round_001_global_after.npz
```

Compact round JSON contains the round, ordered client IDs, original vector dimension,
raw norms, zero flags, PCA coordinates, explained variance and cumulative variance,
labels, strengths, head distances, binary scores, cluster/noise counts, effective
configuration, and library versions. Files are created atomically without overwrite.
The PNG is produced with an Agg canvas, annotated IDs, cluster colors, noise crosses,
a boxed head, and marker sizes reflecting membership strength. It shows round,
cluster/noise counts and retained variance. For more than two PCs the plot explicitly
says it is a 2D projection of the clustering space; clustering and distances still
use all configured PCs. One-component mode uses PC1 with a zero display ordinate.
Figures have no interactive window or pyplot registration and are cleared after use.

TensorBoard receives the plot at `detection/clustering/coordinates`, client fields
under `detection/clustering/client_N/`, and round counts/variance under
`detection/clustering/`. Summaries flush each round:

```bash
tensorboard --logdir results/tensorboard --port 6006
```

Open `http://localhost:6006`. Completed experiments remain available for comparison.

After each successful round analysis, the server also prints a client-to-cluster
table in the streamed Flower logs, independently of training verbosity. It uses
numeric client order, labels noise as `Noise (-1)`, and marks the trusted reference.
The `Membership strength` column displays each client's actual HDBSCAN
`probabilities_` value as a percentage, including the head's measured value.
This is strength of membership in the assigned cluster, not a calibrated benign
probability or the binary selection score. Noise points have zero membership
strength even when the distance-based selection rule accepts them.
Only clients with analyzed updates appear; failed or nonparticipating clients are
not assigned a cluster. This output requires `clustering-enabled=true`.

Compact diagnostics are sufficient for inspecting the actual fitted coordinates and
scores, **not for refitting the high-dimensional pipeline**. Optional NPZ arrays are
named `update_vectors`, `pca_components`, `pca_mean`, `pca_singular_values` as enabled.
To refit, retain raw update vectors (row mapping comes from JSON), or every participating
client snapshot plus the round-start global snapshot and parameter schema. Reproducing
floating-point fits also requires matching package versions, parameters, seed, and
numerical environment. Saved PCA loadings and mean permit inspecting/reapplying the
projection, but cannot recover discarded dimensions from compact coordinates alone.
HDBSCAN estimators/trees and client optimizer trajectories are not serialized. No
automatic deletion of research artifacts is introduced. Analysis memory scales as
`O(ND)` and temporarily holds several matrices; full arrays are not retained across
rounds. Disk usage grows with enabled large artifacts and experiment length.

## Verification and limitations

Run `python -m unittest discover -s tests -v`. The focused files
`test_clustering.py` and `test_clustering_integration.py` test arithmetic, ordering,
shape/schema rejection, zero vectors, PCA input/output, exact score rules, failure
policies, real Keras schema membership, and Flower's strategy lifecycle across two
rounds with synthetic client updates. Tests read JSON, checkpoints, plots, and real
TensorBoard events. The synthetic coherent/opposed groups are marked synthetic and
**do not establish attack-detection effectiveness**. No expensive full federation is
needed for these checks.

A trusted **and representative** reference client is assumed. Two PCA components can
discard useful attack information; projected similarity does not guarantee similarity
in the full parameter space. HDBSCAN noise is not a maliciousness label and membership
strength is not a benign probability. PCA axes and cluster IDs can change between
rounds. Small client populations can make density clustering unstable. Non-IID data
can separate honest updates, and thresholds must be evaluated across datasets,
partition settings, attacks and seeds. This implementation has no authentication or
attestation of the reference client and reports no unmeasured detection precision,
recall or F1.

Algorithm references: [scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
and [contrib HDBSCAN API](https://hdbscan.readthedocs.io/en/latest/api.html).
