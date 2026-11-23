# Four-component client defense and automatic Excel results

The existing message-based Flower `LoggedFedAvg` strategy combines clustering,
direction, magnitude, and trusted validation before aggregating original full
client models. Local training, architecture, partitioning, attack overlays, and
sample-weighted global evaluation are unchanged. This adds a separate inference
stage for client selection; its metrics are not global/final-test metrics.

## Normal invocation and configuration

Install the dependencies in `pyproject.toml`, including `openpyxl`:

```bash
python -m pip install -e .
python run.py
# Optional per-invocation overrides, without rewriting run_config.json:
python run.py --clients 5 --rounds 2
python run.py --clients 2 --rounds 2 --pca-components 1
```

`training.clustering-enabled=true` enables the defense. With `trust-enabled=true`
(default), all four checks execute and combined trust determines selection.
`trust-enabled=false` retains the previous clustering-only selection and artifacts,
without a four-component workbook. `clustering-enabled=false` retains baseline
FedAvg and produces no invented component scores or defense workbook. The current
`run_config.json` enables clustering; the standalone app defaults disable it.

Options use the existing flat keys in the JSON `training` object or Flower run
configuration. The launcher preserves TOML override files and validates settings
before launch. Counts derive from `simulation.num-supernodes`; no population or
PCA dimension is silently changed. The shared head is `clustering-head-client`
(default `client_0`) and must successfully participate. No replacement is chosen.

| Configuration key | Default | Meaning |
|---|---|---|
| `trust-enabled` | `true` | Combined selection when clustering is enabled |
| `direction-threshold-degrees` | `90.0` | Strict upper angle threshold, in `[0,180]` |
| `direction-epsilon` | `1e-12` | Positive denominator epsilon |
| `direction-reference` | `unweighted_all_client_mean` | Required reference; other values rejected |
| `magnitude-threshold` | `0.25` | Relative interval tolerance in `[0,1]` |
| `magnitude-boundary-rtol` | `1e-5` | Nonnegative `isclose` relative tolerance |
| `magnitude-boundary-atol` | `1e-8` | Nonnegative `isclose` absolute tolerance |
| `magnitude-reference` | `round_start_global_model` | Required reference; other values rejected |
| `validation-accuracy-tolerance` | `0.06` | Absolute accuracy difference in `[0,1]` |
| `trust-threshold` | `0.50` | Inclusive selection threshold in `[0,1]` |
| `trust-weight-clustering` | `0.25` | Weight of binary clustering score |
| `trust-weight-direction` | `0.25` | Weight of binary direction score |
| `trust-weight-magnitude` | `0.25` | Weight of binary magnitude score |
| `trust-weight-validation` | `0.25` | Weight of binary validation score |

All numerical settings must be finite. Weights are nonnegative and sum to one
(sum checked within `1e-12`). Clustering settings and population constraints remain
as documented in [the clustering guide](clustering-defense.md).

## Two different references and aligned vectors

Before local training, `configure_train` copies the global weights into an immutable
`W_t`. Every successful client reply contains the complete trained model `W_i`,
not a delta. The reply must declare this convention and the correct round and
ordered Keras schema. Parameter names, shapes, dtypes, trainability, counts, and
ArrayRecord ordering are checked. IDs remain `client_N`, including partial samples.

Let `F` flatten and concatenate trainable parameters in that schema order,
including kernels, biases, and normalization gamma/beta. Optimizer state and
non-trainable buffers are excluded from these vectors. Complete buffers remain
present for validation and FedAvg.

For the `N` successful participating clients, **before any filtering**:

```text
W_mean = (1/N) sum_i W_i
r_i = F(W_i - W_mean)           # direction only
delta_i = F(W_i - W_t)          # magnitude and clustering
```

The implementation computes the trainable portion of the mean after flattening,
which is equivalent by linearity. It is unweighted by example count and includes
clients ultimately rejected. Failed replies cannot contribute a model and are
excluded under the existing successful-population policy. The mean never replaces
actual aggregation. If additive deltas were used instead, the equivalent residual
would be `delta_i - mean_j(delta_j)`; the current wire interface requires full models.

Direction and magnitude use float64 differences and accumulation without PCA,
standardization, or layer/row normalization. Clustering retains its existing
float32 standard updates, row L2 normalization, fresh unwhitened PCA, and contrib
HDBSCAN. Its binary head-relative score is unchanged; membership strength is
`probabilities_`, not benign probability and not the component score.

## Direction

For each non-head client, using residuals formed **before** any angle calculation:

```text
cosine_i = clip(dot(r_i, r_head) / (norm(r_i)*norm(r_head) + epsilon), -1, 1)
angle_i = degrees(arccos(cosine_i))
angle_head = 0.0
D_i = int(angle_i < direction_threshold_degrees)
```

The default threshold is strict: exactly 90 degrees fails. Only head-to-client
angles are computed. A comparison involving an exact zero residual gives 90 degrees
under the epsilon formula; the head angle is explicitly zero even for a zero head.
Zero residuals are flagged. A zero vector has undefined geometric direction, so
90 degrees here is a numerical convention rather than a measured direction.

## Magnitude

```text
m_i = sqrt(sum_j delta_i[j]**2)
lower = m_head * (1 - magnitude_threshold)
upper = m_head * (1 + magnitude_threshold)
within = lower <= m_i <= upper
boundary = isclose(m_i, lower, rtol=1e-5, atol=1e-8)
           or isclose(m_i, upper, rtol=1e-5, atol=1e-8)
M_i = int(within or boundary)
```

The nominal default interval is 75%–125% of the head norm. A head norm of 4 gives
inclusive bounds `[3,5]`, with the explicit numerical boundary tolerances above.
There is no division by the head norm: a zero head gives `[0,0]`, and the absolute
tolerance permits sufficiently tiny nonzero updates. Measurements never use full
weights alone, centered residuals, normalized updates, or PCA coordinates.

## Clean trusted validation

Every client model is evaluated on **the same `client_0` clean local holdout** from
the cached CIFAR-10 partition plan. This compatibility source stays `client_0`
even if `clustering-head-client` is changed. Its actual source is recorded. It is
never a training attack overlay and never the official final test batch.

A dedicated CPU Keras clone loads the full client state, including BN moving
statistics, before each evaluation. Inference uses `training=False`, disabling
dropout and training augmentation and leaving BN buffers unchanged. Server models
and reply arrays are not modified. Metrics reset between clients; fresh explicit
sums aggregate sparse categorical cross-entropy (including the model's existing
regularization losses) and correct classifications by example count, including
the final short batch. Accuracy is stored as a fraction in `[0,1]`. Empty trusted
data, nonfinite metrics, or a missing head abort rather than fabricating accuracy.

```text
minimum_accuracy = accuracy_head - validation_accuracy_tolerance
V_i = int(accuracy_i >= minimum_accuracy)
```

The tolerance is explicitly passed as `0.06` by default: six percentage points,
not a relative 6% reduction. A head accuracy of `0.90` accepts `0.84` inclusively.
Decimal subtraction prevents a one-ULP binary rounding artifact from excluding
that exact boundary. Validation loss remains diagnostic and does not decide `V_i`.

## Trust and aggregation

```text
trust_i = w_c*C_i + w_d*D_i + w_m*M_i + w_v*V_i
selected_i = trust_i >= trust_threshold
```

The library default threshold is `0.50`. The selected `run_config.json` experiment
uses `trust-threshold=0.51`, rejecting `0.50`; with four weights of `0.25`,
three successful checks are then required. The cutoff is inclusive.
Every component must
provide valid aligned results for every analyzed client. Missing/invalid scores
are errors; there are no default zeros or passes. An empty selected set aborts.
Actual selected **original full-model replies** go to Flower's existing
`num-examples` weighted FedAvg. Residuals, normalized vectors, and PCA coordinates
are analysis artifacts only. Clustering diagnostic pass counts describe the
clustering component; `detection/trust/` logs actual combined selection.
After successful aggregation, one Flower terminal log record prints the round,
effective trust threshold, and accepted/rejected client lists with their scores.
Only successful replies appear in those lists, in numeric client order; the
training-error logs handle failed replies. Keras verbosity does not control this
summary. The lists report aggregation decisions, not attack ground truth.

## Automatic workbook and persistence

Normal runs create:

```text
<current_run_results_dir>/defense_scores.xlsx
# Normally results/tensorboard/run_<id>_<timestamp>_<uuid>/defense_scores.xlsx
```

No export command, manual result dictionary, or configuration copy is needed.
The ServerApp constructs the evaluator and `LoggedFedAvg` owns measurement,
selection, and export. Flower's final evaluation callback is wrapped to save
**after aggregation, local validation aggregation, and any existing centralized
evaluation complete**. Initial round 0 is not exported. A failure in a later
round leaves earlier completed rounds in the workbook. Existing policies for
failed non-head replies and optional evaluation remain unchanged.

After **all requested rounds complete**, the same strategy automatically writes
`confusion_matrix/` inside the current `results/tensorboard/run_<id>_<timestamp>_<uuid>/`
directory. It contains separate `clustering.png`, `direction.png`, `magnitude.png`,
and `validation.png` detection confusion matrices pooled across all rounds, plus
matching JSON files with overall/per-round counts, precision, recall, F1, accuracy,
and each observed client decision. No extra command, model inference, or weight
archive is needed. Existing experiments are not modified.

The positive class is **malicious**. Each component's binary score 0 predicts
malicious and score 1 predicts benign; raw cluster/noise labels and combined trust
are not substituted. Matrix rows are actual classes and columns are predicted
classes, in `[benign, malicious]` order: `[[TN, FP], [FN, TP]]`. Ground truth comes
only from this run's saved `attacks/assignment_and_poisoning.json`. It means fixed
configured malicious-client membership, including inactive model-replacement
rounds and assignments with zero poisoned samples. With attacks disabled, all
clients are benign; undefined precision/recall/F1 values are JSON `null`.

Each successful client reply counts once per completed round, including rejected
clients and the trusted head. Failed, missing, or unsampled replies are excluded;
counts are not sample-weighted. The same client appears repeatedly across rounds.
These reports measure component decisions within the combined-defense run, not
four separate training experiments or CIFAR-10 image-classification predictions.
Assignment ground truth is used only by the post-run exporter, never the defenses.
Aborted or early-stopped runs do not produce a final confusion-matrix folder;
their existing completed-round JSON/workbook records remain available. Runs with
combined trust disabled do not fabricate the unavailable component scores and
therefore do not produce these four reports.

The workbook contains `Scores`, `Direction`, `Magnitude`, `Validation`,
`Clustering`, and `Configuration`. Each measurement sheet has one row per actual
client per completed round, numeric round/client ordering, frozen headers,
filters, typed numeric/Boolean cells, and percentage display for fractional
accuracies. The clustering sheet stores **all** PCA coordinates in a labeled JSON
field. Configuration records effective settings, head, validation source, run,
model, dataset, available seeds, attack configuration, and available code revision.
The Flower run ID is stored as text to preserve every digit of its 64-bit value
and allow exact configuration checks when saving later rounds.
Unavailable metadata is labeled as such. The workbook is a **snapshot**: editing
configuration cells does not recalculate any defense.

Compact structured results keyed by stable client ID are also written to
`defense/round_NNN.json` after round completion. `defense_config.json` records
effective settings. Existing clustering artifacts, rejected-client archives, and
global snapshots remain available under their usual paths.

Each save validates alignment and merges existing workbook history by round and
client ID. Identical repeated saves cannot duplicate rows; conflicting historical
rows or mismatched configuration fail clearly. A file lock prevents concurrent
writers. A temporary file in the **same run directory** is fully written before
atomic replacement of the destination. Failure reports the path and original
error, cleans the temporary file where possible, and aborts the experiment under
the existing failure policy while preserving the prior workbook.

The application has **no run-resume mechanism**: each launch creates a unique run
directory. No resume feature or cross-run merging is introduced. If an internal
writer is reconstructed for the same run/configuration, existing completed workbook
rows are preserved rather than replaced by just the newly supplied rounds.

## Verification and limitations

`test_defense.py` checks references, vector alignment, thresholds, zero vectors,
trust, and Keras inference isolation. `test_defense_excel.py` checks export fidelity,
formatting, ordering, idempotency, history, and failure preservation.
`test_defense_integration.py` runs two synthetic rounds through `run.main`, the
normal ServerApp, and Flower strategy lifecycle with synthetic client transport
and a small Keras classifier. It verifies the automatic workbook against actual
computed results and original-model weighted aggregation, plus partial identities
and evaluation failure timing. This is not a real Ray training experiment and
does not establish attack-detection effectiveness.

The defense depends on a trusted and representative head. Malicious clients can
influence the all-client mean used by direction. Magnitudes are sensitive to local
data, optimizer behavior, and training duration. Clean validation accuracy alone
can miss backdoors and can penalize benign non-IID clients; a small/unrepresentative
holdout limits its usefulness. Reusing a local holdout for selection does not make
it independent final testing. Per-client trusted inference adds CPU work each round.
The existing PCA information loss, HDBSCAN instability at small populations, and
unauthenticated client metadata limitations still apply.
