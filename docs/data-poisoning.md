# Simulated data and model poisoning

The TensorFlow/Keras ClientApp supports three optional attack assignments: random
label flipping, a targeted bottom-right square backdoor, and model replacement.
Model replacement trains on the same kind of triggered local data as the backdoor
attack, then scales the submitted trainable update from the round-start global
model. It is one attack type in the shared malicious-client quota, not a second
assignment for a backdoor client.

Attacks are **disabled by default**. Enable them deliberately for an experiment;
clustering remains independently configurable.

## Configuration

Settings are scalar Flower app options declared in `pyproject.toml`. Add or change
these keys inside the existing `training` object of `run_config.json`. The examples
below are fragments to merge, not replacement files. Lists and distributions use
JSON **strings**, because Flower run settings accept only scalar values. The launcher
already uses temporary TOML files to preserve the quoting.

Mixed attacks:

```json
{
  "attacks-enabled": true,
  "attacks-num-malicious": 3,
  "attacks-enabled-types": "[\"label_flip\", \"backdoor\"]",
  "attacks-distribution": "{\"label_flip\":0.6,\"backdoor\":0.4}"
}
```

To include model replacement in the same quota, add it to the enabled list and
distribution. For example, a quota of seven with weights 0.4/0.3/0.3 allocates
three label-flipping clients, two backdoor clients, and two model-replacement
clients. Largest-remainder quota allocation remains deterministic.

```json
{
  "attacks-enabled": true,
  "attacks-num-malicious": 7,
  "attacks-enabled-types": "[\"label_flip\", \"backdoor\", \"model_replacement\"]",
  "attacks-distribution": "{\"label_flip\":0.4,\"backdoor\":0.3,\"model_replacement\":0.3}",
  "attacks-model-replacement-variant": "basic",
  "attacks-model-replacement-scale-mode": "population_weight_estimate"
}
```

Label flipping only:

```json
{
  "attacks-enabled": true,
  "attacks-enabled-types": "[\"label_flip\"]",
  "attacks-label-flip-prob": 0.4
}
```

Backdoor only:

```json
{
  "attacks-enabled": true,
  "attacks-enabled-types": "[\"backdoor\"]",
  "attacks-backdoor-poison-prob": 0.4,
  "attacks-backdoor-target-label": 0,
  "attacks-backdoor-trigger-value": 1.0
}
```

The default distribution is restricted to enabled types and renormalized, so no
other distribution edit is needed for the single-attack examples. Set
`attacks-distribution` to an empty string to choose attack types uniformly per
selected malicious client instead of assigning fixed quotas.

For a clean baseline set `attacks-enabled=false`. Alternatively, use
`attacks-enabled=true` with `attacks-num-malicious=0`: all training remains clean,
assignment diagnostics are saved, and optional backdoor evaluation can still measure
baseline trigger behavior. `attacks-evaluate-asr=false` disables the additional
triggered-holdout evaluation without changing poisoning or clean evaluation.

Run normally, using dynamic client/round counts:

```bash
python run.py --clients 15 --rounds 5
```

With 15 total clients, the default three malicious clients receive two label-flipping
assignments and one backdoor assignment. With three total clients, the ratio cap
limits the effective malicious count to two (one of each under the default weights).
The configured counts and PCA population constraints are still validated before
launch. Attack configuration does not bypass clustering's minimum population.

| Setting | Default | Meaning |
|---|---|---|
| `attacks-enabled` | `false` | Enable poisoning construction |
| `attacks-seed` | `-1` | Inherit `partition-seed` (default 42); otherwise integer `[0,2^32)` |
| `attacks-num-malicious` | `3` | Requested nonnegative malicious-client count |
| `attacks-max-fake-ratio` | `0.7` | Maximum malicious fraction of the total client population, in `[0,1]` |
| `attacks-exclude-head` | `true` | Mandatory; false is rejected |
| `attacks-head-client` | `"client_0"` | Configured protected head ID |
| `attacks-enabled-types` | `'["label_flip","backdoor"]'` | Nonempty JSON list of unique names from `label_flip`, `backdoor`, `model_replacement` |
| `attacks-distribution` | `'{"label_flip":0.6,"backdoor":0.4}'` | JSON weights; empty string or JSON null selects uniform assignment |
| `attacks-label-flip-prob` | `0.4` | Per-sample Bernoulli probability |
| `attacks-backdoor-poison-prob` | `0.4` | Per-sample trigger/target-label Bernoulli probability |
| `attacks-backdoor-target-label` | `0` | Global CIFAR-10 class ID in `0..9` |
| `attacks-backdoor-trigger-value` | `1.0` | Intended pixel value in `[0,1]` (1 is white) |
| `attacks-backdoor-position` | `"bottom_right"` | Only supported position |
| `attacks-backdoor-patch-size-rule` | `"max_2_or_min_dimension_div_12"` | Only supported size rule |
| `attacks-save-indices` | `false` | Save additional source-row IDs and training positions for research |
| `attacks-evaluate-asr` | `true` | When attacks and the backdoor type are enabled, evaluate separate ASR |
| `attacks-model-replacement-variant` | `"basic"` | `basic` or `constrain_and_scale` |
| `attacks-model-replacement-scale-mode` | `"population_weight_estimate"` | Estimate inverse replacement-client weight from configured sample counts, or use a fixed scale |
| `attacks-model-replacement-scale` | `1.0` | Positive scale used when scale mode is `fixed` |
| `attacks-model-replacement-start-round` | `1` | First active round |
| `attacks-model-replacement-end-round` | `0` | Last active round; `0` means through the configured final round |
| `attacks-model-replacement-interval` | `1` | Attack every N rounds, starting at the configured start round |
| `attacks-model-replacement-epochs` | `0` | Replacement-client local epochs; `0` inherits `local-epochs` |
| `attacks-model-replacement-learning-rate` | `0.0` | Replacement-client learning rate; `0` inherits `learning-rate` |
| `attacks-model-replacement-alpha` | `0.5` | Classification weight for `constrain_and_scale`, in `(0,1]` |

The app-level defaults preserve the existing two-attack configuration. To run
model replacement, explicitly include it in `attacks-enabled-types` and give it a
positive distribution weight, as in the example above.

## Reproducible malicious-client assignment

The candidate pool excludes `client_0`, the configured attack head, and the
configured clustering reference (if different). All protected IDs must exist in
this run. No head exclusion override can poison a trusted reference.

For enabled attacks, the effective count is:

```text
min(requested_num_malicious, eligible_client_count,
    floor(total_client_count * max_fake_ratio))
```

A local `np.random.RandomState(seed)` chooses clients without replacement. Selected
IDs are sorted numerically. Assignment is fixed for the entire experiment, including
partial participation. Disabled attacks have an effective count of zero.

For distribution-based assignment, restrict weights to enabled attack types, check
finite nonnegative values and a positive total, and normalize. Take the floor of
each expected count, then allocate remaining clients by largest fractional remainder.
Ties use alphabetically sorted attack names. Finally shuffle the attack-type list
with the seeded assignment RNG and pair it with the selected clients. Unknown
configured attack names are rejected. Without a distribution, choose uniformly from
enabled names for each selected client. Every selected client gets exactly one type.
Explicit malicious-ID lists are not implemented; selection is seed-driven.

The fraction of **clients** selected as malicious is independent of the fraction of
**training samples** selected for poisoning. A probability of 0.4 does not imply that
40% of all experiment data is poisoned, nor that exactly 40% of a client's samples
will be selected.

## Dataset construction and fixed examples

The order is unchanged until the attack stage:

```text
cached CIFAR-10 train batches
  -> existing IID/Dirichlet client allocation
  -> existing local training/validation split
  -> training images converted to float32 / 255
  -> one assigned attack overlay on the training split
  -> normal Keras batching, shuffling, and local training
  -> for active model-replacement clients, scale trainable weights before reply
```

At server initialization, after the clean partition report, `prepare_attack_datasets`
uses the same cached data and partition plan. It computes each attack once and saves
an overlay for each client: complete resulting training labels, selected training
positions, and already-stamped image rows for backdoor and model-replacement
clients. Workers load these exact arrays on active rounds; there is no attack RNG
invocation in the training loop. Model-replacement clients use their clean split
on inactive rounds. Keras may shuffle training batches normally; poisoning remains
attached to the same original examples. Source arrays, benign examples, clean local
validation, and the official CIFAR-10 test batch are never modified in place.

The overlay's training-index digest is checked against the worker's actual partition.
A missing or mismatched overlay raises an error rather than silently changing an
attack into a clean run. Only one client's data is materialized at a time during
construction, and raw dataset loading reuses the existing cache. The runtime path
is absolute and passed through training configuration; no image arrays, training
labels, or attack assignments are transmitted in Flower metrics.

This supports the existing local Ray simulation and separate local processes. A
multi-host deployment would require a shared persistent filesystem at the same
paths. These files are simulation inputs, not a secure distribution mechanism for
mutually untrusted physical participants.

## Attack definitions

`fl_defense.attacks.label_flipping.label_flip_attack(labels, flip_prob=.4, seed=42)`
returns copied integer labels. It supports `(N,)` and `(N,1)` labels, using the sorted
locally observed class set. Empty or single-class inputs return unchanged labels.
Otherwise, draw `rng.rand(N) < flip_prob`; when any sample is selected, draw one
integer offset in `[1, number_of_local_classes)` for **each** sample and map through
the sorted class indices. Only selected labels are replaced. Each replacement is a
different, uniformly chosen locally observed class. Binary classes swap; labels
such as `[2,5,9]` remain within that set. Images are unchanged. This is not a fixed
cyclic mapping or a targeted flip.

`fl_defense.attacks.backdoor.backdoor_attack(images, labels, target_label=0,
poison_prob=.4, trigger_value=1., seed=42)` returns copied images and labels. For each
selected sample, set all channels of the bottom-right patch to the trigger value
and its label to the global target class. Selected examples that originally belong
to the target class **still receive a trigger**. Therefore trigger counts and label
change counts can differ.

### Model replacement

On an active scheduled round, a model-replacement client snapshots the received
global model (W_t), trains on its fixed triggered overlay, and submits:

```text
submitted trainable weights = W_t + scale * (locally trained trainable weights - W_t)
```

The scale is either configured directly or estimated as the inverse fraction of
the configured population's training samples assigned to model replacement. This
estimate assumes representative participation and a server aggregation step of
one. It is not the actual weight after sampling, failures, or defense filtering;
those are logged separately. Only model-replacement clients enter the estimated
replacement weight; label-flipping and ordinary-backdoor clients do not.

Scaling applies to schema-validated trainable tensors. Nontrainable buffers,
including BatchNorm moving statistics, retain their locally trained values. This
adapts the full-model reply to the active model schema and is not an exact
reproduction of every detail in the paper's experiments.

`basic` uses ordinary local training on the triggered overlay, then scales the
submitted update. `constrain_and_scale` adds a mean-squared distance penalty from
the round-start trainable weights during eager local training before applying the
same scaling step. The classification weight `alpha` controls the tradeoff;
existing Keras regularizers are retained. This penalty is a smooth anomaly
surrogate, not a differentiable copy of the server's direction, clustering, or
validation decisions.

The default schedule attacks every round, beginning at round 1. Set an end round
and interval for bounded or intermittent injection. Assignments remain fixed;
on inactive rounds, replacement clients train cleanly and submit an unscaled
model. Triggered ASR evaluation is enabled when either `backdoor` or
`model_replacement` is among the enabled attack types.

```text
patch_size = max(2, min(image_height, image_width) // 12)
```

A 28x28 or 32x32 image gets a 2x2 patch; a 64x64 image gets a 5x5 patch. Images smaller
than 2x2 are rejected. The helper accepts batched HWC images by default, explicit
`layout="CHW"` for channel-first batches, and `layout="HW"` for grayscale batches.
Single-channel HWC/CHW also work. It validates counts, label space, dimensions,
probabilities, pixel range and dtype. Floating images must be finite in `[0,1]`;
raw uint8 inputs use the corresponding rounded `[0,255]` value (white is 255).

The actual project uses 32x32 HWC float32 images scaled to `[0,1]`, without spatial
augmentation or channel mean/std normalization. Thus insertion after scaling is
well-defined and numeric 1.0 is white. The evaluation helper calls the same trigger
function. If preprocessing later adds crop/resize or mean/std normalization, keep
trigger insertion after final spatial transformations and before mean/std, or add
an explicit pixel-space conversion. Do not blindly stamp 1.0 into a different
normalized representation.

Both helpers optionally return their selected mask with `return_mask=True`.
`apply_attacks_for_client` dispatches exactly one configured attack and returns
images, labels, mask, and realized diagnostics in a `PoisonedData` object.

Each attack invocation starts a **local** `RandomState` with the configured attack
seed; it never resets global RNG state and never incorporates the round number.
By design, equally sized clients sharing a seed can have matching positional masks.
There is no implicit independent-per-client seed derivation. Same sample order,
seed, data and parameters reproduce the same poisoned examples. No resampling mode
is implemented.
This seed controls assignments and poisoned examples only; it does not make model
initialization or the complete training process deterministic.

## Clean evaluation and backdoor ASR

Clean `model.evaluate` still receives the untouched local validation split and
returns the existing clean loss/accuracy. Backdoor evaluation is a separate pass:

1. Start with clean held-out examples.
2. Exclude examples whose **original** class equals the target.
3. Stamp **every** remaining example (`poison_prob=1.0`) with the same trigger.
4. Predict with the received global model and count predictions equal to the target.

```text
ASR = successful target predictions / eligible non-target holdout examples
```

This reports local-holdout backdoor ASR, not official test-set ASR or ordinary clean
accuracy. Zero eligible examples produce `asr: null` and denominator zero in JSON;
no false zero-accuracy scalar is emitted. TensorBoard records the denominator and
an unavailable status instead. A previous ASR point is not a measurement for that
empty round.

Evaluation replies carry only `successes` and `eligible` in a separate ConfigRecord.
FedAvg continues aggregating the original clean MetricRecord. The server sums ASR
numerators and denominators across successful reporting clients, rather than
weighting client ASRs by unrelated validation sizes. Failed evaluation clients are
absent from this reported population. The denominator and per-client counts make
that coverage explicit. ASR runs on all evaluating clients, including benign/head
clients; it does not reveal which clients trained on poisoned data.

Ground truth assignments and poison statistics are saved for inspection, not passed
to the clustering defense. Train replies retain the same model arrays, ordinary
training metrics, and identity/schema metadata. No defense receives assignment maps
as an oracle. Optional attacks do not change how clustering scores or selection are
computed.

## Saved outputs and terminal/TensorBoard logs

Within the existing unique experiment directory:

```text
results/tensorboard/run_.../
  config.json
  attacks/
    assignment_and_poisoning.json
    training/
      client_0.npz
      client_1.npz
      ...
    client_1_poisoned_indices.npz     # optional research index artifact
    round_001_asr.json               # when separate ASR is evaluated
    model_replacement/
      per-client submitted-weight diagnostics and per-round summaries
    ...
```

The compact JSON records requested/effective malicious counts, actual malicious
fraction, protected/selected IDs, assignments, distribution quotas, effective
parameters and seed, NumPy version, per-client training size, poisoned counts and
fractions, label-change counts, original/attacked class histograms, backdoor patch
dimensions/layout/stage, and training/selected-source index digests. It also records
total training and poisoned samples and the experiment-wide realized poisoned fraction.
With `attacks-save-indices=true`, the extra index files map selected training
positions to stable row IDs in the concatenated official training batches.

`training/*.npz` overlays are required runtime artifacts even when optional index
exports are disabled: they contain the labels and selected positions needed to reuse
the same poison, plus stamped rows for backdoors. They are distinct from optional
model snapshots and must remain accessible while the experiment runs. Writes are
atomic and do not overwrite existing artifacts. No automatic deletion is introduced;
disk usage grows with selected backdoor examples and experiment count.

The server prints requested/effective counts and a per-client attack/poisoning table
before training. When model replacement is enabled, it reports prepared overlays
at startup and lists each replacement client's active state and scale after each
training round. Per-client JSON records include a digest of the submitted weights;
server summaries verify this digest against the received reply and record successful
reply and accepted aggregation weights. The server prints a separate ASR line after
each evaluated round.
TensorBoard uses the existing centralized writer:

- `attacks/assignment`: experiment ground truth and settings.
- `attacks/client_N/training_data`: per-client JSON diagnostics.
- `attacks/client_N/poisoned_fraction`, `labels_changed`, and other counts.
- `attacks/client_N/training_images` and `training_image_labels`: first five actual
  training examples after poisoning, in original partition training order.
  Image captions show the current class name/ID, original class name/ID, attack
  type, and whether that specific sample was poisoned. An assigned attack can
  leave individual preview samples clean; these are not model predictions.
- `backdoor/client_N/asr`, `eligible`, and `successes`: separate local ASR metrics
  when either backdoor or model replacement is enabled.
- `attacks/client_N/poisoned_images`: up to five actually poisoned samples,
  selected from the fixed mask in training order. These are also saved under
  `images/after_attack/client_N/`, with original and post-attack label names.
  If no samples were poisoned, no such folder is created and
  `attacks/client_N/poisoned_images_status` explains why.
- `server/backdoor_asr`, `backdoor_asr_eligible`, and `backdoor_asr_successes`: summed
  count-based ASR. Existing `server/global_accuracy` and `global_loss` remain clean.

The existing `dataset/` images and summaries explicitly describe **clean pre-poisoning**
partitions. Compare these against `attacks/` for actual training examples and attacked
class distributions. Existing model archives remain controlled by `save-client-weights`.

The same labeled previews are saved under the run's
`images/attack_training/client_N/`, with clean versions under
`images/clean_training/client_N/`. Each directory contains a `preview.png` strip,
individual PNGs named with source row and current class (plus `_from_<class>` when
the label changed), and `labels.json` with the exact sample metadata. Only the
first-five training preview selection is exported there. The separate
`after_attack` folder selects the first five poisoned positions instead;
no masks are resampled. Label-flipping previews retain the original pixels
with changed labels, while backdoor previews include the actual stamped pixels.

## Verification and limits

```bash
python -m unittest discover -s tests -p 'test_attack*.py' -v
python -m unittest discover -s tests -v
```

Tests cover exact RNG behavior, boundaries, class replacement, layouts/dtypes,
trigger geometry, immutable source/holdout data, protected heads, quotas/caps, zero
malicious clients, fixed overlays for IID/Dirichlet partitions, and ASR exclusion and
empty denominators. The synthetic two-round smoke test constructs datasets, trains
a small real Keras model through the real ClientApp handlers, and uses the existing
Flower strategy, clustering, logging, and separate ASR path. It verifies wiring and
checks that defense inputs do not carry attack ground truth. It does **not** establish
attack success or defense effectiveness, and is not a full GPU/Ray experiment.

Seed-driven assignment, mandatory head protection, the fixed trigger position/rule,
and fixed poisoning across rounds are deliberate scope choices. Full physical-client
attestation and malicious-client ground-truth authentication are outside this local
simulation. Attack effectiveness depends on data, hyperparameters, model and defense;
measure clean accuracy/loss and ASR instead of assuming either attack succeeds.
