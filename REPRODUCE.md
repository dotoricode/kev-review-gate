# Reproduction

## Frozen inference

Choose the source checkout for the workflow you need. The immutable
`v0.1.0-experimental` tag reproduces the original standalone inference release;
the [Claude Code/Codex skill setup](skills/kev-review-gate/references/setup.md)
requires a reviewed current main commit, which adds `/health` and `kev_gate.py`.
Both use the same frozen Release weights and pinned base/tokenizer. Updating
the skill/runtime source does not require retraining or replacing those weights.

Use the release tag, requirements.lock, pinned external base snapshot and
Release weights. README provides commands. `evaluate.py --reference
results/final_predictions.json` checks exact rounded answer objects and input
hashes against the frozen step50 responses. No API call/download occurs.
All4 converted head tensors are bitwise equal; adapter bytes are unchanged.
The loader checks weight/base/config hashes, eval mode, temperature and
bf16/fp32 precision. Other hardware/library versions are unvalidated.

## Historical training recipe

Portable copies of the actual pilot_train.py, pilot_eval.py, run_pilot.py,
pilot_metrics.py and training_support/smoke_runner.py are included. Changes
cover relative data/support/vendor paths, private dashboard replacement,
interpreter selection and error redaction. Original/public source hashes are
in provenance/source-files.json. Training mathematics/settings are unchanged.
Vendored Kev dependency files match the pinned commit except one explicitly marked docstring path redaction; do not
install the unrelated PyPI package named `kev`.

For a fresh reproduction attempt, explicitly provision the external base under
models/base and original pinned Kev adapter/config/head.pt under models/kev-4b.
Validate their file digests in provenance/upstream-model-lock.json. Initial
weights are not in this release. The public inference loader never reads
head.pt; only optional historical training consumes trusted pinned/or locally
generated trainer format. In a new clone after provisioning:

```sh
.venv/bin/python pilot_train.py --smoke --start 0 --target 2
.venv/bin/python run_pilot.py
```

Do not train just to use the release. These commands allocate substantial local
memory; MPS and process restrictions are required. No CPU/cloud fallback. No
training ran during packaging. Existing output is not overwritten.

Historical order: dev0, train0→50/dev50, resume50→100/dev100,
resume100→200/dev200. Optimizer/RNG/sampler position restore; frozen LR is
2e-5 × min(1, global_step/10), AdamW decay0.01, clip1, seed321, batch1/accum1.
All200 row indices are distinct, but variants repeat semantics. State384,
packed512, strict. Selection safety gates/tie rule are in pilot_metrics.select
and the frozen protocol. Exact MPS training bits across environments are not
promised; see the amendment for smoke/venue deviations.

The final set/labels are public, so a reproduction is not a new blind test.
New adaptation research requires new sequestered holdouts.

```sh
.venv/bin/python -m pytest tests
shasum -a 256 -c provenance/PACKAGE_SHA256SUMS
```

The package manifest excludes itself, git metadata, downloaded weights/base,
venvs and transient output. Reload verification records the clean-process
test; release notes identify the archive hash and public-download reload.
