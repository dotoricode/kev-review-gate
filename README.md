# Kev Review Gate

An experimental local classifier for whether supplied change evidence needs
independent review: `review`, `skip`, or `unknown`. It does **not generate review
comments**. The provided API always requires review; automatic skipping is disabled,
even when the model predicts `skip`.

This adapts [Jared Palmer's Kev](https://github.com/jaredpalmer/kev) released Kev-4B
LoRA and pointer head on Qwen3.5-4B-Base. No upstream affiliation or endorsement is implied.

## Download and run

Validated on Apple M5 Pro, 48 GB unified memory, macOS, Python 3.12.14. This release
requires Apple Silicon/MLX and the pinned dependencies; no CPU or cloud fallback.

1. Clone a reviewed main commit for the Agent Skill/healthcheck integration and
   install `requirements.lock` in a Python3.12 venv. The original
   `v0.1.0-experimental` tag remains the frozen standalone-inference baseline.
2. Download `kev-review-gate-step50.tar.gz` and `SHA256SUMS` from the
   [GitHub Release](https://github.com/dotoricode/kev-review-gate/releases/tag/v0.1.0-experimental).
   Run `shasum -a 256 -c SHA256SUMS`, then extract the archive under `weights/`.
3. Explicitly provision the external base/tokenizer
   [Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b](https://huggingface.co/Qwen/Qwen3.5-4B-Base/tree/1001bb4d826a52d1f399e183466143f4da7b741b)
   under `models/base`. No base weights are redistributed here. The loader verifies
   the exact file SHA256s in `provenance/upstream-model-lock.json`. No Hugging Face
   account creation or upload is part of this project. Explicit provisioning may
   use the network; inference never downloads anything.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python evaluate.py --base-dir models/base \
  --weights-dir weights/kev-review-gate-step50 \
  --data data/final_test.json --output evaluation.json \
  --reference results/final_predictions.json
.venv/bin/python serve.py --base-dir models/base \
  --weights-dir weights/kev-review-gate-step50
```

The server binds only `127.0.0.1:8019`, accepts `POST /v1/systemone`, and has no
remote endpoint option, cloud client, analytics, payload logs or fallback. macOS
per-process restrictions deny outbound networking and child processes before
loading; no host firewall/security policy is changed. This is a defense layer,
not a general sandbox for an untrusted host.

Requests use `model: "kev-review-gate-step50"`, `state: {"synthetic_diff": "..."}`,
and exactly the `need` question/ordered criteria in `data/manifest.json`.
Unknown contracts, artifact mismatch, model errors and overlong inputs fail
closed to manual review. State384/packed512 tokens, strict with no truncation.
Successful responses also say `status: "review_required"` and
`automatic_skip_enabled: false`.

Localizing this classifier does not localize a later Codex/Claude reviewer.
A downstream workflow sending real diffs to cloud reviewers still transmits
those diffs. This repository does not read repositories or launch reviewers.

## Claude Code and Codex Agent Skill

The [dedicated skill](skills/kev-review-gate/SKILL.md) uses the fixed local
runtime for advisory review-need judgments. [Setup and manual installation](skills/kev-review-gate/references/setup.md)
cover both agents, project/personal discovery, explicit foreground startup and
updates without overwriting another skill. Nothing installs or starts automatically.
Use an approved agent executor on the server's machine; cloud/remote loopback
does not reach a service on your Mac.

Set `KEV_REVIEW_GATE_REPO` to this reviewed clone in your local session:

```sh
python3 "$KEV_REVIEW_GATE_REPO/kev_gate.py" doctor
python3 "$KEV_REVIEW_GATE_REPO/kev_gate.py" fixture
python3 "$KEV_REVIEW_GATE_REPO/kev_gate.py" judge --input "$STATE_FILE"
```

`STATE_FILE` is UTF-8 JSON with exactly `synthetic_diff` as a string; `--input -`
reads stdin. Write evidence with file tools, never interpolate code into shell
arguments. The CLI retains the original fixed question/ordered criteria. It
checks health and response identity against pinned model/base/tokenizer hashes,
temperature and precision before accepting a judgment. It accepts only the
literal `http://127.0.0.1:8019`, ignores proxy environment, refuses redirects and
has no cloud fallback or automatic retries. Errors return nonzero exit status,
`ok:false`, `status:manual`, `decision:null`; semantic `unknown` is a successful
model answer. Doctor self-report is not malicious-process attestation.

The Agent Skill does not expand company-data authorization. Approved Claude or
Codex environments may prepare permitted evidence; this implementation was
tested only with public synthetic fixtures. Real raw git diffs, summaries and
new domains are unvalidated and do not inherit the published synthetic scores.
Automatic review skipping remains disabled. The existing weight assets and
`v0.1.0-experimental` Release are unchanged by this integration.

## Measured experiment

These are synthetic designer labels independently AI-audited, **not human ground
truth or production accuracy**. Each scenario has8 correlated filename variants.

| Frozen model | Final semantic scenarios | Final filename rows |
|---|---:|---:|
| Original Kev-4B | 14/15 | 111/120 |
| Selected step50 | 15/15 | 120/120 |
| Jev1.13.0 reference | 15/15 | 120/120 |

Original Kev returned `unknown` for all8 `http_redirect_code-s1` review variants;
step50 returned `review`. Original also made one `unknown`→`skip` error on
`disk_permission-s3-07`. Scenario plurality hid that single-variant error;
step50 matched the synthetic label. Small synthetic scores do not establish
Jev equivalence, real-code competence or safe unattended review skipping.

Dev scenario scores were11/15 at0 and15/15 at50/100/200. Frozen safety gates and
the earliest-checkpoint tie rule selected50 **before final input arrived**.
Selected exposure: **50 rows/41 semantic scenarios**; entire run:200 rows/69
scenarios. The available train split is576 rows/72 scenarios. Earlier separate
120-case scores (including80/120) are a different dataset and must not be used
as an improvement comparison against this120/120.

The final benchmark is now public and consumed, not a future unseen holdout.
No further selection, training or calibration followed final evaluation.

See [model card](MODEL_CARD.md), [data card](DATA_CARD.md),
[reproduction](REPRODUCE.md), [protocol amendment](PROTOCOL_AMENDMENT.md),
`results/`, `provenance/`, `LICENSE`, `NOTICE` and `licenses/`.
Weights are Release assets, never Git blobs. Base weights, candidate tokenizer,
optimizer/resume states, credentials, private integrations and raw system logs
are excluded.

## Experimental reviewer routing

A separate `kev-review-router-v1` LoRA/head selects a review profile; it does not repurpose the released review-need `step50` weights. `kev_gate.py route --input - --base-dir <pinned-base> --weights-dir <routing-weights>` accepts `task`, `review_kind`, and a caller-verified candidate catalog. The independent-final/risk policy requires Astra, preserving the supported effort. Unknown, unsupported profiles and unavailable artifacts return a manual error, never a default model or a cloud fallback. Inference is offline and starts no reviewer.

The caller must discover available models, freeze the selected model/effort in its review job, show that selection, and pass the same values to the review executor. See [the routing contract](skills/kev-review-gate/references/routing.md). Routing requires separate local artifacts matching `provenance/routing-weights.json`; these weights are not the existing review-gate Release and are not committed into Git or silently downloaded. The selected local training artifact is preserved with the implementation run; reproducing training uses the pinned base and original Kev adapter/head.

The frozen pilot trained 50 updates and selected step25 using dev only. On 20 new synthetic final cases, selected Kev matched 16/20 policy labels and actual Jev matched 14/20; both had 0/4 original high-risk underallocations. Jev is always the comparison group on identical public synthetic state/questions, not a training teacher or ground truth. This small experiment does not establish superior code review, real-world reliability, or savings. Kev's unknown and sol_high recalls were each 2/4, so routing remains experimental. Loading time is excluded from per-request latency measurements. [Protocol](routing/protocol.md), [results and limitations](routing/results/report.md).
