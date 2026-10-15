---
language: en
license: apache-2.0
base_model: Qwen/Qwen3.5-4B-Base
pipeline_tag: text-classification
---

# Kev Review Gate step50 — experimental

Classifies whether supplied change evidence needs review. It does not generate
review comments or prove code correctness. The supplied API keeps manual review
required regardless of class/confidence.

## Identity and changes

- Maintainer: dotoricode; upstream Kev code/model author: Jared Palmer.
- Kev source: `dd375f27fd61486c52204205e10a0d2e10f3b34a`.
- Initial Kev-4B: `6cfce5c2fa4b4bd64026336ab649c5ca78857d52`.
- Base and actual tokenizer: Qwen/Qwen3.5-4B-Base at
  `1001bb4d826a52d1f399e183466143f4da7b741b`.
- Original selected artifact hash:
  `e42761df4d1f32572a4052924700c58f79d89f96161e4a551716749c589be259`.
  This hashes its filename→digest mapping. Public configuration/serialization
  differs and has separate hashes; no whole-directory identity is claimed.
- Trainable LoRA32,464,896 parameters and pointer head1,311,232; frozen
  backbone4,205,751,296. Existing rank16/alpha32/all original target modules.

Selected training:50 optimizer updates, batch1/accum1,50 distinct filename rows
covering41 scenarios. One run continued to200 for scheduled dev measurements;
later weights are not released. AdamW lr/head_lr2e-5, weight_decay0.01, clip1,
seed321, global warmup10 then constant LR. bf16 frozen backbone, fp32 LoRA/head,
direct loading, non-reentrant gradient checkpointing, no autocast/quantization
or CPU fallback. Exact settings/sampler: `pilot/execution_config.json`.

Inference uses the same evaluated MLX path: bf16 backbone; LoRA merged once in
fp32 CPU arithmetic and rounded once; fp32 pointer head; eval mode, dropout
disabled; ordered-choice argmax. Temperature **2.406050072164233** is inherited,
unchanged and not recalibrated for this task. Confidence is not correctness.
Cache256MiB, strict state384/packed512, no truncation or text generation.

The public head uses safetensors, not pickle. All4 tensors were bitwise checked
against the selected head. See `provenance/reload-verification.json` for
clean-process output equivalence; the adapter tensor file is unchanged.

## Tokenizer caveat

Original-saved and selected-saved tokenizer pretokenizers differ in Unicode
mark handling. This is not merely JSON formatting and can matter on other text.
Both actual evaluations loaded the identical pinned **base tokenizer**. The
public loader explicitly uses only that tokenizer and verifies its file hashes.

Train576/dev120/final120 were checked under original-saved, selected-saved and
base tokenizers: serialized IDs, packed IDs, special IDs and question/option
positions matched on all816 inputs. This does not imply arbitrary-input
equivalence. Candidate tokenizer files are deliberately not distributed.

## Evaluation and limits

README and `results/final_metrics.json` report both row and scenario levels.
Selection preceded final input. Parent-held labels were joined only after360
frozen responses. Jev1.13.0 was a comparison, not a training teacher.

Only5 final families/15 scenarios were measured, repeated8 times each.
Synthetic/AI-audited labels are not human ground truth. No deployment study,
realistic prevalence, adversarial evaluation or repository-context validation
exists. English/Python-heavy train, language/subsystem shifts confounded with
family holdout, missing-helper and formatting-proof construction artifacts all
limit interpretation. No ordinary120-trial significance/binomial claim is
appropriate. These results do not justify automatic review skipping or a claim
of Jev parity. Confidence is not task-calibrated.

The now-public final set is consumed; future selection needs fresh holdouts.
License: Apache-2.0 for code, adaptation, adapter/head and external base as
declared at their exact revisions. See NOTICE and license evidence. Base weights
are not redistributed; no upstream endorsement is implied.
