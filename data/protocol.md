# Preregistered local synthetic fine-tuning experiment

Status: FROZEN after independent AI consistency audit, conditional pass for a limited synthetic proof-of-learning pilot. This is not a human audit. Dataset and protocol hashes are recorded in manifest.json before training. No promise of improvement or of a monotonic learning curve.

## Question and scope
Does a small supervised adaptation improve Kev's evidence-grounded review/skip/unknown gate decisions relative to its frozen untuned checkpoint, on newly authored, family-heldout synthetic cases? Jev is a frozen reference, not a training teacher. All training and inference must run in the authorized local container, with no paid APIs, employer data, user computer, or cloud training service. Any required new permission or unavailable compute is a blocker, not permission to switch venues.

Use exactly the question, choice type, and ordered review/skip/unknown criteria copied from the existing 120-case contract. The only task state is `synthetic_diff`. The model input must exclude expected label, rationale, ID, family, scenario, split, variant index, manifest, and audit. Trainer output targets may contain the expected class in the runtime's exact choice response contract; record that serialization before training. Do not teach a different task such as producing free-form rationales. Use identical inference prompt, parser, decoding, generation limit, and runtime choice interface for base and adapted Kev. Jev uses the same task contract through its already frozen runtime.

## Dataset and independence
- Train: 576 rows = 24 source/semantic families × 3 contrasting labeled scenarios × 8 neutral filename variants; 192 per class
- Development: 120 rows = 5 other families × 3 scenarios × 8 filename variants; 40 per class
- Final: 120 rows = 5 further families × 3 scenarios × 8 filename variants; 40 per class
- Effective distinct semantic scenarios: 72 train, 15 dev, 15 final. These rows are repeated exposure, not 816 independently constructed examples. Filename variation is not necessary for supervision and adds no semantic diversity; it is retained only as a neutral surface nuisance. Filenames carry no family or gold-label words. A balanced all-class sampler must not imply more semantic diversity.
- Whole families are split before execution. Review, skip, and unknown contrasts exist within each family. Review examples establish a change; unknown examples genuinely omit needed helper or configuration semantics; skip examples prove text/layout-only edits.
- Synthetic author labels, independently AI-audited, are not human ground truth. General formatting and missing-helper patterns necessarily recur across splits. This tests heldout-family transfer within a constructed task, not unseen-concept generalization.
- English evidence, Python-heavy training (504/576 rows), more heterogeneous dev/final languages. This is a deliberate language and subsystem shift, confounded with family holdout. Do not infer natural-code performance, real-world prevalence, or statistical independence from these balanced scores.
- The original 120 cases, previously examined by the team, are regression-only. Do not mix them into training, select checkpoints on them, or call them an unseen test.
- No source state is an exact duplicate across new splits or matches the original suite. Within-family filename variants are intentional semantic duplicates and remain in one split. Record these two distinct duplicate notions explicitly.

## Final-test sequestration
Only dataset author and independent auditor may inspect final_test.json before checkpoint selection. Filesystem sharing is a procedural boundary, not a security sandbox. The training executor may read train.json, dev.json, and this protocol; it must not read final_test.json, build_dataset.py, audit details containing final examples, or search the containing directory for final examples. The parent must not forward final-family content. Manifest may expose counts/hashes/language distribution, never example contents. Do not import the dataset generator in training. If final contents reach selection or prompt tuning, record contamination and author a replacement holdout before any final performance claim.

## Freeze and smoke
Before substantive training, record model identifier, original model/checkpoint hash, tokenizer revision, precision/quantization, adaptation method, exact trainable modules, optimizer, learning rate, scheduler, batch size, accumulation, seed, maximum input/output lengths, inference settings, available hardware, dependency versions, dataset hashes, and target serialization in execution_config.json. Dataset author does not choose unsupported hardware details. Use seed 321 unless the runtime cannot support it, in which case record the limitation before execution.

Smoke test can use a maximum of 12 train rows (four per class) and at most two optimizer updates. Its only purpose is confirming loss, finite gradients, target masking, parser compatibility, context coverage, serialization, memory use, and runtime. Discard smoke weights and reset optimizer and checkpoint before the real run. Do not inspect dev/final performance during smoke. Tune memory settings only to make this one configuration executable, recording changes before substantive training. If no viable training path exists, report that result rather than substitute prompt tuning.

## Budget and checkpoints
One training run, no hyperparameter sweep. Nominal checkpoints are optimizer updates 0, 50, 100, and 200. Evaluate every checkpoint on the same dev set with fixed deterministic decoding. Record exact effective exposure/epochs: step counts are not comparable without batch size. If smoke timing makes 200 updates infeasible within available compute, publish an amendment BEFORE dev evaluation or substantive training selecting a single scaled schedule of 0/10/25/50 or 0/5/10/20 updates. No schedule changes based on scores. If training diverges or the runtime fails, stop, preserve logs, and identify the failure; do not silently replace the experiment with another run. Technical recovery is allowed only without selecting a different configuration on dev performance and must be documented.

## Development selection, fixed before execution
For each checkpoint, report raw rows and scenario-aggregated predictions. Aggregate each scenario's 8 variants by plurality over review/skip/unknown/invalid; a tie among valid classes becomes unknown, while any tied top count involving invalid becomes invalid. An invalid plurality remains invalid. Report variant disagreement and invalid outputs; invalid outputs remain a fourth prediction/error class, not silently converted to a correct unknown. Compute macro-F1 over the three gold classes with zero_division=0; invalid predictions are false negatives for their gold class.

Primary selection uses the 15 scenario-aggregated dev outcomes (five per gold class), not 120 correlated rows. Baseline is step 0. A trained checkpoint is eligible only if:
1. Its review→skip and unknown→skip counts each do not exceed step 0
2. Its macro-F1 and review recall are each at least step 0's
3. Its skip recall and unknown recall are each at least 0.40 (two of five scenarios)
4. Its raw-row invalid-output rate is no greater than step 0's
These constraints prevent an all-unknown or all-review solution from winning by avoiding skip errors alone.

Include step 0 unconditionally as a selection candidate, regardless of the trained-checkpoint eligibility gates. Among step 0 and eligible trained checkpoints, choose the lexicographically smallest tuple:
(total review→skip plus unknown→skip, review→skip, negative macro-F1, negative balanced accuracy, optimizer step).
If no trained checkpoint qualifies, choose step 0. Equal results favor the earlier checkpoint. The main goal is fewer unsafe skips, then balanced class performance. Freeze selected_checkpoint.json with artifact hash/path and dev metrics before unlocking the final set. Do not select on final or original regression scores. If step 0 wins, explicitly say training did not produce an eligible improvement; do not call step 0 a trained model.

## One final comparison
After selection is fixed, run frozen base Kev, the selected Kev checkpoint, and frozen Jev once each on the same final input set. If step 0 wins, its predictions can be reused for the selected-baseline row and there is no trained-win claim. Evaluate the original 120 cases only afterward as secondary regression; no further tuning or re-selection. Retry only missing/failed technical requests with the same input/configuration; log attempts. No resampling to find favorable outputs.

Report for each model:
- Class confusion matrix, per-class precision/recall/F1, macro-F1, balanced accuracy, accuracy
- Review→skip and unknown→skip counts/rates separately; review→unknown and abstention rate
- Invalid outputs, latency and peak memory where measurable, complete-case coverage
- Row-level descriptive metrics and the primary 15 scenario-aggregated metrics, plus per-family results and within-scenario consistency
- Paired base-vs-selected changes on each scenario, and descriptive family-level changes (only five final families)

Do not attach an ordinary 120-observation binomial interval or claim statistical significance. With only five heldout source families and fifteen constructed scenarios, uncertainty is substantial. Any optional uncertainty analysis must resample complete families and be explicitly exploratory, not evidence of production reliability. Report unfavorable or flat outcomes. A checkpoint curve may rise, fall, or trade safety against coverage; show measured points without claiming an upward trajectory.
