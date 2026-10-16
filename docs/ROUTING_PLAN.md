# Review model routing plan

Extend this repository with a separately versioned local routing task while preserving the released review/skip/unknown task and artifacts. Baby Steps will show the planned reviewer and reasoning effort, then pass the same frozen selection to a new Codex review process.

- Select among sol_medium, sol_high, astra_medium, astra_high and unknown with a separate adapter/head and fixed question contract.
- Validate the caller-provided model catalog. Independent final and risk reviews require Astra, retaining the selected supported effort. Unknown or unavailable routes stop for manual action; no silent model fallback.
- Keep all inference local. Train on new public synthetic scenarios with train/dev/final split by semantic scenario; freeze selection before final evaluation.
- Always use live Jev as the comparison group with the same input/question/options. Jev output is a comparison, not the ground truth or an unreported training teacher.
- Report raw choice quality, policy-applied results, uncertainty, latency, usage, failures and artifact hashes separately. Synthetic policy labels do not demonstrate equal code-review quality or cost savings.
- Publish code and reproducible experiment results in this PR. Preserve existing release assets; do not check weights or credentials into Git.

Acceptance requires red/green contract checks, original task regression tests, actual local training evidence, frozen held-out Kev/Jev comparison, reload verification, and independent PR review with verified fixes. Model deployment is conditional on the declared evaluation criteria; a failed candidate remains experimental.
