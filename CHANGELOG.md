# Changes

## Agent Skill integration on main

Add a Claude Code/Codex skill, stdlib-only loopback doctor/judge/fixture CLI,
and safe `/health` metadata using the same pinned runtime. Service/protocol
errors remain distinct from the model's unknown class. Add mock failure tests
and public synthetic fixture checks. No model, tokenizer, prompt, temperature,
weight asset or existing v0.1.0-experimental Release change.

## v0.1.0-experimental

Continue training the pinned Kev-4B LoRA and pointer head for 50 optimizer updates
on a synthetic review/skip/unknown task. No backbone tensors were trained.
Package the selected checkpoint as safetensors and explicit JSON; remove the
candidate tokenizer copy and pin the evaluated base tokenizer. Add a fixed MLX
loader, local-only API, fail-closed manual-review behavior, synthetic data,
metrics and portable copies of the actual training scripts.

This classifies whether independent review is needed; it does not generate code
reviews. No automatic review bypass is enabled. The existing personal tools were
not changed or installed by this release.
