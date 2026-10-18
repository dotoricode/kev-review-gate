# Experimental joint need and Sol-effort advice

The optional `kev_gate.py discriminate --input -` command accepts the same
`task`, `review_kind`, `candidates` request as route. Supply only an approved Sol
candidate and medium/high efforts. Put requirement, changes, callers,
verification, change_scope and behavior_impact in task as `key: value` lines;
optional missing fields may be omitted. Never silently truncate.

It returns need decision/confidence/probabilities and judgment separately from
model_route and route_probabilities. `automatic_skip_enabled` is always false.
A successful need judgment can have model_route null when routing is unknown or
the recommended effort is unavailable. Preserve that need result and the safe
routing error; do not substitute an effort or drop the probabilities.

This command is opt-in. It requires a separately reviewed
`provenance/discrimination-weights.json` pin and matching v2 weights. If the
candidate fails frozen evaluation criteria, no deployment pin is published and
calling the command returns manual/unavailable. Keep the current v1 runtime
unless a reviewed configuration explicitly selects a validated candidate.

The v2 contract learns advisory need(review/skip/unknown) and Sol
medium/high/unknown together. It does not change mandatory review policy or
start a Codex review. A skip judgment is not authorization to skip. Astra is
excluded. See the repository discrimination/protocol.md for the frozen public
synthetic protocol and its scope limits; a synthetic score is not live review
quality. Compare only against actual Jev, never a text-only simulated Jev.
