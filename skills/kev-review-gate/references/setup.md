# Setup, explicit start and update

This integration is for local Claude Code/Codex agent execution. A browser-only
chat, cloud worker or remote shell cannot reach this machine's loopback. Never
expose/tunnel the server to make that work. Use an approved local executor.

Set `KEV_REVIEW_GATE_REPO` in the local session to an absolute reviewed clone
path. No credentials are needed. Provision the pinned dependencies and release
weights/base as described in that clone's README. The existing experimental
Release is unchanged; use current main for the new healthcheck and skill.

When the user explicitly requests starting the service, show/run this foreground
command using their existing configured paths:

```sh
"$KEV_REVIEW_GATE_REPO/.venv/bin/python" "$KEV_REVIEW_GATE_REPO/serve.py" \
  --base-dir "$BASE_DIR" --weights-dir "$WEIGHTS_DIR"
```

Do not guess paths, replace another listener, install a launch agent or create
a daemon. If8019 is occupied, run doctor; stop on incompatible identity and
report it. Startup applies the existing process restriction, not a host security
change. Ctrl-C stops a manually started foreground service. The skill does not
start/stop services automatically.

Run doctor then fixture explicitly. Fixture uses3 public synthetic cases,
compares rounded frozen answer objects and is not a new accuracy benchmark.
Its actual inference calls must be reported separately from mock tests.

## Install manually, without overwriting another skill

Review the code/commit first. The following are optional commands for the user
to execute. Nothing installs automatically. Choose the relevant agent and
project or personal scope. If a destination exists (including a broken symlink),
stop and inspect it rather than replace it. No global settings edit is needed.

Codex project scope (from the intended authorized project root):

```sh
mkdir -p .agents/skills
test ! -e .agents/skills/kev-review-gate && test ! -L .agents/skills/kev-review-gate && \
  ln -s "$KEV_REVIEW_GATE_REPO/skills/kev-review-gate" .agents/skills/kev-review-gate
```

Claude Code project scope:

```sh
mkdir -p .claude/skills
test ! -e .claude/skills/kev-review-gate && test ! -L .claude/skills/kev-review-gate && \
  ln -s "$KEV_REVIEW_GATE_REPO/skills/kev-review-gate" .claude/skills/kev-review-gate
```

Personal alternatives are `~/.agents/skills/kev-review-gate` for Codex and
`~/.claude/skills/kev-review-gate` for Claude Code, with the same no-overwrite
checks. Restart/reload the agent if its skill list has not refreshed. Invoke
`$kev-review-gate` in Codex or `/kev-review-gate` in Claude Code. For example:
“Use the local Kev review gate for this bounded change; keep normal review.”

For a copied skill, copy the complete skill folder including references and
retain the independent local runtime clone/env variable. To update, first check
`git status --short` in the runtime clone and preserve changes; fetch/review the
new commit, fast-forward only when clean, then rerun doctor/fixture. A symlink
tracks the reviewed clone; a copied skill needs explicit comparison/update.
Do not move the model Release/tag or overwrite another project's integrations.

Official discovery references:
[Codex](https://learn.chatgpt.com/docs/build-skills) and
[Claude Code](https://code.claude.com/docs/en/skills). Enterprise/admin policy may
restrict discovery or execution. A Team account's data authorization does not
make its runtime local; verify where commands execute.
