# SPDX-License-Identifier: Apache-2.0
"""Joint review-need and Sol-only effort contract. No execution authorization."""
MODEL_ID = 'kev-review-discriminator-v2'
POLICY_VERSION = 'kev-discrimination-v2'

def questions():
    return {
      'need': {'type': 'choice', 'instructions': 'Does this change need independent code review? Judge changed behavior, consequences and evidence. Security topic words alone do not make an inert text edit risky. Review kind never overrides evidence.',
        'criteria': {'review': 'Executable behavior, contract, security policy or lifecycle changes need review even when tests pass.',
                     'skip': 'Verified inert text, layout or comments only; no runtime behavior or security policy changes.',
                     'unknown': 'Missing or contradictory evidence prevents establishing changed behavior or verification.'}},
      'route': {'type': 'choice', 'instructions': 'Recommend Sol reasoning effort if an independent review runs. Judge actual changed behavior, cross-path complexity and consequence. Do not infer risk from the component name. Use unknown for insufficient or contradictory evidence.',
        'criteria': {'sol_medium': 'Inert edit or bounded single-path behavior change with clear contracts and verification.',
                     'sol_high': 'Changed sensitive enforcement, memory safety, irreversible data or concurrent cross-path behavior needs deep tracing.',
                     'unknown': 'Insufficient or contradictory behavior, scope or verification evidence.'}}}
