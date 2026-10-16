# SPDX-License-Identifier: Apache-2.0
"""Fixed routing judgment contract and deterministic independent-review guard."""
import hashlib
import json
import re

POLICY_VERSION = 'kev-routing-v1'
MODEL_ID = 'kev-review-router-v1'
REVIEW_KINDS = ('checkpoint', 'independent_final', 'risk_review')
CHOICES = ('sol_medium', 'sol_high', 'astra_medium', 'astra_high', 'unknown')


def questions():
    return {'route': {'type': 'choice',
            'instructions': 'Choose the model tier and reasoning effort needed for an independent code review. Judge complexity, risk and uncertainty from the task. Use unknown when evidence is insufficient.',
            'criteria': {
                'sol_medium': 'Routine bounded change with clear requirements and low risk.',
                'sol_high': 'Moderate complexity needing careful tracing across code paths.',
                'astra_medium': 'Complex or consequential change requiring frontier judgment.',
                'astra_high': 'High risk, deep uncertainty, security or broad architectural review.',
                'unknown': 'Insufficient or conflicting evidence to recommend a review profile.'}}}


def validate_request(payload):
    if not isinstance(payload, dict) or set(payload) != {'task', 'review_kind', 'candidates'}:
        raise ValueError('invalid_routing_input')
    task = payload['task']
    if not isinstance(task, str) or not task.strip() or len(task.encode()) > 16384:
        raise ValueError('invalid_routing_input')
    if payload['review_kind'] not in REVIEW_KINDS:
        raise ValueError('invalid_review_kind')
    candidates = payload['candidates']
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 2:
        raise ValueError('invalid_model_catalog')
    seen = set(); models = set()
    for c in candidates:
        if not isinstance(c, dict) or set(c) != {'model', 'tier', 'efforts'}:
            raise ValueError('invalid_model_catalog')
        if c['tier'] not in ('sol', 'astra') or c['tier'] in seen:
            raise ValueError('invalid_model_catalog')
        if not isinstance(c['model'], str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}', c['model']) or c['model'] in models:
            raise ValueError('invalid_model_catalog')
        if not isinstance(c['efforts'], list) or not c['efforts'] or any(e not in ('minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra') for e in c['efforts']) or len(c['efforts']) != len(set(c['efforts'])):
            raise ValueError('invalid_model_catalog')
        seen.add(c['tier']); models.add(c['model'])
    return {'task': task, 'review_kind': payload['review_kind']}


def select(choice, payload):
    if choice == 'unknown':
        raise ValueError('routing_unknown')
    if choice not in CHOICES:
        raise ValueError('invalid_routing_response')
    tier, effort = choice.split('_')
    catalog = {c['tier']: c for c in payload['candidates']}
    if tier not in catalog or effort not in catalog[tier]['efforts']:
        raise ValueError('recommended_profile_unavailable')
    selected_tier = 'astra' if payload['review_kind'] in ('independent_final', 'risk_review') else tier
    if selected_tier not in catalog or effort not in catalog[selected_tier]['efforts']:
        raise ValueError('required_profile_unavailable')
    canonical = sorted([dict(c, efforts=sorted(c['efforts'])) for c in payload['candidates']], key=lambda c: c['tier'])
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'recommended_model': catalog[tier]['model'], 'selected_model': catalog[selected_tier]['model'],
            'effort': effort, 'choice': choice, 'policy_version': POLICY_VERSION,
            'policy_override': selected_tier != tier, 'catalog_hash': digest}
