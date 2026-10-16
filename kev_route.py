# SPDX-License-Identifier: Apache-2.0
"""Local, fixed-contract routing judgment; never starts a review or chooses a fallback."""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from contract import validate
from routing_policy import MODEL_ID, questions, validate_request, select

ROOT = Path(__file__).resolve().parent
LIMIT = 262144


def route(payload, predictor=None, *, base_dir=None, weights_dir=None):
    try:
        state = validate_request(payload)
        if predictor is None:
            from routing_runtime import RoutingModel
            predictor = RoutingModel(base_dir or ROOT / 'models/base',
                                     weights_dir or os.environ.get('KEV_ROUTING_WEIGHTS', ROOT / 'weights/kev-routing')).predict
        result = predictor(state)
        validate(result['answers'], questions())
        a = result['answers']['route']
        selected = select(a['choice'], payload)
        identity = result['identity']
        if identity['model'] != MODEL_ID or not isinstance(identity['weights_sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', identity['weights_sha256']):
            raise ValueError('invalid_routing_identity')
        selected.update(decision_source='kev', confidence=a['confidence'],
                        router_model=identity['model'], weights_sha256=identity['weights_sha256'])
        return {'ok': True, 'status': 'route_selected', 'model_route': selected,
                'probabilities': a['probabilities'], 'usage': result.get('usage', {}),
                'scope': 'Experimental routing recommendation; execution and model availability require caller verification.'}
    except Exception as e:
        allowed = {'invalid_routing_input', 'invalid_review_kind', 'invalid_model_catalog', 'routing_unknown',
                   'invalid_routing_response', 'recommended_profile_unavailable', 'required_profile_unavailable',
                   'routing_weights_unavailable', 'artifact_integrity_failure', 'routing_identity_mismatch',
                   'routing_input_too_large', 'invalid_routing_identity'}
        code = str(e) if str(e) in allowed else 'routing_unavailable'
        return {'ok': False, 'status': 'manual', 'model_route': None, 'error': {'code': code}}


def main(argv=None):
    from kev_gate import strict_json, GateError, Parser
    try:
        p = Parser(description=__doc__)
        p.add_argument('--input', required=True)
        p.add_argument('--base-dir', default=str(ROOT / 'models/base'))
        p.add_argument('--weights-dir', default=os.environ.get('KEV_ROUTING_WEIGHTS', str(ROOT / 'weights/kev-routing')))
        a = p.parse_args(argv)
        if a.input == '-': raw = sys.stdin.buffer.read(LIMIT + 1)
        else:
            with Path(a.input).open('rb') as f: raw = f.read(LIMIT + 1)
        if len(raw) > LIMIT: raise GateError('input_too_large')
        payload = strict_json(raw)
        result = route(payload, base_dir=a.base_dir, weights_dir=a.weights_dir)
    except Exception:
        result = {'ok': False, 'status': 'manual', 'model_route': None, 'error': {'code': 'invalid_routing_request'}}
    print(json.dumps(result, allow_nan=False))
    return 0 if result['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
