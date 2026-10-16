import json
import subprocess
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def request(kind='checkpoint'):
    return {'task': 'Review a small local change', 'review_kind': kind,
            'candidates': [{'model': 'current-sol', 'tier': 'sol', 'efforts': ['medium', 'high']},
                           {'model': 'current-astra', 'tier': 'astra', 'efforts': ['medium', 'high']}]}


def answer(choice='sol_medium'):
    choices = ['sol_medium', 'sol_high', 'astra_medium', 'astra_high', 'unknown']
    return {'answers': {'route': {'type': 'choice', 'choice': choice, 'confidence': 1.,
                                 'probabilities': {c: float(c == choice) for c in choices}}},
            'usage': {'input_tokens': 100},
            'identity': {'model': 'kev-review-router-v1', 'weights_sha256': 'a' * 64}}


def test_route_maps_only_current_catalog_and_preserves_effort():
    from kev_route import route
    result = route(request(), predictor=lambda state: answer('sol_high'))
    assert result['ok'] is True
    assert result['model_route']['selected_model'] == 'current-sol'
    assert result['model_route']['effort'] == 'high'
    assert result['model_route']['decision_source'] == 'kev'


@pytest.mark.parametrize('kind', ['independent_final', 'risk_review'])
def test_final_and_risk_force_astra(kind):
    from kev_route import route
    result = route(request(kind), predictor=lambda state: answer())
    assert result['model_route']['recommended_model'] == 'current-sol'
    assert result['model_route']['selected_model'] == 'current-astra'
    assert result['model_route']['policy_override'] is True
    assert result['model_route']['effort'] == 'medium'


def test_unknown_and_unavailable_effort_fail_closed():
    from kev_route import route
    assert route(request(), predictor=lambda state: answer('unknown'))['ok'] is False
    r = request(); r['candidates'][0]['efforts'] = ['medium']
    assert route(r, predictor=lambda state: answer('sol_high'))['ok'] is False


@pytest.mark.parametrize('mutation', [
    lambda p: p['candidates'].append(p['candidates'][0]),
    lambda p: p.update(review_kind='other'),
    lambda p: p.update(task=''),
    lambda p: p['candidates'][0].update(model='--danger'),
    lambda p: p['candidates'][0].update(efforts=['invalid']),
])
def test_invalid_input_never_calls_model(mutation):
    from kev_route import route
    p = request(); mutation(p)
    def forbidden(state):
        raise AssertionError('model was called')
    assert route(p, predictor=forbidden)['ok'] is False


def test_model_receives_only_fixed_state():
    from kev_route import route
    states = []
    route(request(), predictor=lambda state: states.append(state) or answer())
    assert states == [{'task': 'Review a small local change', 'review_kind': 'checkpoint'}]


def test_route_cli_invalid_json_has_typed_error():
    r = subprocess.run([sys.executable, str(ROOT / 'kev_gate.py'), 'route', '--input', '-'],
                       input='{"task":1,"task":2}', text=True, capture_output=True)
    assert r.returncode == 2
    assert json.loads(r.stdout)['ok'] is False


def test_inference_failure_never_returns_fallback_or_internal_details():
    from kev_route import route
    def broken(state):
        raise RuntimeError('private error detail')
    result = route(request(), predictor=broken)
    assert result == {'ok': False, 'status': 'manual', 'model_route': None,
                      'error': {'code': 'routing_unavailable'}}


def test_malformed_answer_fails_closed():
    from kev_route import route
    for output in [None, {'answers': None}, answer('sol_medium') | {'identity': {'model': 'wrong', 'weights_sha256': 'g' * 64}}]:
        assert route(request(), predictor=lambda state: output)['ok'] is False


def test_missing_artifact_never_loads_tensors(tmp_path, monkeypatch):
    import routing_runtime as runtime
    monkeypatch.setattr(runtime, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='routing_weights_unavailable'):
        runtime.verify_routing_artifacts(tmp_path, tmp_path)


def routing_artifacts(tmp_path, monkeypatch):
    import routing_runtime as runtime
    from routing_policy import MODEL_ID, POLICY_VERSION, questions
    from contract import sha
    monkeypatch.setattr(runtime, 'ROOT', tmp_path)
    provenance = tmp_path / 'provenance'; provenance.mkdir()
    weights = tmp_path / 'weights'; weights.mkdir()
    lock = {'base_id': 'pinned-base', 'base_revision': 'pinned-revision', 'files': {}}
    (provenance / 'upstream-model-lock.json').write_text(json.dumps(lock))
    config = dict(base='pinned-base', base_revision='pinned-revision', tokenizer_revision='pinned-revision',
                  temperature=2.406050072164233, option_isolation=False, special_embeddings=False,
                  model_id=MODEL_ID, policy_version=POLICY_VERSION, questions_sha256=sha(questions()))
    for name in runtime.FILES:
        (weights / name).write_text(json.dumps(config) if name == 'model_config.json' else '{}')
    manifest = {'model': MODEL_ID, 'policy_version': POLICY_VERSION,
                'public_files': {name: runtime.file_sha(weights / name) for name in sorted(runtime.FILES)}}
    (weights / 'routing_manifest.json').write_text(json.dumps(manifest))
    (provenance / 'routing-weights.json').write_text(json.dumps(manifest))
    return runtime, weights, provenance, manifest


def test_verified_artifacts_require_pinned_digest(tmp_path, monkeypatch):
    runtime, weights, _, _ = routing_artifacts(tmp_path, monkeypatch)
    assert runtime.verify_routing_artifacts(tmp_path, weights)[1] == weights
    (weights / 'head.safetensors').write_text('tampered')
    with pytest.raises(ValueError, match='artifact_integrity_failure'):
        runtime.verify_routing_artifacts(tmp_path, weights)


def test_untrusted_manifest_and_old_task_configuration_rejected(tmp_path, monkeypatch):
    runtime, weights, provenance, manifest = routing_artifacts(tmp_path, monkeypatch)
    manifest['model'] = 'old-gate'
    (weights / 'routing_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='routing_identity_mismatch'):
        runtime.verify_routing_artifacts(tmp_path, weights)
    manifest['model'] = 'kev-review-router-v1'
    config_path = weights / 'model_config.json'
    config = json.loads(config_path.read_text()); config['questions_sha256'] = 'old-contract'
    config_path.write_text(json.dumps(config))
    manifest['public_files']['model_config.json'] = runtime.file_sha(config_path)
    for path in [weights / 'routing_manifest.json', provenance / 'routing-weights.json']:
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='routing_identity_mismatch'):
        runtime.verify_routing_artifacts(tmp_path, weights)

@pytest.mark.parametrize('kind',['independent_final','risk_review'])
def test_explicit_sol_only_catalog_does_not_force_excluded_astra(kind):
    from kev_route import route
    payload=request(kind);payload['candidates']=payload['candidates'][:1]
    result=route(payload,predictor=lambda state:answer('sol_high'))
    assert result['ok'] is True
    assert result['model_route']['selected_model']=='current-sol'
    assert result['model_route']['effort']=='high'
    assert result['model_route']['policy_override'] is False
    # A raw recommendation for an excluded tier still requires manual handling.
    result=route(payload,predictor=lambda state:answer('astra_high'))
    assert result['ok'] is False
    assert result['error']['code']=='recommended_profile_unavailable'
