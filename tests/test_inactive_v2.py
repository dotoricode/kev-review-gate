"""Merging the v2 source must not activate it or load candidate weights by default."""
import builtins
import json

import pytest


def test_legacy_doctor_does_not_import_or_select_v2(monkeypatch, capsys):
    import kev_gate
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in ('kev_discriminate', 'discrimination_runtime'):
            pytest.fail('The existing doctor command activated v2')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    monkeypatch.setattr(kev_gate, 'doctor', lambda **kwargs: {
        'ok': True, 'status': 'ready', 'model': kev_gate.MODEL, 'automatic_skip_enabled': False})
    assert kev_gate.main(['doctor']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['model'] == 'kev-review-gate-step50'
    assert result['automatic_skip_enabled'] is False


def test_missing_deployment_pin_blocks_v2_before_any_model_import(tmp_path, monkeypatch):
    import discrimination_runtime
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(('kev.', 'mlx', 'torch', 'safetensors')):
            pytest.fail('An unpromoted v2 attempted to load a model')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(discrimination_runtime, 'ROOT', tmp_path)
    monkeypatch.setattr(builtins, '__import__', guarded_import)
    with pytest.raises(FileNotFoundError) as error:
        discrimination_runtime.DiscriminationModel(tmp_path/'base', tmp_path/'weights')
    assert error.value.filename == str(tmp_path/'provenance/discrimination-weights.json')
