# SPDX-License-Identifier: Apache-2.0
"""Bind candidate identity to verified weights and a completed frozen training run, without loading tensors."""
import json
import math
import re
from pathlib import Path
from contract import sha
from discrimination import MODEL_ID, POLICY_VERSION, questions
from runtime import file_sha, verify_directory

ROOT = Path(__file__).resolve().parent
FILES = {'adapter_config.json', 'adapter_model.safetensors', 'head.safetensors', 'model_config.json'}


def require(condition, code):
    if not condition:
        raise ValueError(code)


def candidate_evidence(training_dir, step, protocol_sha256):
    require(type(step) is int and step in (0, 25, 50), 'invalid_candidate_step')
    stage = Path(training_dir) / f'stage-{step:04d}'
    weights = stage / 'weights'
    manifest = json.loads((weights / 'routing_manifest.json').read_text())
    require(manifest.get('model') == MODEL_ID and manifest.get('policy_version') == POLICY_VERSION
            and set(manifest.get('public_files', {})) == FILES, 'candidate_manifest_mismatch')
    require(all(isinstance(v, str) and re.fullmatch('[a-f0-9]{64}', v)
                for v in manifest['public_files'].values()), 'candidate_manifest_mismatch')
    verify_directory(weights, manifest['public_files'])
    config = json.loads((weights / 'model_config.json').read_text())
    lock = json.loads((ROOT / 'provenance/upstream-model-lock.json').read_text())
    pinned = {'model_id': MODEL_ID, 'policy_version': POLICY_VERSION, 'selected_step': step,
              'base': lock['base_id'], 'base_revision': lock['base_revision'],
              'tokenizer_revision': lock['base_revision'], 'questions_sha256': sha(questions())}
    require(all(type(config.get(k)) is type(v) and config[k] == v for k, v in pinned.items()),
            'candidate_config_mismatch')
    report_path = stage / 'report.json'
    report = json.loads(report_path.read_text())
    inputs = {'base_revision': lock['base_revision'], 'original_model_revision': lock['model_revision'],
              'base_files_sha256': {k.removeprefix('base/'): v for k, v in lock['files'].items() if k.startswith('base/')},
              'original_files_sha256': {n: lock['files']['kev-4b/' + n] for n in
                                       ('adapter_config.json', 'adapter_model.safetensors', 'head.pt')}}
    require(report.get('status') == 'completed' and report.get('protocol_sha256') == protocol_sha256
            and report.get('verified_inputs') == inputs and report.get('export') == manifest,
            'candidate_training_evidence_mismatch')
    if step == 0:
        require(type(report.get('step')) is int and report['step'] == 0
                and report.get('initialization') == 'pinned_original_untrained_on_routing',
                'candidate_training_step_mismatch')
    else:
        start = 0 if step == 25 else 25
        require(type(report.get('target')) is int and report['target'] == step
                and type(report.get('start')) is int and report['start'] == start,
                'candidate_training_step_mismatch')
        rows = report.get('steps', [])
        require([r.get('step') for r in rows] == list(range(start + 1, step + 1))
                and all(type(r.get('step')) is int for r in rows), 'incomplete_training_schedule')
        for key in ('loss', 'adapter_grad_norm', 'head_grad_norm'):
            require(all(type(r.get(key)) in (int, float) and math.isfinite(r[key]) and r[key] >= 0 for r in rows),
                    'invalid_training_measurements')
        require(report.get('base_frozen') is True and report.get('adapter_changed') is True
                and report.get('head_changed') is True
                and isinstance(report.get('base_sha256_before'), str)
                and re.fullmatch('[a-f0-9]{64}', report['base_sha256_before'])
                and report['base_sha256_before'] == report.get('base_sha256_after')
                and all(any(r[k] > 0 for r in rows) for k in ('adapter_grad_norm', 'head_grad_norm')),
                'training_update_evidence_required')
        files = report.get('resume_files_sha256', {})
        require(set(files) == {'checkpoint/adapter_config.json', 'checkpoint/adapter_model.safetensors',
                               'checkpoint/head.pt', 'resume.pt'}, 'training_checkpoint_evidence_required')
        verify_directory(stage, files)
        if step == 50:
            prior, _ = candidate_evidence(training_dir, 25, protocol_sha256)
            require(report.get('resume_source_report_sha256') == prior['training_report_sha256'],
                    'resume_source_evidence_mismatch')
    evidence = {'step': step, 'training_report_sha256': file_sha(report_path),
                'artifact_manifest_sha256': sha(manifest),
                'model_config_sha256': manifest['public_files']['model_config.json'],
                'protocol_sha256': protocol_sha256}
    return evidence, manifest
