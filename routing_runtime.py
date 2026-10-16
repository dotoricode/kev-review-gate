# SPDX-License-Identifier: Apache-2.0
"""Isolated offline MLX routing model with a separately pinned artifact contract."""
import json
import sys
from pathlib import Path
from contract import sha, validate
from routing_policy import MODEL_ID, POLICY_VERSION, questions, REVIEW_KINDS
from runtime import file_sha, verify_directory
from isolation import isolate

ROOT = Path(__file__).resolve().parent
FILES = {'adapter_config.json', 'adapter_model.safetensors', 'head.safetensors', 'model_config.json'}


def verify_routing_artifacts(base_dir, weights_dir):
    base, weights = Path(base_dir).resolve(), Path(weights_dir).resolve()
    manifest_path = ROOT / 'provenance/routing-weights.json'
    if not manifest_path.is_file() or not weights.is_dir():
        raise ValueError('routing_weights_unavailable')
    expected = json.loads(manifest_path.read_text())
    if expected.get('model') != MODEL_ID or expected.get('policy_version') != POLICY_VERSION or set(expected.get('public_files', {})) != FILES:
        raise ValueError('routing_identity_mismatch')
    if json.loads((weights / 'routing_manifest.json').read_text()) != expected:
        raise ValueError('routing_identity_mismatch')
    verify_directory(weights, expected['public_files'])
    lock = json.loads((ROOT / 'provenance/upstream-model-lock.json').read_text())
    verify_directory(base, {k.removeprefix('base/'): v for k, v in lock['files'].items() if k.startswith('base/')})
    c = json.loads((weights / 'model_config.json').read_text())
    required = {'base': lock['base_id'], 'base_revision': lock['base_revision'],
                'tokenizer_revision': lock['base_revision'], 'temperature': 2.406050072164233,
                'option_isolation': False, 'special_embeddings': False,
                'model_id': MODEL_ID, 'policy_version': POLICY_VERSION,
                'questions_sha256': sha(questions())}
    if any(type(c.get(k)) is not type(v) or c[k] != v for k, v in required.items()):
        raise ValueError('routing_identity_mismatch')
    return base, weights, c, expected


class RoutingModel:
    def __init__(self, base_dir, weights_dir):
        isolate()
        base, weights, config, manifest = verify_routing_artifacts(base_dir, weights_dir)
        sys.path.insert(0, str(ROOT / 'vendor'))
        import torch
        from safetensors.torch import load_file
        from kev.model import load_tokenizer, pad_id
        from kev.mlx_model import load_base, merge_lora, MLXDecisionModel
        import kev.mlx_model as mm
        self.tokenizer = load_tokenizer(str(base))
        mm.CACHE_LIMIT = 256 * 1024 ** 2
        lm = load_base(base)
        merge_lora(lm, weights)
        self.model = MLXDecisionModel(lm, pad_id(self.tokenizer), head_dim=256)
        self.model.head.load_state_dict(load_file(weights / 'head.safetensors'))
        self.model.eval()
        self.model.head.temperature = config['temperature']
        if self.model.dtype != 'bfloat16' or self.model.lm.training or any(m.training for m in self.model.head.modules()) or any(p.dtype != torch.float32 for p in self.model.head.parameters()):
            raise ValueError('routing_identity_mismatch')
        self.identity = {'model': MODEL_ID, 'policy_version': POLICY_VERSION,
                         'weights_sha256': sha(manifest['public_files']), 'backend': 'mlx',
                         'temperature': config['temperature'], 'base_revision': config['base_revision']}

    def predict(self, state):
        from kev.api import SystemOneRequest, to_record, to_answers
        if not isinstance(state, dict) or set(state) != {'task', 'review_kind'} or not isinstance(state['task'], str) or state['review_kind'] not in REVIEW_KINDS:
            raise ValueError('invalid_routing_input')
        q = questions()
        payload = {'state': state, 'questions': q, 'model': MODEL_ID}
        rec, meta = to_record(SystemOneRequest.model_validate(payload))
        enc = self.model.encode(self.tokenizer, rec, max_state=384, max_branch=512, strict=True)
        if len(enc['ids']) > 512:
            raise ValueError('routing_input_too_large')
        answers = to_answers([x.tolist() for x in self.model.probs(enc)], meta)
        validate(answers, q)
        return {'answers': answers, 'usage': {'input_tokens': len(enc['ids'])},
                'identity': dict(self.identity), 'input_sha256': sha({'state': state, 'questions': q})}
