# SPDX-License-Identifier: Apache-2.0
"""Offline, pinned MLX inference. No downloads, cloud clients or review bypass."""
import os,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
from isolation import isolate
from contract import validate,sha

def file_sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def verify_directory(root,expected):
    for name,digest in expected.items():
        p=root/name
        if not p.is_file() or p.is_symlink() or file_sha(p)!=digest:raise ValueError('artifact_integrity_failure')

class ReviewGate:
    def __init__(self,base_dir,weights_dir):
        isolate()  # fail closed before loading tensors; applies to this process only
        import torch
        import mlx.core as mx
        from safetensors.torch import load_file
        from kev.model import load_tokenizer,pad_id
        from kev.mlx_model import load_base,merge_lora,MLXDecisionModel
        import kev.mlx_model as mm
        lock=json.loads((ROOT/'provenance/upstream-model-lock.json').read_text())
        expected=json.loads((ROOT/'provenance/release-weights.json').read_text())['public_files']
        base=Path(base_dir).resolve();weights=Path(weights_dir).resolve()
        verify_directory(base,{k.removeprefix('base/'):v for k,v in lock['files'].items() if k.startswith('base/')})
        verify_directory(weights,expected)
        self.config=json.loads((weights/'model_config.json').read_text())
        c=self.config
        if c['base']!=lock['base_id'] or c['base_revision']!=lock['base_revision'] or c['tokenizer_revision']!=lock['base_revision'] or c['temperature']!=2.406050072164233 or c['option_isolation'] or c['special_embeddings']:raise ValueError('pinned_configuration_required')
        # Never use the saved candidate tokenizer. This is exactly the evaluated base tokenizer.
        self.tokenizer=load_tokenizer(str(base))
        mm.CACHE_LIMIT=256*1024**2
        lm=load_base(base);merge_lora(lm,weights)
        self.model=MLXDecisionModel(lm,pad_id(self.tokenizer),head_dim=256)
        self.model.head.load_state_dict(load_file(weights/'head.safetensors'))
        self.model.eval();self.model.head.temperature=c['temperature']
        if self.model.backend!='mlx' or self.model.dtype!='bfloat16' or self.model.lm.training or any(m.training for m in self.model.head.modules()) or any(p.dtype!=torch.float32 for p in self.model.head.parameters()):raise ValueError('evaluation_precision_required')
        self.identity={'model':'kev-review-gate-step50','version':'v0.1.0-experimental','backend':'mlx','base_revision':lock['base_revision'],'temperature':c['temperature'],'backbone_dtype':'bfloat16','head_dtype':'float32','automatic_skip_enabled':False}
        self.identity.update(health_schema=1,base_model_id=lock['base_id'],
            tokenizer_id=lock['base_id'],tokenizer_revision=lock['base_revision'],
            tokenizer_files_sha256={n:lock['files']['base/'+n] for n in ['tokenizer.json','tokenizer_config.json']},
            weights_files_sha256=expected,eval_mode=True,max_state_tokens=384,max_packed_tokens=512)

    def predict(self,payload):
        from kev.api import SystemOneRequest,to_record,to_answers
        if set(payload)!={'state','questions','model'} or payload['model']!='kev-review-gate-step50':raise ValueError('request_contract')
        # Single reviewed task only, never silently repurpose a calibrated head for arbitrary questions.
        contract=json.loads((ROOT/'data/manifest.json').read_text())['contract']
        q={'need':{'type':'choice','instructions':contract['question'],'criteria':{x['value']:x['description'] for x in contract['criteria']}}}
        if payload['questions']!=q or not isinstance(payload['state'],dict) or set(payload['state'])!={'synthetic_diff'} or not isinstance(payload['state']['synthetic_diff'],str):raise ValueError('fixed_task_contract')
        rec,meta=to_record(SystemOneRequest.model_validate(payload));enc=self.model.encode(self.tokenizer,rec,max_state=384,max_branch=512,strict=True)
        if len(enc['ids'])>512:raise ValueError('packed_limit')
        answers=to_answers([x.tolist() for x in self.model.probs(enc)],meta);validate(answers,q)
        return {'model':payload['model'],'answers':answers,'usage':{'input_tokens':len(enc['ids'])},'identity':self.identity,'status':'review_required','automatic_skip_enabled':False,'input_sha256':sha({'state':payload['state'],'questions':payload['questions']}),'token_ids_sha256':sha(enc['ids'])}
