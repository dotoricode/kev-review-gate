# SPDX-License-Identifier: Apache-2.0
"""Explicitly pinned, offline joint classifier. It cannot skip or execute reviews."""
import json, sys
from pathlib import Path
from contract import sha, validate
from discrimination import MODEL_ID, POLICY_VERSION, questions
from runtime import verify_directory
from isolation import isolate
ROOT=Path(__file__).resolve().parent
FILES={'adapter_config.json','adapter_model.safetensors','head.safetensors','model_config.json'}

class DiscriminationModel:
    def __init__(self,base_dir,weights_dir,*,manifest_path=None):
        base,weights=Path(base_dir).resolve(),Path(weights_dir).resolve()
        # Production has a repository pin; experiment eval supplies an explicitly frozen candidate manifest.
        expected=json.loads(Path(manifest_path or ROOT/'provenance/discrimination-weights.json').read_text())
        if expected.get('model')!=MODEL_ID or expected.get('policy_version')!=POLICY_VERSION or set(expected.get('public_files',{}))!=FILES:raise ValueError('discrimination_identity_mismatch')
        if json.loads((weights/'routing_manifest.json').read_text())!=expected:raise ValueError('discrimination_identity_mismatch')
        verify_directory(weights,expected['public_files'])
        lock=json.loads((ROOT/'provenance/upstream-model-lock.json').read_text())
        verify_directory(base,{k.removeprefix('base/'):v for k,v in lock['files'].items() if k.startswith('base/')})
        cfg=json.loads((weights/'model_config.json').read_text())
        required={'model_id':MODEL_ID,'policy_version':POLICY_VERSION,'base':lock['base_id'],'base_revision':lock['base_revision'],'tokenizer_revision':lock['base_revision'],'questions_sha256':sha(questions()),'temperature':2.406050072164233,'option_isolation':False,'special_embeddings':False}
        if any(type(cfg.get(k)) is not type(v) or cfg[k]!=v for k,v in required.items()):raise ValueError('discrimination_identity_mismatch')
        isolate();sys.path.insert(0,str(ROOT/'vendor'))
        from kev.model import load_tokenizer,pad_id
        from kev.mlx_model import load_base,merge_lora,MLXDecisionModel
        from safetensors.torch import load_file
        import kev.mlx_model as mm
        mm.CACHE_LIMIT=256*1024**2
        self.tokenizer=load_tokenizer(str(base));lm=load_base(base);merge_lora(lm,weights)
        self.model=MLXDecisionModel(lm,pad_id(self.tokenizer),head_dim=256)
        self.model.head.load_state_dict(load_file(weights/'head.safetensors'));self.model.head.temperature=cfg['temperature'];self.model.eval()
        self.identity={'model':MODEL_ID,'policy_version':POLICY_VERSION,'weights_sha256':sha(expected['public_files'])}
        self.manifest=expected

    def predict(self,state):
        from kev.api import SystemOneRequest,to_record,to_answers
        if not isinstance(state,dict) or set(state)!={'task','review_kind'}:raise ValueError('invalid_discrimination_input')
        record,meta=to_record(SystemOneRequest.model_validate({'state':state,'questions':questions(),'model':MODEL_ID}))
        encoded=self.model.encode(self.tokenizer,record,max_state=384,max_branch=512,strict=True)
        if len(encoded['ids'])>512:raise ValueError('discrimination_input_too_large')
        answers=to_answers([x.tolist() for x in self.model.probs(encoded)],meta);validate(answers,questions())
        return {'answers':answers,'identity':self.identity,'usage':{'input_tokens':len(encoded['ids'])},'input_sha256':sha({'state':state,'questions':questions()})}
