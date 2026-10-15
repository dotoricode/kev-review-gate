# SPDX-License-Identifier: Apache-2.0
# Modified for public packaging: relative data/vendor paths, local-only helpers, no dashboard dependency. Training math unchanged.
"""One resumed LoRA+pointer-head run; separate processes keep training/MLX peaks apart."""
import sys,os,json,hashlib,time,argparse,dataclasses,random
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'training_support'))
import smoke_runner as s  # canonical Kev imports, offline env and original Seatbelt isolation
import torch
from kev.checkpoint import Checkpoint,write_meta
from kev.model import DecisionModel,load_tokenizer
from kev.data import materialize
from kev.train import question_loss
CFG=json.loads((ROOT/'pilot/execution_config.json').read_text())
def records():
    rows=json.loads((ROOT/'training-input/train.json').read_text())['cases'];out=[]
    for c in rows:
        out.append({'state':{'synthetic_diff':c['state']['synthetic_diff']},'questions':{'need':{'type':c['type'],'instructions':c['question'],'criteria':{x['value']:x['description'] for x in c['criteria']},'label':c['expected'],'src':'new_synthetic_pilot'}}})
    return rows,out
def cpu_tree(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu()
    if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_tree(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
    return x
def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--start',type=int,default=0);p.add_argument('--target',type=int,default=2);a=p.parse_args()
    if not torch.backends.mps.is_available():raise RuntimeError('MPS_required_no_CPU_fallback')
    for filename,digest in CFG['dataset_hashes'].items():assert hashlib.sha256((ROOT/'data'/filename).read_bytes()).hexdigest()==digest
    out=ROOT/'pilot'/('protocol-smoke' if a.smoke else f'stage-{a.target:04d}')
    out.mkdir(exist_ok=False)
    ckout=out/'checkpoint' if a.smoke else ROOT/'pilot/checkpoints'/f'step-{a.target:04d}'
    if ckout.exists():raise RuntimeError('checkpoint_exists_refusing_overwrite')
    if a.smoke:
        assert a.start==0 and a.target==2
    else:
        schedule=json.loads((ROOT/'pilot/schedule.json').read_text())['schedule']
        assert a.start in schedule and a.target in schedule and schedule.index(a.target)==schedule.index(a.start)+1
    s.OUT=out;s.START=time.perf_counter();s.REPORT={'kind':'protocol_smoke' if a.smoke else 'substantive_pilot_segment','board_name':'본학습 protocol smoke' if a.smoke else '본학습 pilot','status':'running','steps_requested':a.target,'steps_completed':a.start,'start_step':a.start,'data_sha256':CFG['dataset_hashes']['train.json'],'model_revision':CFG['original_model_revision'],'source_revision':CFG['source_revision'],'config_sha256':hashlib.sha256((ROOT/'pilot/execution_config.json').read_bytes()).hexdigest(),'events':[],'steps':[],'checks':{},'cpu_fallback':False}
    s.event('segment_preflight',memory=s.snapshot())
    import threading
    threading.Thread(target=s.monitor,daemon=True).start()
    torch.set_num_threads(2);torch.mps.set_per_process_memory_fraction(.60);torch.manual_seed(CFG['seed'])
    source=ROOT/'models/kev-4b' if a.start==0 else ROOT/'pilot/checkpoints'/f'step-{a.start:04d}'
    ck=Checkpoint(str(source));meta=ck.meta;tok=load_tokenizer(str(ROOT/'models/base'))
    s.event('model_load_started')
    model=DecisionModel(str(ROOT/'models/base'),tok,'mps',lora=meta.lora,head_dim=meta.head_dim,option_isolation=meta.option_isolation,special_embeddings=meta.special_embeddings,dtype=torch.bfloat16,direct_load=True)
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.lm.config.use_cache=False
    ours=dataclasses.replace(meta,head=None,weights_dtype='bf16',extra={});init=ck.warm_start(model,ours);model.head.temperature=meta.temperature
    s.REPORT['checks']['original_or_resumed_adapter_head_loaded']='passed';s.event('model_loaded',warm_start_tensors=init['tensors'],memory=s.snapshot())
    named=list(model.named_parameters());base=[(n,p) for n,p in named if not p.requires_grad];adapter=[p for n,p in named if p.requires_grad and 'lora_' in n];head=list(model.head.parameters());trainable=list(model.trainable_parameters())
    assert {id(p) for p in trainable}=={id(p) for p in adapter+head}
    names=[n for n,p in named if p.requires_grad];names_sha=hashlib.sha256(json.dumps(names).encode()).hexdigest()
    optimizer=torch.optim.AdamW(trainable,lr=CFG['lr'],weight_decay=CFG['weight_decay'])
    if a.start:
        resume=torch.load(ROOT/'pilot'/f'stage-{a.start:04d}'/'resume.pt',map_location='cpu',weights_only=True)
        assert resume['step']==a.start and resume['parameter_names_sha256']==names_sha
        optimizer.load_state_dict(resume['optimizer']);torch.set_rng_state(resume['cpu_rng']);torch.mps.set_rng_state(resume['mps_rng'])
        s.REPORT['checks']['optimizer_rng_resumed']='passed'
    before=s.digest_params(base);adapter_before=s.digest_params([(n,p) for n,p in named if p.requires_grad and 'lora_' in n]);head_before=s.digest_params(list(model.head.named_parameters()))
    rows,reqs=records();order=CFG['training_order'];smoke_indices=order[:12]
    if a.smoke:
        from collections import Counter
        assert Counter(rows[i]['expected'] for i in smoke_indices)=={'review':4,'skip':4,'unknown':4}
        for i in smoke_indices:model.encode(tok,materialize(reqs[i]),max_state=384,max_branch=512,strict=True)
    s.event('training_started',memory=s.snapshot())
    for step in range(a.start+1,a.target+1):
        started=time.perf_counter();model.train();optimizer.zero_grad(set_to_none=True)
        lr=CFG['lr']*min(1,step/10)
        for group in optimizer.param_groups:group['lr']=lr
        idx=order[step-1];rec=materialize(reqs[idx]);enc=model.encode(tok,rec,max_state=384,max_branch=512,strict=True)
        assert len(enc['ids'])<=512
        z=model(enc);loss=question_loss(z[0],rec['questions'][0],'mps',ord_w=0);loss.backward();torch.mps.synchronize()
        ga,gh=s.norm(adapter),s.norm(head)
        assert all(p.grad is None for _,p in base) and any(p.grad is not None for p in adapter) and any(p.grad is not None for p in head)
        torch.nn.utils.clip_grad_norm_(trainable,CFG['clip_grad_norm'],error_if_nonfinite=True);optimizer.step();torch.mps.synchronize()
        value=float(loss.detach().cpu());s.REPORT['steps_completed']=step
        item={'step':step,'loss':value,'lr':lr,'adapter_grad_norm':ga,'head_grad_norm':gh,'packed_tokens':len(enc['ids']),'step_seconds':round(time.perf_counter()-started,3),'memory':s.snapshot()}
        s.REPORT['steps'].append(item);s.event('optimizer_step',**item);torch.mps.empty_cache()
    assert any(x['adapter_grad_norm']>0 for x in s.REPORT['steps']) and any(x['head_grad_norm']>0 for x in s.REPORT['steps'])
    s.event('weight_integrity_check');after=s.digest_params(base);assert before==after
    s.REPORT['base_sha256_before']=before;s.REPORT['base_sha256_after']=after;s.REPORT['checks']['base_frozen']='passed'
    s.REPORT['checks']['adapter_changed']=adapter_before!=s.digest_params([(n,p) for n,p in named if p.requires_grad and 'lora_' in n]);s.REPORT['checks']['head_changed']=head_before!=s.digest_params(list(model.head.named_parameters()))
    assert s.REPORT['checks']['adapter_changed'] and s.REPORT['checks']['head_changed']
    s.event('checkpoint_save_started');ckout.parent.mkdir(exist_ok=True)
    model.lm.save_pretrained(ckout);ours.head={k:v.detach().cpu() for k,v in model.head.state_dict().items()};ours.extra={'pilot_step':a.target,'train_sha256':CFG['dataset_hashes']['train.json'],'original_model_revision':CFG['original_model_revision']};write_meta(ckout,ours);tok.save_pretrained(ckout)
    if not a.smoke:
        torch.save({'step':a.target,'optimizer':cpu_tree(optimizer.state_dict()),'cpu_rng':torch.get_rng_state(),'mps_rng':torch.mps.get_rng_state(),'parameter_names_sha256':names_sha},out/'resume.pt')
    s.REPORT['checkpoint_files_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ckout.iterdir() if p.is_file()}
    s.REPORT['exposure']={'rows_seen':a.target,'row_epochs':a.target/576,'unique_semantic_scenarios':len({rows[i]['scenario_id'] for i in order[:a.target]})}
    s.REPORT['status']='completed';s.REPORT['weights_discarded_for_main_initialization']=a.smoke;s.event('checkpoint_saved',memory=s.snapshot())
    print(json.dumps({'status':'completed','target':a.target,'step_seconds':[x['step_seconds'] for x in s.REPORT['steps']]}),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as err:
        s.REPORT['status']='safety_stop' if s.MEMORY_STOP else 'failed';s.REPORT['failed_stage']=s.STAGE
        import re
        msg=re.sub(r'/(?:[^/]+)/[^\s\"\']+','[local-path]',str(err))[:800]
        s.REPORT['error']={'type':type(err).__name__,'message':msg,'safety_reason':s.MEMORY_STOP};s.event('failed',error_type=type(err).__name__,operation=msg,memory=s.snapshot());print(json.dumps(s.REPORT['error']))
    finally:s.STOP.set()
