# SPDX-License-Identifier: Apache-2.0
# Modified for public packaging: relative data/vendor paths, local-only helpers, no dashboard dependency. Training math unchanged.
"""Local plumbing smoke, canonical Kev model/head/warm-start/loss. No benchmark data."""
import os,sys,json,time,hashlib,datetime,threading,signal,ctypes,dataclasses,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
for k,v in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','PYTORCH_ENABLE_MPS_FALLBACK':'0','TOKENIZERS_PARALLELISM':'false','DO_NOT_TRACK':'1'}.items():os.environ[k]=v
for k in list(os.environ):
    if any(x in k for x in ['API_KEY','HF_TOKEN','GATEWAY','ANTHROPIC']):os.environ.pop(k,None)
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'vendor'))
from isolation import isolate
isolate()
import torch,psutil
from kev.checkpoint import Checkpoint,Meta,write_meta
from kev.model import DecisionModel,load_tokenizer
from kev.train import question_loss
from kev.data import load_records,materialize
from kev.api import SystemOneRequest,to_record,to_answers
from state import update
torch.set_num_threads(2)
REPORT={'kind':'technical_training_smoke_only','attempt':2,'direct_load':True,'status':'preparing','model_revision':'6cfce5c2fa4b4bd64026336ab649c5ca78857d52','source_revision':'dd375f27fd61486c52204205e10a0d2e10f3b34a','steps_requested':10,'steps_completed':0,'microbatch':1,'accumulation':1,'device':'mps','frozen_backbone_dtype':'bfloat16','adapter_head_dtype':'float32','autocast':False,'cpu_fallback':False,'checks':{k:'not_run' for k in ['warm_start_adapter_head','gradient_adapter','gradient_head','base_frozen','checkpoint_save','mlx_reload','typed_api']},'events':[],'steps':[]}
LOCK=threading.RLock();START=time.perf_counter();STOP=threading.Event();MEMORY_STOP=None;STAGE='preparing';MODEL=None
BASE_SWAP=psutil.swap_memory().used
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def snapshot():
    pressure=ctypes.c_int();size=ctypes.c_size_t(ctypes.sizeof(pressure));lib=ctypes.CDLL(None)
    code=lib.sysctlbyname(b'kern.memorystatus_vm_pressure_level',ctypes.byref(pressure),ctypes.byref(size),None,0)
    return {'available_bytes':psutil.virtual_memory().available,'swap_used_bytes':psutil.swap_memory().used,'swap_delta_bytes':psutil.swap_memory().used-BASE_SWAP,'memory_pressure_level':pressure.value if code==0 else None,'rss_bytes':psutil.Process().memory_info().rss,'mps_allocated_bytes':torch.mps.current_allocated_memory() if torch.backends.mps.is_available() else 0,'mps_driver_bytes':torch.mps.driver_allocated_memory() if torch.backends.mps.is_available() else 0}
def event(stage,**values):
    global STAGE
    with LOCK:
        STAGE=stage;item={'utc':utc(),'elapsed_seconds':round(time.perf_counter()-START,3),'stage':stage,'step':REPORT['steps_completed'],**values}
        REPORT['events'].append(item)
        (OUT/'smoke-report.json').write_text(json.dumps(REPORT,indent=2))
        (OUT/'smoke-events.json').write_text(json.dumps({'model_revision':REPORT['model_revision'],'data_sha256':REPORT.get('data_sha256'),'events':REPORT['events']},indent=2))
        label=REPORT.get('board_name','로컬 학습 smoke');total=REPORT.get('steps_requested',10)
        s=json.loads((ROOT/'status.json').read_text());steps=[x for x in s['steps'] if x['name']!=label]
        steps.append({'name':label,'status':REPORT['status'],'detail':f"{stage} · step {REPORT['steps_completed']}/{total} · loss {REPORT['steps'][-1]['loss'] if REPORT['steps'] else '미측정'} · final test 미실행"})
        update(steps=steps,training_smoke={'stage':stage,'status':REPORT['status'],'step':REPORT['steps_completed'],'total_steps':total,'remaining_steps':max(0,total-REPORT['steps_completed']),'latest':values},event=label+': '+stage)
def monitor():
    global MEMORY_STOP
    while not STOP.wait(2):
        m=snapshot();event(STAGE,memory=m)
        reason='low_available_memory' if m['available_bytes']<4*1024**3 else 'swap_growth' if m['swap_delta_bytes']>512*1024**2 else 'critical_memory_pressure' if m['memory_pressure_level'] is not None and m['memory_pressure_level']>=4 else 'warning_with_reduced_headroom' if m['memory_pressure_level']==2 and (m['available_bytes']<8*1024**3 or m['swap_delta_bytes']>128*1024**2) else None
        if reason:
            MEMORY_STOP=reason;REPORT['status']='safety_stop';event('safety_stop',reason=reason,memory=m)
            os.kill(os.getpid(),signal.SIGINT);return
def digest_params(named):
    h=hashlib.sha256()
    for name,p in named:
        h.update(name.encode());h.update(str(p.dtype).encode());h.update(str(tuple(p.shape)).encode())
        flat=p.detach().reshape(-1)
        for start in range(0,p.numel(),1024*1024):
            part=flat[start:start+1024*1024].to('cpu').contiguous().view(torch.uint8)
            h.update(part.numpy().tobytes())
    return h.hexdigest()
def norm(params):
    terms=[p.grad.detach().float().square().sum() for p in params if p.grad is not None]
    return float(torch.stack(terms).sum().sqrt().cpu()) if terms else 0.0
def make_data():
    mapping={'red':'review','blue':'skip','gray':'unknown'};rows=[]
    for i,color in enumerate(['red','blue','gray','red','gray','blue','red','blue']):
        rows.append({'state':f'Local synthetic routing exercise {i}. Rule: red means review; blue means skip; gray means unknown. Observed flag: {color}.','questions':{'need':{'type':'choice','instructions':'Apply only the stated flag rule. Which action is required?','criteria':{'review':'The rule says review','skip':'The rule says skip','unknown':'The rule says unknown'},'label':mapping[color]}}})
    raw=''.join(json.dumps(x,separators=(',',':'))+'\n' for x in rows).encode();p=OUT/'smoke-train.jsonl';p.write_bytes(raw)
    REPORT['data_sha256']=hashlib.sha256(raw).hexdigest();REPORT['training_records']=8
    return load_records(p,source='new_local_smoke')
def probe_answers(model,tok,reqs):
    model.eval();out=[]
    with torch.no_grad():
        for req in reqs:
            rec=materialize(req);enc=model.encode(tok,rec,max_state=128,max_branch=256,strict=True)
            ps=[p.tolist() for p in model.probs(enc)];_,meta=to_record(SystemOneRequest.model_validate({'state':req['state'],'questions':{k:{a:b for a,b in q.items() if a in ('type','instructions','criteria')} for k,q in req['questions'].items()}}))
            out.append(to_answers(ps,meta))
    return out
def main():
    global MODEL
    if not torch.backends.mps.is_available():raise RuntimeError('explicit_mps_unavailable_no_cpu_fallback')
    if (OUT/'candidate').exists():raise RuntimeError('candidate_exists_refusing_overwrite')
    torch.mps.set_per_process_memory_fraction(.60);torch.manual_seed(7)
    REPORT['status']='running';reqs=make_data();event('preflight',memory=snapshot(),baseline_swap_bytes=BASE_SWAP)
    threading.Thread(target=monitor,daemon=True).start()
    ck=Checkpoint(str(ROOT/'models/kev-4b'));meta=ck.meta
    tok=load_tokenizer(str(ROOT/'models/base'))
    event('model_load_started')
    MODEL=DecisionModel(str(ROOT/'models/base'),tok,'mps',lora=meta.lora,head_dim=meta.head_dim,option_isolation=meta.option_isolation,special_embeddings=meta.special_embeddings,dtype=torch.bfloat16,direct_load=True)
    MODEL.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});MODEL.lm.config.use_cache=False
    ours=dataclasses.replace(meta,head=None,weights_dtype='bf16',extra={})
    source=ck.warm_start(MODEL,ours);MODEL.head.temperature=meta.temperature
    REPORT['checks']['warm_start_adapter_head']='passed'
    named=list(MODEL.named_parameters());base=[(n,p) for n,p in named if not p.requires_grad];adapter=[p for n,p in named if p.requires_grad and 'lora_' in n];head=list(MODEL.head.parameters());trainable=list(MODEL.trainable_parameters())
    if not adapter or not base or {id(p) for p in trainable}!={id(p) for p in adapter+head}:raise RuntimeError('unexpected_trainable_parameter_set')
    REPORT['parameters']={'frozen':sum(p.numel() for _,p in base),'adapter':sum(p.numel() for p in adapter),'head':sum(p.numel() for p in head),'warm_start_tensors':source['tensors'],'trainable_dtypes':sorted({str(p.dtype) for p in trainable})}
    event('model_loaded',memory=snapshot())
    before_base=digest_params(base);before_adapter=digest_params([(n,p) for n,p in named if p.requires_grad and 'lora_' in n]);before_head=digest_params(list(MODEL.head.named_parameters()))
    REPORT['frozen_base_sha256_before']=before_base
    encs=[MODEL.encode(tok,materialize(r),max_state=128,max_branch=256,strict=True) for r in reqs]
    REPORT['packed_tokens']=[len(e['ids']) for e in encs]
    REPORT['before_training_answers']=probe_answers(MODEL,tok,reqs)
    optimizer=torch.optim.AdamW(trainable,lr=2e-5,weight_decay=.01)
    event('training_started',memory=snapshot())
    for step in range(1,11):
        started=time.perf_counter();MODEL.train();optimizer.zero_grad(set_to_none=True)
        lr=2e-5*min(1.,step/2)
        for group in optimizer.param_groups:group['lr']=lr
        idx=(step-1)%8;rec=materialize(reqs[idx]);logits=MODEL(encs[idx])
        loss=question_loss(logits[0],rec['questions'][0],'mps',ord_w=0)
        loss.backward();torch.mps.synchronize()
        ga,gh=norm(adapter),norm(head)
        if not ga>0 or not gh>0 or not all(p.grad is None for _,p in base):raise RuntimeError('gradient_flow_or_freeze_check')
        torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True);optimizer.step();torch.mps.synchronize()
        value=float(loss.detach().cpu())
        REPORT['steps_completed']=step;REPORT['steps'].append({'step':step,'loss':value,'lr':lr,'adapter_grad_norm':ga,'head_grad_norm':gh,'step_seconds':round(time.perf_counter()-started,3),'memory':snapshot()})
        event('optimizer_step',**REPORT['steps'][-1]);torch.mps.empty_cache()
    REPORT['checks']['gradient_adapter']=REPORT['checks']['gradient_head']='passed'
    event('verifying_weights')
    after_base=digest_params(base);REPORT['frozen_base_sha256_after']=after_base
    if before_base!=after_base:raise RuntimeError('frozen_base_changed')
    REPORT['checks']['base_frozen']='passed'
    REPORT['adapter_changed']=before_adapter!=digest_params([(n,p) for n,p in named if p.requires_grad and 'lora_' in n]);REPORT['head_changed']=before_head!=digest_params(list(MODEL.head.named_parameters()))
    if not REPORT['adapter_changed'] or not REPORT['head_changed']:raise RuntimeError('trainable_weights_not_changed')
    REPORT['after_training_answers']=probe_answers(MODEL,tok,reqs)
    event('checkpoint_save_started')
    candidate=OUT/'candidate';MODEL.lm.save_pretrained(candidate);ours.head={k:v.detach().cpu() for k,v in MODEL.head.state_dict().items()};ours.extra={'smoke_steps':10,'smoke_data_sha256':REPORT['data_sha256'],'init_model_revision':REPORT['model_revision']};write_meta(candidate,ours);tok.save_pretrained(candidate)
    REPORT['checks']['checkpoint_save']='passed';REPORT['status']='training_completed_reload_pending';event('checkpoint_saved',memory=snapshot())
if __name__=='__main__':
    try:main()
    except BaseException as error:
        REPORT['status']='safety_stop' if MEMORY_STOP else 'failed';REPORT['failed_stage']=STAGE
        msg=re.sub(r'/(?:[^/]+)/[^\s\"\']+','[local-path]',str(error))[:900]
        REPORT['error']={'type':type(error).__name__,'message':msg,'safety_reason':MEMORY_STOP}
        event('failed',error_type=type(error).__name__,operation=msg,memory=snapshot())
        print(json.dumps(REPORT['error']),flush=True)
    finally:
        STOP.set()
