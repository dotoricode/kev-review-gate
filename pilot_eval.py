# SPDX-License-Identifier: Apache-2.0
# Modified for public packaging: relative data/vendor paths, local-only helpers, no dashboard dependency. Training math unchanged.
import os,sys,json,time,datetime,threading,signal,hashlib,argparse,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
ROOT=Path(__file__).resolve().parent
for k,v in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','DO_NOT_TRACK':'1'}.items():os.environ[k]=v
for k in list(os.environ):
    if any(x in k for x in ['API_KEY','HF_TOKEN','GATEWAY','ANTHROPIC']):os.environ.pop(k,None)
from isolation import isolate
isolate()
import psutil,ctypes
from state import update
from pilot_metrics import aggregate
STOP=threading.Event();LOCK=threading.RLock();REPORT={};OUT=None;START=time.perf_counter();SWAP=psutil.swap_memory().used;STAGE='initial';SAFETY=None
def memory():
    pressure=ctypes.c_int();size=ctypes.c_size_t(4);code=ctypes.CDLL(None).sysctlbyname(b'kern.memorystatus_vm_pressure_level',ctypes.byref(pressure),ctypes.byref(size),None,0)
    return {'available_bytes':psutil.virtual_memory().available,'swap_used_bytes':psutil.swap_memory().used,'swap_delta_bytes':psutil.swap_memory().used-SWAP,'memory_pressure_level':pressure.value if code==0 else None,'rss_bytes':psutil.Process().memory_info().rss}
def event(stage,**values):
    global STAGE
    with LOCK:
        STAGE=stage;REPORT['events'].append({'stage':stage,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':round(time.perf_counter()-START,3),'checkpoint_step':REPORT['step'],'completed_rows':len(REPORT['rows']),**values})
        OUT.write_text(json.dumps(REPORT,indent=2));s=json.loads((ROOT/'status.json').read_text());steps=[x for x in s['steps'] if x['name']!='본학습 dev 평가'];steps.append({'name':'본학습 dev 평가','status':REPORT['status'],'detail':f"checkpoint {REPORT['step']} · dev {len(REPORT['rows'])}/120행 /15시나리오 · final test 봉인"});update(steps=steps,pilot_evaluation={'step':REPORT['step'],'stage':stage,'completed_rows':len(REPORT['rows']),'total_rows':120,'status':REPORT['status']},event='dev 평가: '+stage)
def monitor():
    global SAFETY
    while not STOP.wait(2):
        m=memory();event(STAGE,memory=m)
        if m['available_bytes']<4*1024**3 or m['swap_delta_bytes']>512*1024**2 or m['memory_pressure_level'] is not None and m['memory_pressure_level']>=4:
            SAFETY='memory_or_swap_safety_stop';os.kill(os.getpid(),signal.SIGINT);return
def main():
    global REPORT,OUT
    p=argparse.ArgumentParser();p.add_argument('--step',type=int,required=True);a=p.parse_args()
    cfg=json.loads((ROOT/'pilot/execution_config.json').read_text());schedule=json.loads((ROOT/'pilot/schedule.json').read_text());assert a.step in schedule['schedule']
    OUT=ROOT/'pilot'/f'dev-step-{a.step:04d}.json'
    if OUT.exists():raise RuntimeError('existing_dev_results_refuse_repeat')
    REPORT={'step':a.step,'split':'new_dev','status':'running','label_provenance':'Synthetic designer expectations, AI consistency audited; not human truth','n_independent_semantic_scenarios':15,'n_correlated_filename_rows':120,'rows':[],'events':[],'inference':'fixed in-process MLX; no generation; same prompt/parser/temperature for every checkpoint','config_sha256':hashlib.sha256((ROOT/'pilot/execution_config.json').read_bytes()).hexdigest(),'schedule_sha256':hashlib.sha256((ROOT/'pilot/schedule.json').read_bytes()).hexdigest()}
    event('eval_load_started',memory=memory());threading.Thread(target=monitor,daemon=True).start()
    import kev.checkpoint as c
    import kev.mlx_model as mm
    from kev.checkpoint import Checkpoint,LoadOptions
    from kev.api import SystemOneRequest,to_record,to_answers
    import mlx.core as mx
    from contract import validate
    import contract as e
    mm.CACHE_LIMIT=256*1024**2
    candidate=ROOT/'models/kev-4b' if a.step==0 else ROOT/'pilot/checkpoints'/f'step-{a.step:04d}'
    base=ROOT/'models/base'
    def resolve(run):
        if str(run) not in [str(candidate),str(base),str(base)+'@']:raise ValueError('offline_path_allowlist')
        return str(base if str(run).startswith(str(base)) else candidate)
    c.resolve_run=resolve;ck=Checkpoint(str(candidate));ck.meta.base=str(base);ck.meta.base_revision=None;tok,model=ck.load('mps',LoadOptions(backend='mlx'));assert model.backend=='mlx'
    REPORT['checkpoint_relative_path']=str(candidate.relative_to(ROOT));REPORT['checkpoint_files_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in candidate.iterdir() if p.is_file()};REPORT['checkpoint_artifact_sha256']=hashlib.sha256(json.dumps(REPORT['checkpoint_files_sha256'],sort_keys=True,separators=(',',':')).encode()).hexdigest();event('eval_model_loaded',memory=memory(),mlx_peak_bytes=mx.get_peak_memory())
    rows=json.loads((ROOT/'training-input/dev.json').read_text())['cases']
    for case in rows:
        questions={'need':{'type':case['type'],'instructions':case['question'],'criteria':{x['value']:x['description'] for x in case['criteria']}}};state={'synthetic_diff':case['state']['synthetic_diff']};request={'state':state,'questions':questions,'model':'kev-latest'}
        row={k:case[k] for k in ['id','family','scenario_id','variant_id','expected']};row['input_sha256']=e.sha({'state':state,'questions':questions});started=time.perf_counter()
        rec,meta=to_record(SystemOneRequest.model_validate(request));enc=model.encode(tok,rec,max_state=384,max_branch=512,strict=True);assert len(enc['ids'])<=512
        answers=to_answers([p.tolist() for p in model.probs(enc)],meta)
        try:validate(answers,questions);row.update(predicted=answers['need']['choice'],answers=answers,status='model_answer')
        except (ValueError,KeyError,TypeError):row.update(predicted='invalid',status='invalid_output',answers=answers)
        row.update(client_latency_ms=round((time.perf_counter()-started)*1000,3),input_tokens=len(enc['ids']),saved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());REPORT['rows'].append(row)
        if len(REPORT['rows'])%8==0:event('dev_rows',memory=memory(),mlx_peak_bytes=mx.get_peak_memory())
    REPORT['metrics']=aggregate(REPORT['rows']);REPORT['status']='completed';REPORT['typed_api_validated']=True;REPORT['elapsed_total_seconds']=round(time.perf_counter()-START,3);event('dev_completed',memory=memory(),mlx_peak_bytes=mx.get_peak_memory(),mlx_active_bytes=mx.get_active_memory());print(json.dumps({'step':a.step,'status':'completed','metrics':REPORT['metrics']['scenarios']}))
if __name__=='__main__':
    try:main()
    except BaseException as err:
        if OUT is not None and REPORT:
            REPORT['status']='safety_stop' if SAFETY else 'failed';REPORT['failed_stage']=STAGE;REPORT['error']={'type':type(err).__name__,'reason':SAFETY or re.sub(r'/(?:[^/]+)/[^\s\"\']+','[local-path]',str(err))[:500]};event('eval_failed',memory=memory())
        print('pilot_eval_failed',type(err).__name__)
    finally:STOP.set()
