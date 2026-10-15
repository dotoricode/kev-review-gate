# SPDX-License-Identifier: Apache-2.0
# Modified for public packaging: relative data/vendor paths, local-only helpers, no dashboard dependency. Training math unchanged.
"""Fixed schedule orchestrator. All child inference/training processes isolate themselves."""
import os,sys,json,subprocess,datetime,hashlib
from pathlib import Path
from pilot_metrics import select
ROOT=Path(__file__).resolve().parent;P=ROOT/'pilot';PY=Path(sys.executable)
STATUS=P/'run-status.json'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def status(phase,**values):
    x={'phase':phase,'utc':now(),**values};STATUS.write_text(json.dumps(x,indent=2));print(json.dumps(x),flush=True)
def child(script,args,log,result):
    env={k:v for k,v in os.environ.items() if not any(s in k for s in ['API_KEY','HF_TOKEN','GATEWAY','ANTHROPIC'])}
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTORCH_ENABLE_MPS_FALLBACK='0')
    with (P/log).open('w') as f:r=subprocess.run([str(PY),str(ROOT/script),*args],cwd=ROOT,stdout=f,stderr=f,env=env)
    if r.returncode!=0 or not result.exists():raise RuntimeError('child_process_failed:'+script)
    value=json.loads(result.read_text())
    if value.get('status')!='completed':raise RuntimeError('child_incomplete:'+script+':'+value.get('status','unknown'))
    return value
def main():
    if (P/'selected_checkpoint.json').exists():raise SystemExit('Selection already frozen; no repeat.')
    fd=os.open(P/'main-run.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
    (P/'orchestrator.pid').write_text(str(os.getpid()))
    schedule=json.loads((P/'schedule.json').read_text())['schedule'];evaluations=[]
    status('baseline_dev_started',step=0)
    evaluations.append(child('pilot_eval.py',['--step','0'],'eval-step-0000.log',P/'dev-step-0000.json'))
    for start,target in zip(schedule,schedule[1:]):
        status('training_segment_started',start=start,target=target)
        child('pilot_train.py',['--start',str(start),'--target',str(target)],f'train-to-{target:04d}.log',P/f'stage-{target:04d}'/'smoke-report.json')
        status('checkpoint_dev_started',step=target)
        evaluations.append(child('pilot_eval.py',['--step',str(target)],f'eval-step-{target:04d}.log',P/f'dev-step-{target:04d}.json'))
    chosen,eligible=select(evaluations)
    selection={'status':'frozen_before_final_test','selected_step':chosen['step'],'trained_model_selected':chosen['step']!=0,'checkpoint_relative_path':chosen['checkpoint_relative_path'],'checkpoint_artifact_sha256':chosen['checkpoint_artifact_sha256'],'eligible_steps':eligible,'primary_scenario_metrics':chosen['metrics']['scenarios'],'all_checkpoint_dev_metrics':[{'step':r['step'],'metrics':r['metrics']} for r in evaluations],'selection_rule':'Frozen safety gates and lexicographic unsafe skips, review-to-skip, -macroF1, -balanced_accuracy, step; baseline included unconditionally and ties prefer earlier','configuration_sha256':hashlib.sha256((P/'execution_config.json').read_bytes()).hexdigest(),'schedule_sha256':hashlib.sha256((P/'schedule.json').read_bytes()).hexdigest(),'protocol_sha256':'99b9084d3d92300eb5c16b2be73f78233efe54ee1d94fe3b7832f65b94d0ad4f','frozen_utc':now(),'final_test':'not_received_not_opened_not_used','limitations':'576 train rows are72 semantic scenarios; dev120 rows are15 scenarios in5 families; fully synthetic AI-audited expectations, not human truth; no production reliability claim'}
    (P/'selected_checkpoint.json').write_text(json.dumps(selection,indent=2));status('selection_frozen',selected_step=chosen['step'],eligible_steps=eligible,final_test='awaiting_parent')
if __name__=='__main__':
    try:main()
    except BaseException as err:status('stopped',error_type=type(err).__name__,reason=str(err)[:250]);raise
