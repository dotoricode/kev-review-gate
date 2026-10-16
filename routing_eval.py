# SPDX-License-Identifier: Apache-2.0
"""Identical-input live Jev comparison and local Kev evaluation; no fallback labels."""
import argparse, hashlib, importlib.util, json, math, statistics, sys, time, urllib.error, urllib.request
from pathlib import Path
from routing_policy import questions, CHOICES, select
from routing_data import sha
ROOT=Path(__file__).resolve().parent
# This tool reproduces one frozen experiment; a new experiment needs a new reviewed pin.
FROZEN_SELECTION_SHA256='7b8c5ff4abd44656cff420b1752d6292d9a1a4d827fbfba1846cc5b6325d8deb'
FROZEN_PROTOCOL_SHA256='e2e889843ac28f4acfb67d20712681c717dcc06727cbe9be69bf59b81c0f633d'


def load_frozen_protocol():
 raw=(ROOT/'routing/data/protocol.json').read_bytes()
 if hashlib.sha256(raw).hexdigest()!=FROZEN_PROTOCOL_SHA256:raise ValueError('frozen_protocol_integrity_failure')
 return json.loads(raw)


def aggregate(rows):
 measured=[r for r in rows if r.get('status')=='completed']
 per={c:[] for c in CHOICES}
 for r in measured:per[r['expected']].append(r['predicted']==r['expected'])
 recalls={c:sum(v)/len(v) if v else None for c,v in per.items()}
 risks=[r for r in measured if r['expected']=='astra_high']
 lat=[r['latency_ms'] for r in measured]
 return {'total':len(rows),'measured':len(measured),'unavailable':len(rows)-len(measured),
  'accuracy':sum(r['predicted']==r['expected'] for r in measured)/len(measured) if measured else None,
  'macro_accuracy':statistics.mean(v for v in recalls.values() if v is not None) if measured else None,
  'class_recall':recalls,'high_risk_cases':len(risks),
  'high_risk_downrouting':sum(r['predicted'] not in ('astra_high','unknown') for r in risks),
  'latency_median_ms':statistics.median(lat) if lat else None,'latency_mean_ms':statistics.mean(lat) if lat else None}


def select_checkpoint(reports):
 valid=[(step,r) for step,r in reports.items() if step in (0,25,50) and r.get('status')=='completed' and r['metrics']['measured']==20]
 if len(valid)!=3:raise ValueError('fixed_dev_schedule_incomplete')
 selected=min(valid,key=lambda item:(-item[1]['metrics']['macro_accuracy'],item[1]['metrics']['high_risk_downrouting'],item[0]))
 return {'selected_step':selected[0],'selection_basis':'dev_only_fixed_schedule','metrics':selected[1]['metrics']}


def policy_metrics(rows,cases):
 by_id={c['id']:c for c in cases};applied=[]
 catalog=[{'model':'sol','tier':'sol','efforts':['medium','high']},{'model':'astra','tier':'astra','efforts':['medium','high']}]
 for row in rows:
  r=dict(row)
  if r.get('status')=='completed':
   c=by_id[row['id']];payload=dict(c['state'],candidates=catalog)
   def effective(choice):
    if choice=='unknown':return 'unknown'
    decision=select(choice,payload)
    return decision['selected_model']+'_'+decision['effort']
   r['expected']=effective(r['expected']);r['predicted']=effective(r['predicted'])
  applied.append(r)
 result=aggregate(applied)
 result['policy_profile_underallocation']=result.pop('high_risk_downrouting')
 result['policy_astra_high_targets']=result.pop('high_risk_cases')
 by_result={r['id']:r for r in applied}
 original_risks=[r for r in rows if r.get('status')=='completed' and r['expected']=='astra_high']
 result['original_high_risk_cases']=len(original_risks)
 result['original_high_risk_downrouting']=sum(r['predicted'] not in ('astra_high','unknown') for r in original_risks)
 result['policy_residual_high_risk_downrouting']=sum(by_result[r['id']]['predicted'] not in ('astra_high','unknown') for r in original_risks)
 return result


def validate_dev_reports(reports):
 protocol=load_frozen_protocol()
 dev_path=ROOT/'routing/data/dev.json'
 dev_bytes=dev_path.read_bytes();digest=hashlib.sha256(dev_bytes).hexdigest()
 if digest!=protocol['dataset_hashes']['dev.json'] or protocol['questions_sha256']!=sha(questions()):raise ValueError('frozen_dev_contract_mismatch')
 cases={c['id']:c for c in json.loads(dev_bytes)['cases']}
 pins=json.loads((ROOT/'provenance/routing-evaluation-weights.json').read_text())['candidates']
 for step,r in reports.items():
  if r.get('provider')!='Kev' or r.get('dataset_sha256')!=digest or r.get('questions_sha256')!=sha(questions()) or r.get('artifact_manifest')!=pins[str(step)]:raise ValueError('invalid_dev_report_identity')
  rows=r.get('rows')
  if not isinstance(rows,list) or len(rows)!=20 or {row.get('id') for row in rows}!=set(cases):raise ValueError('invalid_dev_report_cases')
  for row in rows:
   c=cases[row['id']]
   if row.get('status')!='completed' or row.get('family')!=c['family'] or row.get('expected')!=c['expected'] or row.get('input_sha256')!=sha({'state':c['state'],'questions':questions()}):raise ValueError('invalid_dev_report_case')
   answer=validate_answer({'answers':row.get('answers')})
   if row.get('predicted')!=answer['choice']:raise ValueError('invalid_dev_report_prediction')
  if r.get('metrics')!=aggregate(rows) or r.get('raw_metrics')!=aggregate(rows):raise ValueError('invalid_dev_report_metrics')


def freeze_selection(results,output):
 load_frozen_protocol()
 reports={step:json.loads((Path(results)/f'kev-dev-step{step}.json').read_text()) for step in (0,25,50)}
 validate_dev_reports(reports)
 decision=select_checkpoint(reports);decision['questions_sha256']=sha(questions())
 decision['dev_report_sha256']={str(step):sha(r) for step,r in reports.items()}
 decision['artifact_manifest']=reports[decision['selected_step']]['artifact_manifest']
 decision['selection_sha256']=sha(decision)
 with Path(output).open('x') as stream:json.dump(decision,stream,indent=2)
 return decision


def claim_final_attempt(selection_path,output,provider,weights):
 if not selection_path:raise ValueError('final_requires_frozen_selection')
 if provider not in ('Jev','Kev'):raise ValueError('invalid_provider')
 decision=json.loads(Path(selection_path).read_text());digest=decision.pop('selection_sha256',None)
 if digest!=FROZEN_SELECTION_SHA256 or sha(decision)!=digest or decision.get('questions_sha256')!=sha(questions()):raise ValueError('selection_integrity_failure')
 if provider=='Kev' and json.loads((Path(weights)/'routing_manifest.json').read_text())!=decision['artifact_manifest']:raise ValueError('selected_weights_required')
 experiment=ROOT/'routing/results'
 legacy=experiment/('final-attempt-'+provider.lower()+'.json')
 if legacy.exists():raise FileExistsError('final_attempt_already_consumed')
 marker=experiment/'final-attempts'/FROZEN_PROTOCOL_SHA256/digest/(provider.lower()+'.json')
 marker.parent.mkdir(parents=True,exist_ok=True)
 with marker.open('x') as stream:json.dump({'selection_sha256':digest,'provider':provider,'status':'claimed_once_before_final_data_load','output':str(Path(output).name)},stream)
 return digest


def validate_answer(raw):
 if not isinstance(raw,dict) or not isinstance(raw.get('answers'),dict) or set(raw['answers'])!={'route'}:raise ValueError('invalid_response')
 answer=raw['answers'].get('route')
 if not isinstance(answer,dict) or set(answer)!={'type','choice','confidence','probabilities'}:raise ValueError('invalid_response')
 if answer.get('type')!='choice' or answer.get('choice') not in CHOICES:raise ValueError('invalid_choice')
 confidence=answer.get('confidence')
 if type(confidence) not in (int,float) or not math.isfinite(confidence) or not 0<=confidence<=1:raise ValueError('invalid_confidence')
 ps=answer.get('probabilities')
 if not isinstance(ps,dict) or set(ps)!=set(CHOICES):raise ValueError('missing_probabilities')
 if any(type(p) not in (int,float) or not math.isfinite(p) or p<0 or p>1 for p in ps.values()) or abs(sum(ps.values())-1)>0.02:raise ValueError('invalid_probabilities')
 if ps[answer['choice']]!=max(ps.values()):raise ValueError('nonmax_choice')
 return answer


def jev_call(state):
 spec=importlib.util.spec_from_file_location('jev_helper',Path.home()/'.agents/skills/help-me-jev/scripts/jev.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 if m.sensitive(state):raise ValueError('sensitive_input_blocked')
 key=m.api_key()
 if not key:raise ValueError('credentials_unavailable')
 payload={'state':state,'questions':questions(),'model':'jev-latest'}
 req=urllib.request.Request(m.API,json.dumps(payload).encode(),{'Authorization':'Bearer '+key,'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(req,timeout=30) as response:return json.load(response)
 except urllib.error.HTTPError as exc:raise ValueError(f'http_{exc.code}') from None
 except (OSError,TimeoutError):raise ValueError('transport_unavailable') from None


def local_predictor(base,weights):
 # Both evaluation candidates and the base must match repository pins before GPU imports.
 from routing_runtime import verify_routing_artifacts
 base,weights,cfg,verified_manifest=verify_routing_artifacts(base,weights,evaluation=True)
 from isolation import isolate
 isolate();sys.path.insert(0,str(ROOT/'vendor'))
 from kev.model import load_tokenizer,pad_id
 from kev.mlx_model import load_base,merge_lora,MLXDecisionModel
 from safetensors.torch import load_file
 from kev.api import SystemOneRequest,to_record,to_answers
 import kev.mlx_model as mm
 mm.CACHE_LIMIT=256*1024**2
 tok=load_tokenizer(str(base));lm=load_base(Path(base));merge_lora(lm,Path(weights));model=MLXDecisionModel(lm,pad_id(tok),head_dim=256)
 model.head.load_state_dict(load_file(Path(weights)/'head.safetensors'));model.head.temperature=cfg['temperature'];model.eval()
 def predict(state):
  rec,meta=to_record(SystemOneRequest.model_validate({'state':state,'questions':questions(),'model':'kev-review-router-v1'}))
  enc=model.encode(tok,rec,max_state=384,max_branch=512,strict=True)
  if len(enc['ids'])>512:raise ValueError('packed_token_limit')
  return {'answers':to_answers([x.tolist() for x in model.probs(enc)],meta),'usage':{'input_tokens':len(enc['ids'])},'token_ids_sha256':sha(enc['ids'])}
 predict.artifact_manifest=verified_manifest
 return predict


def evaluate(cases,predict,provider,journal=None):
 if len({c['id'] for c in cases})!=len(cases):raise ValueError('duplicate_case_id')
 rows=[]
 def persist(row):
  if journal is not None:
   journal.write(json.dumps(row,allow_nan=False)+'\n');journal.flush();__import__('os').fsync(journal.fileno())
 for c in cases:
  t=time.perf_counter();row={k:c[k] for k in ('id','family','expected')};row['input_sha256']=sha({'state':c['state'],'questions':questions()})
  persist({'id':c['id'],'status':'attempt_claimed_before_provider_call','provider':provider})
  try:
   raw=predict(c['state']);answer=validate_answer(raw)
   row.update(status='completed',predicted=answer['choice'],answers={'route':answer},usage=raw.get('usage'),token_ids_sha256=raw.get('token_ids_sha256'),reported_model=raw.get('model'))
  except ValueError as exc:
   allowed=('credentials_unavailable','sensitive_input_blocked','transport_unavailable','missing_probabilities','invalid_probabilities','invalid_choice','invalid_response','invalid_confidence','nonmax_choice')
   reason=str(exc);row.update(status='unavailable',error_code=reason if reason in allowed or reason.startswith('http_') else 'invalid_response')
  row['latency_ms']=round((time.perf_counter()-t)*1000,3);rows.append(row);persist(row)
  print(json.dumps({'provider':provider,'case':c['id'],'status':row['status']}),flush=True)
 return {'provider':provider,'questions_sha256':sha(questions()),'rows':rows,'metrics':aggregate(rows),'raw_metrics':aggregate(rows),'policy_metrics':policy_metrics(rows,cases),'status':'completed' if all(r['status']=='completed' for r in rows) else 'incomplete'}


def main():
 p=argparse.ArgumentParser();p.add_argument('--provider',choices=['Jev','Kev']);p.add_argument('--data');p.add_argument('--output',required=True);p.add_argument('--select-from');p.add_argument('--selection');p.add_argument('--base-dir');p.add_argument('--weights-dir');a=p.parse_args();out=Path(a.output)
 if out.exists():raise ValueError('output_exists_refusing_overwrite')
 protocol=load_frozen_protocol()
 if a.select_from:
  print(json.dumps(freeze_selection(a.select_from,out)));return
 if a.provider is None or a.data is None:raise ValueError('provider_and_data_required')
 final_selection=None
 dataset_bytes=Path(a.data).read_bytes();dataset_digest=hashlib.sha256(dataset_bytes).hexdigest()
 if dataset_digest not in protocol['dataset_hashes'].values():raise ValueError('dataset_not_in_frozen_protocol')
 if dataset_digest==protocol['dataset_hashes']['final_test.json']:final_selection=claim_final_attempt(a.selection,out,a.provider,a.weights_dir)
 out.parent.mkdir(parents=True,exist_ok=True)
 journal=out.with_suffix('.journal.jsonl').open('x')
 cases=json.loads(dataset_bytes)['cases'];predict=jev_call if a.provider=='Jev' else local_predictor(a.base_dir,a.weights_dir)
 with journal:result=evaluate(cases,predict,a.provider,journal)
 if a.provider=='Kev':result['artifact_manifest']=predict.artifact_manifest
 else:result['requested_model']='jev-latest'
 result['final_selection_sha256']=final_selection
 result['dataset_sha256']=dataset_digest
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result['metrics']))
if __name__=='__main__':main()
