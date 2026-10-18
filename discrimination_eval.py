# SPDX-License-Identifier: Apache-2.0
"""Once-only frozen joint-classifier experiment; live Jev is never a teacher."""
import argparse,hashlib,json,statistics,time,urllib.request,urllib.error,importlib.util
from pathlib import Path
from contract import sha,validate
from discrimination import questions
ROOT=Path(__file__).resolve().parent
PROTOCOL_SHA='76da5416a8ab3a66de1807473fe729820b3320da9a7f1d9062c64cbe5a62cb84'

def protocol():
    raw=(ROOT/'discrimination/data/protocol.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PROTOCOL_SHA:raise ValueError('protocol_integrity_failure')
    p=json.loads(raw)
    if p['questions_sha256']!=sha(questions()):raise ValueError('questions_integrity_failure')
    return p

def metrics(rows):
    measured=[r for r in rows if r['status']=='completed'];recalls={}
    for task,q in questions().items():
        recalls[task]={label:sum(r['predicted'][task]==label for r in measured if r['expected'][task]==label)/sum(r['expected'][task]==label for r in rows) for label in q['criteria']}
    family={r['family']:[] for r in rows}
    for r in rows:family[r['family']].append(r['status']=='completed' and r['predicted']==r['expected'])
    out={'total':len(rows),'measured':len(measured),'unavailable':len(rows)-len(measured),'recall':recalls,
      'need_macro_recall':statistics.mean(recalls['need'].values()),'route_macro_recall':statistics.mean(recalls['route'].values()),
      'sensitive_downroutes':sum(r['status']!='completed' or r['predicted']['need']!='review' or r['predicted']['route']!='sol_high' for r in rows if r['pair']=='sensitive'),
      'missing_false_precision':sum(r['status']=='completed' and (r['predicted']['need']!='unknown' or r['predicted']['route']!='unknown') for r in rows if r['pair']=='missing'),
      'inert_false_review':sum(r['status']=='completed' and r['predicted']['need']=='review' for r in rows if r['pair']=='inert'),
      'paired_correct_fraction':sum(all(v) for v in family.values())/len(family),
      'predictions':{task:{label:sum(r['predicted'][task]==label for r in measured) for label in q['criteria']} for task,q in questions().items()}}
    out['joint_macro_recall']=(out['need_macro_recall']+out['route_macro_recall'])/2
    return out

def jev(state):
    spec=importlib.util.spec_from_file_location('jev_helper',Path.home()/'.agents/skills/help-me-jev/scripts/jev.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    if helper.sensitive(state):raise ValueError('sensitive_input_blocked')
    key=helper.api_key()
    if not key:raise ValueError('credentials_unavailable')
    req=urllib.request.Request(helper.API,json.dumps({'state':state,'questions':questions(),'model':'jev-latest'}).encode(),{'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=30) as response:return json.load(response)
    except urllib.error.HTTPError as exc:raise ValueError('http_'+str(exc.code)) from None
    except (OSError,TimeoutError):raise ValueError('transport_unavailable') from None

def selection_result(results,training_dir):
    reports={step:json.loads((results/f'kev-dev-step{step}.json').read_text()) for step in (0,25,50)}
    dev_raw=(ROOT/'discrimination/data/dev.json').read_bytes()
    if hashlib.sha256(dev_raw).hexdigest()!=protocol()['dataset_hashes']['dev.json']:raise ValueError('dataset_integrity_failure')
    cases={c['id']:c for c in json.loads(dev_raw)['cases']}
    from discrimination_evidence import candidate_evidence
    for step,report in reports.items():
        evidence,manifest=candidate_evidence(training_dir,step,PROTOCOL_SHA)
        if type(report.get('candidate_step')) is not int or report['candidate_step']!=step or report.get('training_evidence')!=evidence or report.get('artifact_manifest')!=manifest:raise ValueError('candidate_report_evidence_mismatch')
        if report.get('status')!='completed' or report['protocol_sha256']!=PROTOCOL_SHA or report['provider']!='Kev' or report['split']!='dev' or report['questions_sha256']!=sha(questions()) or report['dataset_sha256']!=hashlib.sha256(dev_raw).hexdigest() or report['metrics']!=metrics(report['rows']):raise ValueError('invalid_dev_report')
        if report['metrics']['unavailable'] or len(report['rows'])!=len(cases) or {r['id'] for r in report['rows']}!=set(cases):raise ValueError('incomplete_dev_schedule')
        for row in report['rows']:
            case=cases[row['id']];validate(row['answers'],questions())
            if any(row[k]!=case[k] for k in ('expected','family','pair')) or row['predicted']!={k:row['answers'][k]['choice'] for k in questions()} or row['input_sha256']!=sha({'state':case['state'],'questions':questions()}):raise ValueError('invalid_dev_case')
    step=min(reports,key=lambda s:(-reports[s]['metrics']['joint_macro_recall'],reports[s]['metrics']['sensitive_downroutes'],s))
    result={'selected_step':step,'protocol_sha256':PROTOCOL_SHA,'questions_sha256':sha(questions()),'dev_report_sha256':{str(k):sha(v) for k,v in reports.items()},'candidate_evidence':{str(k):v['training_evidence'] for k,v in reports.items()},'artifact_manifest':reports[step]['artifact_manifest'],'selection':'max mean(need_macro_recall,route_macro_recall), min sensitive errors, min step'}
    result['selection_sha256']=sha(result)
    return result

def selection(results,output,training_dir):
    result=selection_result(results,training_dir)
    with output.open('x') as stream:json.dump(result,stream,indent=2)
    return result

def verify_selection(results,training_dir):
    chosen=json.loads((results/'selection.json').read_text())
    if chosen!=selection_result(results,training_dir):raise ValueError('selection_evidence_changed')
    return chosen

def promote(m):
    c=protocol()['promotion']
    return (m['measured']==m['total'] and m['need_macro_recall']>=c['minimum_need_macro_recall'] and m['route_macro_recall']>=c['minimum_route_macro_recall'] and m['sensitive_downroutes']==0 and m['missing_false_precision']==0 and m['inert_false_review']<=c['inert_false_review_max'] and m['paired_correct_fraction']>=c['paired_correct_fraction_min'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--provider',choices=['Kev','Jev']);p.add_argument('--split',choices=['dev','final_test']);p.add_argument('--weights-dir');p.add_argument('--training-dir',required=True);p.add_argument('--results',required=True);p.add_argument('--select',action='store_true');p.add_argument('--step',type=int);a=p.parse_args();frozen=protocol();results=Path(a.results);results.mkdir(parents=True,exist_ok=True)
    if a.select:print(json.dumps(selection(results,results/'selection.json',Path(a.training_dir))));return
    if not a.provider or not a.split:raise ValueError('provider_split_required')
    name=f'{a.provider.lower()}-{a.split}'+(f'-step{a.step}' if a.split=='dev' and a.provider=='Kev' else '')
    output=results/(name+'.json');journal=results/(name+'.journal.jsonl')
    # Each named run claims before loading data or model. Never overwrite or retry a partial run.
    with (results/(name+'.attempt.json')).open('x') as stream:json.dump({'provider':a.provider,'split':a.split,'protocol_sha256':PROTOCOL_SHA,'status':'claimed'},stream)
    chosen=None
    if a.split=='final_test':
        chosen=verify_selection(results,Path(a.training_dir));digest=chosen['selection_sha256']
        ledger=ROOT/'discrimination/results/final-attempts'/PROTOCOL_SHA
        ledger.mkdir(parents=True,exist_ok=True)
        with (ledger/(a.provider.lower()+'.json')).open('x') as stream:json.dump({'provider':a.provider,'protocol_sha256':PROTOCOL_SHA,'selection_sha256':digest,'status':'claimed_once_before_final_data_load'},stream)
    raw=(ROOT/f'discrimination/data/{a.split}.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=frozen['dataset_hashes'][a.split+'.json']:raise ValueError('dataset_integrity_failure')
    cases=json.loads(raw)['cases'];manifest=None;evidence=None;candidate_step=None
    if a.provider=='Jev':predict=jev
    else:
        from discrimination_runtime import DiscriminationModel
        weights=Path(a.weights_dir);manifest=json.loads((weights/'routing_manifest.json').read_text())
        from discrimination_evidence import candidate_evidence
        candidate_step=chosen['selected_step'] if chosen else a.step
        evidence,verified_manifest=candidate_evidence(Path(a.training_dir),candidate_step,PROTOCOL_SHA)
        if manifest!=verified_manifest:raise ValueError('candidate_weights_evidence_mismatch')
        if chosen and (manifest!=chosen['artifact_manifest'] or type(json.loads((weights/'model_config.json').read_text()).get('selected_step')) is not int or json.loads((weights/'model_config.json').read_text())['selected_step']!=chosen['selected_step']):raise ValueError('selected_weights_required')
        if not chosen and (a.step not in (0,25,50) or json.loads((weights/'model_config.json').read_text())['selected_step']!=a.step):raise ValueError('invalid_candidate_step')
        predictor=DiscriminationModel(ROOT/'models/base',weights,manifest_path=weights/'routing_manifest.json');predict=predictor.predict
    rows=[]
    with journal.open('x') as stream:
      for c in cases:
        row={k:c[k] for k in ('id','family','pair','expected')};row['input_sha256']=sha({'state':c['state'],'questions':questions()});start=time.perf_counter()
        stream.write(json.dumps({'id':c['id'],'status':'claimed_before_call'})+'\n');stream.flush()
        try:
            r=predict(c['state']);validate(r['answers'],questions());row.update(status='completed',predicted={k:r['answers'][k]['choice'] for k in questions()},answers=r['answers'],usage=r.get('usage'),reported_model=r.get('model'))
        except Exception as exc:
            code=str(exc);allowed={'credentials_unavailable','sensitive_input_blocked','transport_unavailable'};row.update(status='unavailable',error_code=code if code in allowed or code.startswith('http_') else 'invalid_response')
        row['latency_ms']=round(1000*(time.perf_counter()-start),3);rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush();print(json.dumps({'case':c['id'],'status':row['status']}),flush=True)
    m=metrics(rows);baselines={}
    for need,route in [('review','sol_high'),('review','sol_medium')]:
        constant=[dict(r,status='completed',predicted={'need':need,'route':route}) for r in rows];baselines[need+'_'+route]=metrics(constant)
    report={'provider':a.provider,'requested_model':'jev-latest' if a.provider=='Jev' else 'kev-review-discriminator-v2','split':a.split,'protocol_sha256':PROTOCOL_SHA,'dataset_sha256':hashlib.sha256(raw).hexdigest(),'questions_sha256':sha(questions()),'artifact_manifest':manifest,'candidate_step':candidate_step,'training_evidence':evidence,'selection_sha256':chosen['selection_sha256'] if chosen else None,'rows':rows,'metrics':m,'constant_baselines':baselines,'promotion_passed':promote(m),'status':'completed' if m['unavailable']==0 else 'incomplete'}
    with output.open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(m))
if __name__=='__main__':main()
