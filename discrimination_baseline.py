# SPDX-License-Identifier: Apache-2.0
"""Existing operating models on the new dev evidence; question contracts differ."""
import argparse,json,time
from pathlib import Path
from collections import Counter
from contract import sha
from discrimination_eval import protocol
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['need','route'],required=True);parser.add_argument('--output',required=True);a=parser.parse_args();protocol();output=Path(a.output)
    with output.with_suffix('.attempt.json').open('x') as f:json.dump({'task':a.task,'split':'dev','status':'claimed'},f)
    cases=json.loads((ROOT/'discrimination/data/dev.json').read_text())['cases']
    if a.task=='need':
        from runtime import ReviewGate
        model=ReviewGate(ROOT/'models/base',ROOT/'weights/kev-review-gate-step50');c=json.loads((ROOT/'data/manifest.json').read_text())['contract'];q={'need':{'type':'choice','instructions':c['question'],'criteria':{x['value']:x['description'] for x in c['criteria']}}}
        def predict(state):return model.predict({'state':{'synthetic_diff':state['task']},'questions':q,'model':'kev-review-gate-step50'})
    else:
        from routing_runtime import RoutingModel
        model=RoutingModel(ROOT/'models/base',ROOT/'weights/kev-routing');predict=model.predict
    rows=[]
    for case in cases:
        row={k:case[k] for k in ('id','family','pair','expected')};start=time.perf_counter()
        try:
            raw=predict(case['state']);answer=raw['answers'][a.task];row.update(status='completed',choice=answer['choice'],answer=answer)
            if a.task=='route':
                from routing_policy import select
                try:row['policy']=select(answer['choice'],dict(case['state'],candidates=[{'tier':'sol','model':'gpt-6.1-sol','efforts':['medium','high']}]))
                except ValueError as exc:row['policy_error']=str(exc)
        except Exception:row.update(status='unavailable')
        row['latency_ms']=round(1000*(time.perf_counter()-start),3);rows.append(row)
    report={'task':a.task,'split':'dev','model_identity':model.identity,'rows':rows,'choices':dict(Counter(r['choice'] for r in rows if r['status']=='completed')),'measured':sum(r['status']=='completed' for r in rows),'expected_label_agreement':sum(r.get('choice')==r['expected'][a.task] for r in rows)/len(rows),'scope':'Same new dev evidence, old fixed question contracts and label spaces. Descriptive baseline, not strict same-question accuracy comparison; no Astra-to-Sol remapping.'}
    with output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','model_identity')}))
if __name__=='__main__':main()
