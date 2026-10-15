# SPDX-License-Identifier: Apache-2.0
"""Evaluate only a named local JSON split; never sends a request outside this process."""
import argparse,json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from runtime import ReviewGate
from pilot_metrics import aggregate

def main():
    p=argparse.ArgumentParser();p.add_argument('--base-dir',required=True);p.add_argument('--weights-dir',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--reference');a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise SystemExit('output_exists')
    data=json.loads(Path(a.data).read_text());cases=data if isinstance(data,list) else data['cases'];ref=None
    if a.reference:
        refs=json.loads(Path(a.reference).read_text());ref={x['id']:x for r in refs if r['model']=='Kev4B-pilot-step50' for x in r['rows']}
    gate=ReviewGate(a.base_dir,a.weights_dir);rows=[]
    for c in cases:
        q={'need':{'type':c['type'],'instructions':c['question'],'criteria':{x['value']:x['description'] for x in c['criteria']}}}
        t=time.perf_counter();r=gate.predict({'state':c['state'],'questions':q,'model':'kev-review-gate-step50'})
        row={k:c[k] for k in ['id','family','scenario_id','expected']};row.update(predicted=r['answers']['need']['choice'],answers=r['answers'],input_sha256=r['input_sha256'],token_ids_sha256=r['token_ids_sha256'],client_latency_ms=round((time.perf_counter()-t)*1000,3))
        if ref is not None:
            z=ref[c['id']];assert r['input_sha256']==z['input_sha256'];assert r['answers']==z['answers'],'published_output_differs_from_frozen_evaluation'
        rows.append(row)
    report={'identity':gate.identity,'rows':rows,'metrics':aggregate(rows),'frozen_reference_exact_match':ref is not None,'status':'completed'}
    out.write_text(json.dumps(report,indent=2));print(json.dumps({'status':'completed','rows':len(rows),'frozen_reference_exact_match':ref is not None}))
if __name__=='__main__':main()
