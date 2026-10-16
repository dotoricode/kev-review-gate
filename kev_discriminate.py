# SPDX-License-Identifier: Apache-2.0
"""Opt-in local advisory; preserves need and route probabilities separately."""
import argparse,json,os,re,sys
from pathlib import Path
from contract import validate
from discrimination import questions,MODEL_ID,POLICY_VERSION
from routing_policy import validate_request,select
ROOT=Path(__file__).resolve().parent

def discriminate(payload,predictor=None,*,base_dir=None,weights_dir=None):
    try:
        state=validate_request(payload)
        if any(c['tier']!='sol' for c in payload['candidates']):raise ValueError('sol_only_catalog_required')
        if predictor is None:
            from discrimination_runtime import DiscriminationModel
            predictor=DiscriminationModel(base_dir or ROOT/'models/base',weights_dir or ROOT/'weights/kev-discrimination').predict
        raw=predictor(state);validate(raw['answers'],questions());need=raw['answers']['need'];route=raw['answers']['route'];identity=raw['identity']
        if identity.get('model')!=MODEL_ID or not isinstance(identity.get('weights_sha256'),str) or not re.fullmatch('[0-9a-f]{64}',identity['weights_sha256']):raise ValueError('discrimination_identity_mismatch')
        result={'ok':True,'status':'review_required','automatic_skip_enabled':False,'decision':need['choice'],'confidence':need['confidence'],'probabilities':need['probabilities'],'judgment':need,'model_route':None,'route_probabilities':route['probabilities'],'route_judgment':route,'identity':identity,'usage':raw.get('usage',{}),'scope':'Experimental advice. Caller review policy and execution authorization remain separate.'}
        if route['choice']=='unknown':result['error']={'code':'routing_unknown'}
        else:
            try:
                chosen=select(route['choice'],payload)
            except ValueError as exc:
                if str(exc) not in {'recommended_profile_unavailable','required_profile_unavailable'}:raise
                result['error']={'code':str(exc)}
            else:
                chosen.update(policy_version=POLICY_VERSION,decision_source='kev',confidence=route['confidence'],router_model=MODEL_ID,weights_sha256=identity['weights_sha256']);result['model_route']=chosen
        return result
    except Exception as exc:
        allowed={'invalid_routing_input','invalid_review_kind','invalid_model_catalog','sol_only_catalog_required','discrimination_identity_mismatch','discrimination_input_too_large','recommended_profile_unavailable'}
        return {'ok':False,'status':'manual','automatic_skip_enabled':False,'model_route':None,'error':{'code':str(exc) if str(exc) in allowed else 'discrimination_unavailable'}}

def main(argv=None):
    from kev_gate import Parser,strict_json
    try:
        p=Parser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--base-dir',default=str(ROOT/'models/base'));p.add_argument('--weights-dir',default=os.environ.get('KEV_DISCRIMINATION_WEIGHTS',str(ROOT/'weights/kev-discrimination')));a=p.parse_args(argv)
        if a.input=='-':raw=sys.stdin.buffer.read(262145)
        else:
            with Path(a.input).open('rb') as f:raw=f.read(262145)
        if len(raw)>262144:raise ValueError('input_too_large')
        result=discriminate(strict_json(raw),base_dir=a.base_dir,weights_dir=a.weights_dir)
    except Exception:result={'ok':False,'status':'manual','automatic_skip_enabled':False,'model_route':None,'error':{'code':'invalid_discrimination_request'}}
    print(json.dumps(result,allow_nan=False));return 0 if result['ok'] else 2
if __name__=='__main__':raise SystemExit(main())
