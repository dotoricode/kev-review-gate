# SPDX-License-Identifier: Apache-2.0
"""Stdlib-only, fixed-loopback client. Never reads a repository or starts a service."""
import argparse,hashlib,http.client,json,math,socket,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from contract import validate,sha
URL='http://127.0.0.1:8019'
MODEL='kev-review-gate-step50'
LIMIT=262144

class GateError(Exception):
    pass

def strict_json(raw):
    def pairs(items):
        out={}
        for k,v in items:
            if k in out:raise ValueError('duplicate_key')
            out[k]=v
        return out
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite')))

def expected_identity():
    lock=json.loads((ROOT/'provenance/upstream-model-lock.json').read_text())
    return {'model':MODEL,'version':'v0.1.0-experimental','backend':'mlx',
        'base_revision':lock['base_revision'],'temperature':2.406050072164233,
        'backbone_dtype':'bfloat16','head_dtype':'float32','automatic_skip_enabled':False,
        'health_schema':1,'base_model_id':lock['base_id'],'tokenizer_id':lock['base_id'],
        'tokenizer_revision':lock['base_revision'],
        'tokenizer_files_sha256':{n:lock['files']['base/'+n] for n in ['tokenizer.json','tokenizer_config.json']},
        'weights_files_sha256':json.loads((ROOT/'provenance/release-weights.json').read_text())['public_files'],
        'eval_mode':True,'max_state_tokens':384,'max_packed_tokens':512}

def require_identity(identity):
    # Canonical JSON distinguishes booleans, integers and strings as well as values.
    if not isinstance(identity,dict) or json.dumps(identity,sort_keys=True)!=json.dumps(expected_identity(),sort_keys=True):
        raise GateError('model_identity_mismatch')

def questions():
    c=json.loads((ROOT/'data/manifest.json').read_text())['contract']
    return {'need':{'type':c['type'],'instructions':c['question'],
        'criteria':{x['value']:x['description'] for x in c['criteria']}}}

def exchange(method,path,body=None,*,url=URL,timeout=15):
    # Exact literal only: no DNS, URL credentials, alternate port, query, proxy or redirect.
    if url!=URL:raise GateError('nonlocal_or_unsupported_endpoint')
    if type(timeout) not in (float,int) or not math.isfinite(timeout) or not 0<timeout<=60:
        raise GateError('invalid_timeout')
    c=http.client.HTTPConnection('127.0.0.1',8019,timeout=timeout)
    try:
        c.request(method,path,body,{'Host':'127.0.0.1:8019','Content-Type':'application/json','Connection':'close'})
        r=c.getresponse()
        if 300<=r.status<400:raise GateError('redirect_refused')
        if r.status!=200:raise GateError('inference_rejected' if r.status==422 else 'service_http_error')
        if r.getheader('Content-Type','').split(';')[0].strip()!='application/json':raise GateError('invalid_response')
        raw=r.read(1048577)
        if len(raw)>1048576:raise GateError('invalid_response')
        try:return strict_json(raw)
        except (ValueError,UnicodeError,TypeError):raise GateError('invalid_response') from None
    except (socket.timeout,TimeoutError):raise GateError('service_timeout') from None
    except (OSError,http.client.HTTPException):raise GateError('service_unavailable') from None
    finally:c.close()

def doctor(**opts):
    r=exchange('GET','/health',**opts)
    if not isinstance(r,dict) or r.get('status')!='ready':raise GateError('invalid_health_response')
    require_identity(r.get('identity'))
    return {'ok':True,'status':'ready','identity':r['identity'],'automatic_skip_enabled':False}

def judge(state,**opts):
    if not isinstance(state,dict) or set(state)!={'synthetic_diff'} or not isinstance(state['synthetic_diff'],str):raise GateError('invalid_input')
    q=questions();payload={'state':state,'questions':q,'model':MODEL}
    raw=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
    if len(raw)>LIMIT:raise GateError('input_too_large')
    doctor(**opts)  # refuse old/wrong services before sending any input
    r=exchange('POST','/v1/systemone',raw,**opts)
    if not isinstance(r,dict):raise GateError('invalid_response')
    require_identity(r.get('identity'))
    if r.get('model')!=MODEL:raise GateError('model_identity_mismatch')
    if r.get('status')!='review_required' or r.get('automatic_skip_enabled') is not False:raise GateError('unsafe_response_policy')
    if r.get('input_sha256')!=sha({'state':state,'questions':q}):raise GateError('input_hash_mismatch')
    try:
        validate(r['answers'],q)
        n=r['usage']['input_tokens']
        if type(n)!=int or not 0<n<=512:raise ValueError('tokens')
    except (ValueError,KeyError,TypeError):raise GateError('invalid_response') from None
    a=r['answers']['need']
    concentrated=(max(a['probabilities'].values())-1/3)/(1-1/3)
    if abs(a['confidence']-concentrated)>.0002:raise GateError('invalid_response')
    return {'ok':True,'status':'review_required','decision':a['choice'],
        'confidence':a['confidence'],'probabilities':a['probabilities'],
        'automatic_skip_enabled':False,'input_sha256':r['input_sha256'],
        'usage':r['usage'],'identity':r['identity'],
        'scope':'Experimental synthetic evidence contract; raw git diff and new domains are unvalidated.'}

def fixture(**opts):
    cases=json.loads((ROOT/'tests/fixtures/skill_cases.json').read_text())
    refs={x['id']:x for r in json.loads((ROOT/'results/final_predictions.json').read_text()) if r['model']=='Kev4B-pilot-step50' for x in r['rows']}
    rows=[]
    for case in cases:
        r=judge(case['state'],**opts);a=refs[case['id']]['answers']['need']
        match=r['decision']==a['choice'] and r['probabilities']==a['probabilities'] and r['confidence']==a['confidence']
        if not match:raise GateError('fixture_mismatch')
        rows.append({'id':case['id'],'decision':r['decision'],'frozen_answer_match':True})
    return {'ok':True,'status':'synthetic_fixture_passed','actual_model_calls':len(rows),'rows':rows,'automatic_skip_enabled':False}

class Parser(argparse.ArgumentParser):
    def error(self,message):raise GateError('invalid_arguments')

def main(argv=None):
    try:
        p=Parser(description=__doc__);p.add_argument('command',choices=['doctor','judge','fixture'])
        p.add_argument('--input',help='UTF-8 state JSON file; - reads stdin')
        p.add_argument('--url',default=URL,help='Only the exact default loopback URL is accepted')
        p.add_argument('--timeout',type=float,default=15)
        a=p.parse_args(argv)
        if a.url!=URL:raise GateError('nonlocal_or_unsupported_endpoint')
        opts={'url':a.url,'timeout':a.timeout}
        if a.command=='judge':
            if a.input is None:raise GateError('input_required')
            try:
                if a.input=='-':raw=sys.stdin.buffer.read(LIMIT+1)
                else:
                    with Path(a.input).open('rb') as f:raw=f.read(LIMIT+1)
                if len(raw)>LIMIT:raise GateError('input_too_large')
                state=strict_json(raw)
            except (OSError,ValueError,UnicodeError):raise GateError('invalid_input') from None
            result=judge(state,**opts)
        else:
            if a.input is not None:raise GateError('invalid_arguments')
            result=doctor(**opts) if a.command=='doctor' else fixture(**opts)
        print(json.dumps(result,allow_nan=False));return 0
    except GateError as err:code=str(err)
    except Exception:code='local_client_error'
    print(json.dumps({'ok':False,'status':'manual','decision':None,'error':{'code':code},'automatic_skip_enabled':False}));return 2

if __name__=='__main__':raise SystemExit(main())
