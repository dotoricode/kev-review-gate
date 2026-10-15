"""Mock transport tests: zero model inference and zero external connections."""
import copy,json,socket
import pytest
import kev_gate as k

STATE={'synthetic_diff':'Invented evidence: comments only, no executable change.'}

def answer(label='unknown'):
    p={v:(.8 if v==label else .1) for v in ['review','skip','unknown']}
    return {'model':k.MODEL,'answers':{'need':{'type':'choice','choice':label,'confidence':.7,'probabilities':p}},
        'identity':k.expected_identity(),'status':'review_required','automatic_skip_enabled':False,
        'input_sha256':k.sha({'state':STATE,'questions':k.questions()}),'usage':{'input_tokens':123}}

def transport(monkeypatch,health=None,response=None,status=200,error=None,content_type='application/json'):
    calls=[]
    health={'status':'ready','identity':k.expected_identity()} if health is None else health
    response=answer() if response is None else response
    class Reply:
        def __init__(self,method):self.method=method;self.status=200 if method=='GET' else status
        def getheader(self,*a):return content_type
        def read(self,n):return json.dumps(health if self.method=='GET' else response).encode()
    class Connection:
        def __init__(self,host,port,timeout):assert (host,port)==('127.0.0.1',8019)
        def request(self,method,path,body,headers):self.method=method;calls.append((method,path,body,headers))
        def getresponse(self):
            if error:raise error
            return Reply(self.method)
        def close(self):pass
    monkeypatch.setattr(k.http.client,'HTTPConnection',Connection)
    return calls

def test_semantic_unknown_is_success_and_input_is_original_contract(monkeypatch):
    calls=transport(monkeypatch);r=k.judge(STATE)
    assert r['ok'] and r['decision']=='unknown' and not r['automatic_skip_enabled']
    assert [(x[0],x[1]) for x in calls]==[('GET','/health'),('POST','/v1/systemone')]
    assert json.loads(calls[1][2])=={'state':STATE,'questions':k.questions(),'model':k.MODEL}

@pytest.mark.parametrize('url',['https://127.0.0.1:8019','http://localhost:8019','http://127.0.0.1:80','http://[::1]:8019','http://127.0.0.1:8019/','http://127.0.0.1:8019?x=1','http://user@127.0.0.1:8019','https://example.invalid','http://127.0.0.1.example.invalid:8019'])
def test_disallowed_destination_never_connects(monkeypatch,url):
    calls=transport(monkeypatch)
    with pytest.raises(k.GateError,match='nonlocal_or_unsupported_endpoint'):k.judge(STATE,url=url)
    assert calls==[]

@pytest.mark.parametrize('field,value',[('model','different-model'),('base_revision','other'),('tokenizer_revision','other'),('temperature',1.0),('backbone_dtype','float32'),('eval_mode',False),('automatic_skip_enabled',True),('weights_files_sha256',{})])
def test_wrong_health_prevents_payload_transmission(monkeypatch,field,value):
    h={'status':'ready','identity':k.expected_identity()};h['identity'][field]=value;calls=transport(monkeypatch,health=h)
    with pytest.raises(k.GateError,match='model_identity_mismatch'):k.judge(STATE)
    assert len(calls)==1 and calls[0][0]=='GET' and calls[0][2] is None

@pytest.mark.parametrize('error,code',[(ConnectionRefusedError(),'service_unavailable'),(socket.timeout(),'service_timeout')])
def test_service_failures_are_errors_not_unknown(monkeypatch,error,code):
    transport(monkeypatch,error=error)
    with pytest.raises(k.GateError,match=code):k.judge(STATE)

@pytest.mark.parametrize('status,code',[(301,'redirect_refused'),(307,'redirect_refused'),(422,'inference_rejected'),(500,'service_http_error')])
def test_http_failures_no_redirect_retry_or_fallback(monkeypatch,status,code):
    calls=transport(monkeypatch,status=status)
    with pytest.raises(k.GateError,match=code):k.judge(STATE)
    assert len(calls)==2

@pytest.mark.parametrize('mutate',[
    lambda r:r.update(model='wrong'),
    lambda r:r.update(automatic_skip_enabled=True),
    lambda r:r.update(input_sha256='wrong'),
    lambda r:r['answers']['need'].update(confidence=float('nan')),
    lambda r:r['answers']['need'].update(confidence=.99),
    lambda r:r['answers']['need'].update(choice='wrong'),
    lambda r:r['answers']['need'].update(probabilities={'review':1.,'skip':1.,'unknown':1.}),
    lambda r:r.update(usage={'input_tokens':513}),
    lambda r:r['identity'].update(tokenizer_revision='changed-after-health'),
])
def test_malformed_or_changed_response_fails_closed(monkeypatch,mutate):
    r=answer();mutate(r);transport(monkeypatch,response=r)
    with pytest.raises(k.GateError):k.judge(STATE)

def test_proxy_environment_is_not_used(monkeypatch):
    for name in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy']:monkeypatch.setenv(name,'http://example.invalid:9999')
    calls=transport(monkeypatch);assert k.judge(STATE)['ok'];assert len(calls)==2

def test_cli_does_not_echo_input_paths_or_parse_errors(monkeypatch,capsys,tmp_path):
    p=tmp_path/'SENSITIVE_SENTINEL.json';p.write_text('{"synthetic_diff":"SENSITIVE_SENTINEL","unexpected":"secret"}')
    assert k.main(['judge','--input',str(p)])==2
    captured=capsys.readouterr();assert 'SENSITIVE_SENTINEL' not in captured.out+captured.err;assert str(p) not in captured.out+captured.err
    r=json.loads(captured.out);assert not r['ok'] and r['decision'] is None and r['status']=='manual'
    assert k.main(['doctor','--timeout','SENSITIVE_SENTINEL'])==2
    captured=capsys.readouterr();assert 'SENSITIVE_SENTINEL' not in captured.out+captured.err

def test_strict_input_rejects_duplicate_keys_before_network(monkeypatch,capsys,tmp_path):
    p=tmp_path/'state.json';p.write_text('{"synthetic_diff":"a","synthetic_diff":"b"}')
    calls=transport(monkeypatch);assert k.main(['judge','--input',str(p)])==2;assert calls==[]

@pytest.mark.parametrize('state',[{'raw_diff':'unvalidated key'},'text',{'synthetic_diff':5},{'synthetic_diff':'x'*(k.LIMIT+1)}])
def test_invalid_or_oversized_input_not_sent(monkeypatch,state):
    calls=transport(monkeypatch)
    with pytest.raises(k.GateError):k.judge(state)
    assert calls==[]

def test_identity_json_key_order_is_irrelevant():
    k.require_identity(dict(reversed(list(k.expected_identity().items()))))

@pytest.mark.parametrize('timeout',[0,-1,float('nan'),float('inf'),61])
def test_invalid_timeout_never_connects(monkeypatch,timeout):
    calls=transport(monkeypatch)
    with pytest.raises(k.GateError,match='invalid_timeout'):k.doctor(timeout=timeout)
    assert calls==[]

def test_fixture_mismatch_is_not_reported_as_pass(monkeypatch):
    monkeypatch.setattr(k,'judge',lambda *a,**kw:{'decision':'wrong','probabilities':{},'confidence':0})
    with pytest.raises(k.GateError,match='fixture_mismatch'):k.fixture()
