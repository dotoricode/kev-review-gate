"""HTTP fail-closed checks without loading model weights."""
import json,threading,http.client
from http.server import HTTPServer
from serve import Handler
class Stub:
    def predict(self,payload):
        if payload!={'synthetic_probe':True}:raise ValueError('bad_contract')
        return {'status':'review_required','automatic_skip_enabled':False}
def call(server,body,host='127.0.0.1:8019',path='/v1/systemone',origin=None):
    c=http.client.HTTPConnection('127.0.0.1',server.server_port);h={'Host':host,'Content-Type':'application/json'}
    if origin:h['Origin']=origin
    c.request('POST',path,body,h);r=c.getresponse();out=(r.status,json.loads(r.read()));c.close();return out
def test_fixed_host_contract_and_safe_failure():
    class TestHandler(Handler):gate=Stub()
    server=HTTPServer(('127.0.0.1',0),TestHandler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        assert call(server,'{"synthetic_probe":true}')[0]==200
        for body,host,path,origin in [('{}','127.0.0.1:8019','/v1/systemone',None),('{','127.0.0.1:8019','/v1/systemone',None),('{"a":1,"a":2}','127.0.0.1:8019','/v1/systemone',None),('{"x":NaN}','127.0.0.1:8019','/v1/systemone',None),('{}','example.invalid','/v1/systemone',None),('{}','127.0.0.1:8019','/wrong',None),('{}','127.0.0.1:8019','/v1/systemone','https://example.invalid')]:
            code,result=call(server,body,host,path,origin);assert code==422;assert result=={'status':'manual','decision':'review_required','automatic_skip_enabled':False}
    finally:server.shutdown();server.server_close();thread.join()
