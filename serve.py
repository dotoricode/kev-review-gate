# SPDX-License-Identifier: Apache-2.0
"""Fixed IPv4 loopback server. Payloads and errors are never logged."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from http.server import BaseHTTPRequestHandler,HTTPServer
from runtime import ReviewGate
FAIL={'status':'manual','decision':'review_required','automatic_skip_enabled':False}
class Handler(BaseHTTPRequestHandler):
    gate=None
    def do_GET(self):
        if self.path!='/health' or self.headers.get('Host')!='127.0.0.1:8019' or self.headers.get('Origin') or self.headers.get('Transfer-Encoding'):
            result=FAIL;code=422
        else:result={'status':'ready','identity':self.gate.identity};code=200
        raw=json.dumps(result,allow_nan=False).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    def do_POST(self):
        try:
            if self.path!='/v1/systemone' or self.headers.get('Host')!='127.0.0.1:8019' or self.headers.get('Origin') or self.headers.get('Transfer-Encoding'):raise ValueError('loopback_request_only')
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<=262144:raise ValueError('request_size')
            def pairs(items):
                out={}
                for k,v in items:
                    if k in out:raise ValueError('duplicate_json_key')
                    out[k]=v
                return out
            payload=json.loads(self.rfile.read(n),object_pairs_hook=pairs,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite_json')))
            result=self.gate.predict(payload);code=200
        except Exception:result=FAIL;code=422
        raw=json.dumps(result,allow_nan=False).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    def log_message(self,*args):pass

def main():
    p=argparse.ArgumentParser();p.add_argument('--base-dir',required=True);p.add_argument('--weights-dir',required=True);a=p.parse_args()
    try:Handler.gate=ReviewGate(a.base_dir,a.weights_dir)
    except Exception:print(json.dumps(FAIL));raise SystemExit(1)
    print('Review Gate listening on http://127.0.0.1:8019; automatic skip disabled',flush=True)
    HTTPServer(('127.0.0.1',8019),Handler).serve_forever()
if __name__=='__main__':main()
