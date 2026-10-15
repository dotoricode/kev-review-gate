"""Portable experiment progress: overwrite only local status.json, never transmit logs."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent/'status.json'
def update(*args,**kwargs):
    p=json.loads(P.read_text()) if P.exists() else {'steps':[]}
    p.update(kwargs);P.write_text(json.dumps(p,indent=2))
if not P.exists():P.write_text('{"steps":[]}')
