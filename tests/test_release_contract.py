import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_metadata_not_input():
    for split in ['train','dev','final_test']:
        rows=json.loads((ROOT/'data'/f'{split}.json').read_text())['cases']
        assert all(set(x['state'])=={'synthetic_diff'} for x in rows)
def test_semantic_counts_and_consumed_test():
    for split,n,s in [('train',576,72),('dev',120,15),('final_test',120,15)]:
        x=json.loads((ROOT/'data'/f'{split}.json').read_text());assert len(x['cases'])==n;assert len({r['scenario_id'] for r in x['cases']})==s
    assert 'PUBLIC_CONSUMED' in x['status']
def test_frozen_safety_defaults():
    import sys
    sys.path.insert(0,str(ROOT));from serve import FAIL
    assert FAIL=={'status':'manual','decision':'review_required','automatic_skip_enabled':False}
def test_upstream_source_hashes():
    r=json.loads((ROOT/'provenance/source-files.json').read_text())
    for name,sha in r['unmodified_vendor_files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
