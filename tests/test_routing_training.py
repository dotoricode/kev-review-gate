# SPDX-License-Identifier: Apache-2.0
import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from routing_data import build,freeze,sha
from routing_eval import aggregate,validate_answer,evaluate,claim_final_attempt,select_checkpoint,policy_metrics
from routing_policy import CHOICES,questions

class RoutingTrainingTests(unittest.TestCase):
 def test_families_and_exact_states_do_not_cross_partitions(self):
  splits=build();self.assertEqual({k:len(v) for k,v in splits.items()},{'train':80,'dev':20,'final_test':20})
  for a,b in [('train','dev'),('train','final_test'),('dev','final_test')]:
   self.assertFalse({x['family'] for x in splits[a]} & {x['family'] for x in splits[b]})
   self.assertFalse({x['state']['task'] for x in splits[a]} & {x['state']['task'] for x in splits[b]})
  for name,rows in splits.items():
   for c in CHOICES:self.assertEqual(sum(x['expected']==c for x in rows),len(rows)//5)
   for row in rows:
    self.assertEqual(set(row['state']),{'task','review_kind'})
    for value in CHOICES:self.assertNotIn(value,row['state']['task'])
 def test_freeze_is_deterministic_and_refuses_overwrite(self):
  with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
   self.assertEqual(freeze(a),freeze(b))
   with self.assertRaises(ValueError):freeze(a)
   protocol=json.loads((Path(a)/'protocol.json').read_text());self.assertEqual(protocol['steps'],[0,25,50])
 def test_probabilityless_Jev_is_incomplete_not_surrogate(self):
  rows=build()['dev'][:1]
  report=evaluate(rows,lambda state:{'answers':{'route':{'type':'choice','choice':'unknown','confidence':1}}},'Jev')
  self.assertEqual(report['status'],'incomplete');self.assertIsNone(report['metrics']['accuracy'])
  self.assertEqual(report['rows'][0]['error_code'],'invalid_response')
 def test_malformed_API_shapes_normalize_to_ValueError(self):
  for raw in ({'answers':[]},{'answers':{'route':None}},[],None):
   with self.assertRaises(ValueError):validate_answer(raw)
 def test_final_attempt_is_once_and_selection_is_tamper_evident(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);decision={'questions_sha256':sha(questions()),'artifact_manifest':{},'selected_step':25};decision['selection_sha256']=sha(decision)
   path=root/'selected_checkpoint.json';path.write_text(json.dumps(decision))
   self.assertEqual(claim_final_attempt(path,root/'final.json','Jev',None),decision['selection_sha256'])
   with self.assertRaises(FileExistsError):claim_final_attempt(path,root/'other.json','Jev',None)
   decision['selected_step']=50;path.write_text(json.dumps(decision))
   with self.assertRaises(ValueError):claim_final_attempt(path,root/'final.json','Kev',None)
 def test_partial_call_journal_preserves_unavailable_and_duplicate_rejected(self):
  cases=build()['dev'][:1]
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'journal'
   with path.open('x') as stream:evaluate(cases,lambda state:None,'Jev',stream)
   rows=[json.loads(line) for line in path.read_text().splitlines()]
   self.assertEqual([r['status'] for r in rows],['attempt_claimed_before_provider_call','unavailable'])
  with self.assertRaises(ValueError):evaluate(cases+cases,lambda state:self.fail('must_not_call'),'Jev')
 def test_policy_metrics_do_not_hide_raw_risk_underroute(self):
  c={'id':'risk','state':{'task':'security review','review_kind':'risk_review'}}
  row={'id':'risk','expected':'astra_high','predicted':'sol_high','status':'completed','latency_ms':1}
  self.assertEqual(aggregate([row])['high_risk_downrouting'],1)
  self.assertEqual(policy_metrics([row],[c])['accuracy'],1)
  self.assertEqual(row['predicted'],'sol_high')
 def test_selection_requires_all_fixed_steps_and_uses_declared_tie_break(self):
  report={'status':'completed','metrics':{'measured':20,'macro_accuracy':.9,'high_risk_downrouting':0}}
  with self.assertRaises(ValueError):select_checkpoint({0:report})
  self.assertEqual(select_checkpoint({0:report,25:report,50:report})['selected_step'],0)
 def test_confidence_and_nonmax_choices_are_rejected(self):
  for confidence in (None,False,float('nan'),-1,2):
   with self.assertRaises(ValueError):validate_answer({'answers':{'route':{'type':'choice','choice':'sol_medium','confidence':confidence,'probabilities':{k:(1 if k=='sol_medium' else 0) for k in CHOICES}}}})
  with self.assertRaises(ValueError):validate_answer({'answers':{'route':{'type':'choice','choice':'unknown','confidence':1,'probabilities':{k:(1 if k=='sol_medium' else 0) for k in CHOICES}}}})
 def test_invalid_probabilities_rejected(self):
  for ps in ({k:.3 for k in CHOICES},{k:float('nan') for k in CHOICES}):
   with self.assertRaises(ValueError):validate_answer({'answers':{'route':{'type':'choice','choice':'unknown','probabilities':ps}}})
 def test_risky_underroute_and_unknown_are_reported_separately(self):
  rows=[{'expected':'astra_high','predicted':'sol_medium','status':'completed','latency_ms':1},
        {'expected':'astra_high','predicted':'unknown','status':'completed','latency_ms':2},
        {'expected':'astra_high','status':'unavailable'}]
  m=aggregate(rows);self.assertEqual(m['high_risk_downrouting'],1);self.assertEqual(m['unavailable'],1);self.assertEqual(m['accuracy'],0)

if __name__=='__main__':unittest.main()
