# SPDX-License-Identifier: Apache-2.0
import json,sys,tempfile,unittest,copy,shutil,builtins,hashlib,subprocess,os,ast
from unittest import mock
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from routing_data import build,freeze,sha
from routing_eval import aggregate,validate_answer,evaluate,claim_final_attempt,select_checkpoint,policy_metrics,freeze_selection,local_predictor
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
   root=Path(temp);source=Path(__file__).resolve().parents[1]/'routing/results/selected_checkpoint.json';decision=json.loads(source.read_text())
   path=root/'selected_checkpoint.json';path.write_text(json.dumps(decision))
   import routing_eval as evaluation
   with mock.patch.object(evaluation,'ROOT',root):
    self.assertEqual(claim_final_attempt(path,root/'final.json','Jev',None),decision['selection_sha256'])
    with self.assertRaises(FileExistsError):claim_final_attempt(path,root/'other.json','Jev',None)
   decision['selected_step']=50;path.write_text(json.dumps(decision))
   with self.assertRaises(ValueError):claim_final_attempt(path,root/'final.json','Kev',None)
 def test_tampered_protocol_cannot_disguise_final_as_dev_for_either_provider(self):
  import routing_eval as evaluation
  source=Path(__file__).resolve().parents[1]
  for provider in ('Jev','Kev'):
   with tempfile.TemporaryDirectory() as temp:
    root=Path(temp);(root/'routing/data').mkdir(parents=True)
    protocol=json.loads((source/'routing/data/protocol.json').read_text());hashes=protocol['dataset_hashes'];hashes['dev.json'],hashes['final_test.json']=hashes['final_test.json'],hashes['dev.json'];(root/'routing/data/protocol.json').write_text(json.dumps(protocol))
    disguised=root/'renamed-dev.json';shutil.copy2(source/'routing/data/final_test.json',disguised)
    args=['routing_eval.py','--provider',provider,'--data',str(disguised),'--output',str(root/'out.json')]
    with mock.patch.object(evaluation,'ROOT',root),mock.patch.object(sys,'argv',args),mock.patch.object(evaluation,'jev_call',side_effect=AssertionError('Jev reached')),mock.patch.object(evaluation,'local_predictor',side_effect=AssertionError('GPU reached')):
     with self.assertRaises(ValueError):evaluation.main()
 def test_freeze_requires_intact_frozen_protocol_even_with_valid_dev_reports(self):
  import routing_eval as evaluation
  source=Path(__file__).resolve().parents[1]
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);(root/'routing/data').mkdir(parents=True);(root/'provenance').mkdir()
   for name in ('dev.json','protocol.json'):shutil.copy2(source/'routing/data'/name,root/'routing/data'/name)
   shutil.copy2(source/'provenance/routing-evaluation-weights.json',root/'provenance/routing-evaluation-weights.json')
   p=root/'routing/data/protocol.json';protocol=json.loads(p.read_text());protocol['seed']=999;p.write_text(json.dumps(protocol))
   with mock.patch.object(evaluation,'ROOT',root):
    with self.assertRaises(ValueError):freeze_selection(source/'routing/results',root/'selection.json')
 def test_dataset_digest_and_case_parsing_use_one_byte_snapshot(self):
  import routing_eval as evaluation
  source=Path(__file__).resolve().parents[1]
  dev_bytes=(source/'routing/data/dev.json').read_bytes();final_text=(source/'routing/data/final_test.json').read_text();original_read=Path.read_text
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);data=root/'input.json';data.write_bytes(dev_bytes)
   def changed_second_read(path,*args,**kwargs):return final_text if path==data else original_read(path,*args,**kwargs)
   args=['routing_eval.py','--provider','Jev','--data',str(data),'--output',str(root/'out.json')]
   with mock.patch.object(sys,'argv',args),mock.patch.object(Path,'read_text',changed_second_read),mock.patch.object(evaluation,'evaluate',return_value={'metrics':{},'rows':[]}) as evaluate_mock:
    evaluation.main()
   expected=[c['id'] for c in json.loads(dev_bytes)['cases']]
   self.assertEqual([c['id'] for c in evaluate_mock.call_args.args[0]],expected)
 def test_renamed_final_still_reaches_once_only_guard_before_provider(self):
  import routing_eval as evaluation
  source=Path(__file__).resolve().parents[1]
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);(root/'routing/data').mkdir(parents=True);shutil.copy2(source/'routing/data/protocol.json',root/'routing/data/protocol.json');disguised=root/'dev-alias.json';shutil.copy2(source/'routing/data/final_test.json',disguised)
   args=['routing_eval.py','--provider','Jev','--data',str(disguised),'--output',str(root/'out.json'),'--selection',str(source/'routing/results/selected_checkpoint.json')]
   with mock.patch.object(evaluation,'ROOT',root),mock.patch.object(sys,'argv',args),mock.patch.object(evaluation,'claim_final_attempt',side_effect=FileExistsError('consumed')) as claim,mock.patch.object(evaluation,'jev_call',side_effect=AssertionError('Jev reached')):
    with self.assertRaises(FileExistsError):evaluation.main()
    claim.assert_called_once()
 def test_self_signed_alternate_selection_is_rejected(self):
  import routing_eval as evaluation
  source=Path(__file__).resolve().parents[1]/'routing/results'
  decision=json.loads((source/'selected_checkpoint.json').read_text());decision['selected_step']=50;decision['artifact_manifest']=json.loads((source/'kev-dev-step50.json').read_text())['artifact_manifest'];decision.pop('selection_sha256');decision['selection_sha256']=sha(decision)
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);path=root/'edited-selection.json';path.write_text(json.dumps(decision))
   with mock.patch.object(evaluation,'ROOT',root):
    with self.assertRaises(ValueError):claim_final_attempt(path,root/'final.json','Jev',None)
 def test_copied_selection_cannot_claim_second_final_and_legacy_is_consumed(self):
  import routing_eval as evaluation
  source=Path(__file__).resolve().parents[1]/'routing/results/selected_checkpoint.json'
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);(root/'left').mkdir();(root/'right').mkdir();left=root/'left/selection.json';right=root/'right/selection.json';shutil.copy2(source,left);shutil.copy2(source,right)
   with mock.patch.object(evaluation,'ROOT',root):
    claim_final_attempt(left,root/'first.json','Jev',None)
    with self.assertRaises(FileExistsError):claim_final_attempt(right,root/'second.json','Jev',None)
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);(root/'routing/results').mkdir(parents=True);legacy=root/'routing/results/final-attempt-jev.json';legacy.write_text('legacy consumed marker');path=root/'selection.json';shutil.copy2(source,path)
   with mock.patch.object(evaluation,'ROOT',root):
    with self.assertRaises(FileExistsError):claim_final_attempt(path,root/'final.json','Jev',None)
   self.assertEqual(legacy.read_text(),'legacy consumed marker')
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
  self.assertEqual(policy_metrics([row],[c])['original_high_risk_downrouting'],1)
  self.assertEqual(row['predicted'],'sol_high')
 def test_training_protocol_parses_the_bytes_it_authenticated(self):
  import routing_train as training
  source=Path(__file__).resolve().parents[1]/'routing/data';original_read=Path.read_text;modified=json.loads((source/'protocol.json').read_text());modified['seed']=999
  def changed_second_read(path,*args,**kwargs):return json.dumps(modified) if path==source/'protocol.json' else original_read(path,*args,**kwargs)
  with mock.patch.object(Path,'read_text',changed_second_read):self.assertEqual(training.verify_protocol(source)['seed'],321)
 def test_verified_training_cases_survive_later_disk_rewrites(self):
  import routing_train as training
  source=Path(__file__).resolve().parents[1]/'routing/data'
  with tempfile.TemporaryDirectory() as temp:
   data=Path(temp)
   for name in ('protocol.json','train.json','dev.json','final_test.json'):shutil.copy2(source/name,data/name)
   protocol,cases=training.verify_protocol(data,return_datasets=True)
   expected=json.loads((source/'train.json').read_bytes())['cases'];(data/'train.json').write_bytes((source/'final_test.json').read_bytes())
   self.assertEqual(cases['train'],expected)
   self.assertEqual(set(cases),{'train','dev','final_test'})
   self.assertEqual([c['id'] for c in cases['train'][:50]],[c['id'] for c in expected[:50]])
 def test_dev_report_validation_parses_its_authenticated_snapshot(self):
  import routing_eval as evaluation
  source=Path(__file__).resolve().parents[1];dev_path=source/'routing/data/dev.json';original_read=Path.read_text;final_text=(source/'routing/data/final_test.json').read_text()
  reports={step:json.loads((source/f'routing/results/kev-dev-step{step}.json').read_text()) for step in (0,25,50)}
  def changed_second_read(path,*args,**kwargs):return final_text if path==dev_path else original_read(path,*args,**kwargs)
  with mock.patch.object(Path,'read_text',changed_second_read):evaluation.validate_dev_reports(reports)
 def test_training_input_hashes_reject_changed_base_and_original(self):
  import routing_train as training
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);(root/'provenance').mkdir();base=root/'base';original=root/'original';base.mkdir();original.mkdir()
   (base/'model.bin').write_bytes(b'base')
   names=('adapter_config.json','adapter_model.safetensors','head.pt')
   for name in names:(original/name).write_bytes(name.encode())
   lock={'base_revision':'base-pin','model_revision':'original-pin','files':{'base/model.bin':hashlib.sha256(b'base').hexdigest(),**{'kev-4b/'+name:hashlib.sha256(name.encode()).hexdigest() for name in names}}}
   (root/'provenance/upstream-model-lock.json').write_text(json.dumps(lock))
   with mock.patch.object(training,'ROOT',root):
    training.verify_training_inputs(base,original)
    (base/'model.bin').write_bytes(b'changed')
    with self.assertRaises(ValueError):training.verify_training_inputs(base,original)
    (base/'model.bin').write_bytes(b'base');(original/'head.pt').write_bytes(b'changed')
    with self.assertRaises(ValueError):training.verify_training_inputs(base,original)
 def test_resume_integrity_rejects_mutated_checkpoint_and_missing_provenance(self):
  import routing_train as training
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);stage=root/'stage-0025';(stage/'checkpoint').mkdir(parents=True)
   names=('checkpoint/adapter_config.json','checkpoint/adapter_model.safetensors','checkpoint/head.pt','resume.pt')
   for name in names:(stage/name).write_bytes(name.encode())
   report={'status':'completed','target':25,'protocol_sha256':training.FROZEN_PROTOCOL_SHA256,'verified_inputs':{},'resume_files_sha256':{name:hashlib.sha256(name.encode()).hexdigest() for name in names}}
   (stage/'report.json').write_text(json.dumps(report));training.verify_resume(root,25,{})
   (stage/'checkpoint/head.pt').write_bytes(b'changed')
   with self.assertRaises(ValueError):training.verify_resume(root,25,{})
   report.pop('resume_files_sha256');(stage/'report.json').write_text(json.dumps(report))
   with self.assertRaises(ValueError):training.verify_resume(root,25,{})
 def test_training_target_zero_rejects_unverified_inputs_before_loading(self):
  import routing_train as training
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);freeze(root/'data')
   args=['routing_train.py','--base-dir',str(root/'wrong-base'),'--original-dir',str(root/'wrong-original'),'--data-dir',str(root/'data'),'--output',str(root/'out'),'--target','0']
   with mock.patch.object(sys,'argv',args),mock.patch('isolation.isolate',side_effect=AssertionError('unverified_input_reached_loading')):
    with self.assertRaises(ValueError):training.main()
 def test_training_protocol_guards_survive_python_optimization(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);freeze(root)
   with (root/'train.json').open('a') as stream:stream.write(' ')
   env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]))
   script="from pathlib import Path;from routing_train import verify_protocol;verify_protocol(Path("+repr(str(root))+"))"
   result=subprocess.run([sys.executable,'-O','-c',script],capture_output=True,text=True,env=env)
   self.assertNotEqual(result.returncode,0,'optimized interpreter bypassed modified dataset')
  import routing_train as training
  self.assertFalse(any(isinstance(node,ast.Assert) for node in ast.walk(ast.parse(Path(training.__file__).read_text()))),'mandatory training guards must not use assert')
 def test_artifact_validation_precedes_any_model_loading(self):
  original_import=builtins.__import__
  def guard(name,*args,**kwargs):
   if name.startswith('kev.'):raise AssertionError('GPU imports reached before artifact verification')
   return original_import(name,*args,**kwargs)
  with mock.patch('isolation.isolate'),mock.patch('routing_runtime.verify_routing_artifacts',side_effect=ValueError('artifact_integrity_failure')),mock.patch('builtins.__import__',side_effect=guard):
   with self.assertRaisesRegex(ValueError,'artifact_integrity_failure'):local_predictor('base','weights')
 def test_each_pinned_candidate_rejects_tampered_tensor_manifest_and_base(self):
  import routing_runtime as runtime
  from contract import sha as contract_sha
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);(root/'provenance').mkdir();base=root/'base';base.mkdir();(base/'model.bin').write_bytes(b'base')
   lock={'base_id':'fixture-base','base_revision':'pinned-base','files':{'base/model.bin':hashlib.sha256(b'base').hexdigest()}}
   (root/'provenance/upstream-model-lock.json').write_text(json.dumps(lock));pins={};dirs={}
   for step in (0,25,50):
    weights=root/f'weights-{step}';weights.mkdir();dirs[step]=weights
    config={'base':lock['base_id'],'base_revision':lock['base_revision'],'tokenizer_revision':lock['base_revision'],'temperature':2.406050072164233,'option_isolation':False,'special_embeddings':False,'model_id':runtime.MODEL_ID,'policy_version':runtime.POLICY_VERSION,'questions_sha256':contract_sha(questions()),'selected_step':step}
    for name in runtime.FILES:(weights/name).write_bytes(json.dumps(config).encode() if name=='model_config.json' else f'{name}-{step}'.encode())
    pins[str(step)]={'model':runtime.MODEL_ID,'policy_version':runtime.POLICY_VERSION,'public_files':{name:hashlib.sha256((weights/name).read_bytes()).hexdigest() for name in runtime.FILES}}
    (weights/'routing_manifest.json').write_text(json.dumps(pins[str(step)]))
   (root/'provenance/routing-weights.json').write_text(json.dumps(pins['25']))
   (root/'provenance/routing-evaluation-weights.json').write_text(json.dumps({'candidates':pins}))
   with mock.patch.object(runtime,'ROOT',root):
    for step in (0,25,50):
     weights=dirs[step];runtime.verify_routing_artifacts(base,weights,evaluation=True)
     target=weights/'head.safetensors';saved=target.read_bytes();target.write_bytes(b'tampered')
     with self.assertRaises(ValueError):runtime.verify_routing_artifacts(base,weights,evaluation=True)
     target.write_bytes(saved)
    with self.assertRaises(ValueError):runtime.verify_routing_artifacts(base,dirs[0])
    (base/'model.bin').write_bytes(b'tampered')
    with self.assertRaises(ValueError):runtime.verify_routing_artifacts(base,dirs[25],evaluation=True)
    (base/'model.bin').write_bytes(b'base')
    altered=copy.deepcopy(pins['25']);altered['public_files']['head.safetensors']=hashlib.sha256(b'tampered').hexdigest();(dirs[25]/'routing_manifest.json').write_text(json.dumps(altered));(dirs[25]/'head.safetensors').write_bytes(b'tampered')
    with self.assertRaises(ValueError):runtime.verify_routing_artifacts(base,dirs[25],evaluation=True)
 def test_freeze_rejects_final_stale_contract_wrong_provider_and_wrong_step(self):
  source=Path(__file__).resolve().parents[1]/'routing/results'
  mutations=[lambda r:r.update(dataset_sha256='final-digest'),lambda r:r.update(questions_sha256='stale'),lambda r:r.update(provider='Jev'),lambda r:r.update(artifact_manifest=json.loads((source/'kev-dev-step50.json').read_text())['artifact_manifest'])]
  for mutation in mutations:
   with tempfile.TemporaryDirectory() as temp:
    root=Path(temp)
    for step in (0,25,50):shutil.copy2(source/f'kev-dev-step{step}.json',root/f'kev-dev-step{step}.json')
    p=root/'kev-dev-step0.json';r=json.loads(p.read_text());mutation(r);p.write_text(json.dumps(r))
    with self.assertRaises(ValueError):freeze_selection(root,root/'selection.json')
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
