# SPDX-License-Identifier: Apache-2.0
"""Synthetic files exercise provenance rejection without loading a model or calling Jev."""
import copy
import json
import sys
from pathlib import Path
import pytest
from contract import sha
from discrimination import MODEL_ID, POLICY_VERSION, questions
from discrimination_data import build
import discrimination_eval as ev
from discrimination_evidence import candidate_evidence
from runtime import file_sha


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def experiment(tmp_path):
    training, results = tmp_path/'training', tmp_path/'results'
    lock = json.loads((ev.ROOT/'provenance/upstream-model-lock.json').read_text())
    inputs = {'base_revision':lock['base_revision'], 'original_model_revision':lock['model_revision'],
              'base_files_sha256':{k.removeprefix('base/'):v for k,v in lock['files'].items() if k.startswith('base/')},
              'original_files_sha256':{n:lock['files']['kev-4b/'+n] for n in ('adapter_config.json','adapter_model.safetensors','head.pt')}}
    for step in (0,25,50):
        stage=training/f'stage-{step:04d}'; weights=stage/'weights'; weights.mkdir(parents=True)
        config={'model_id':MODEL_ID,'policy_version':POLICY_VERSION,'selected_step':step,
                'base':lock['base_id'],'base_revision':lock['base_revision'],'tokenizer_revision':lock['base_revision'],
                'questions_sha256':sha(questions())}
        dump(weights/'model_config.json',config)
        for name in ('adapter_config.json','adapter_model.safetensors','head.safetensors'):
            (weights/name).write_text('synthetic fixture '+str(step))
        manifest={'model':MODEL_ID,'policy_version':POLICY_VERSION,
                  'public_files':{p.name:file_sha(p) for p in weights.iterdir()}}
        dump(weights/'routing_manifest.json',manifest)
        report={'status':'completed','protocol_sha256':ev.PROTOCOL_SHA,'verified_inputs':inputs,'export':manifest}
        if not step:
            report.update(step=0,initialization='pinned_original_untrained_on_routing')
        else:
            start=0 if step==25 else 25
            report.update(start=start,target=step,steps=[{'step':s,'loss':1.,'adapter_grad_norm':1.,'head_grad_norm':1.} for s in range(start+1,step+1)],
                          base_frozen=True,adapter_changed=True,head_changed=True,base_sha256_before='a'*64,base_sha256_after='a'*64)
            files={}
            for name in ('checkpoint/adapter_config.json','checkpoint/adapter_model.safetensors','checkpoint/head.pt','resume.pt'):
                p=stage/name;p.parent.mkdir(exist_ok=True);p.write_text('synthetic checkpoint');files[name]=file_sha(p)
            report['resume_files_sha256']=files
            if step==50:report['resume_source_report_sha256']=file_sha(training/'stage-0025/report.json')
        dump(stage/'report.json',report)
        evidence,_=candidate_evidence(training,step,ev.PROTOCOL_SHA)
        rows=[]
        for case in build()['dev']:
            answers={task:{'type':'choice','choice':case['expected'][task],'confidence':1.,
                           'probabilities':{label:float(label==case['expected'][task]) for label in q['criteria']}}
                     for task,q in questions().items()}
            rows.append({**{k:case[k] for k in ('id','family','pair','expected')},'status':'completed',
                         'predicted':case['expected'],'answers':answers,'input_sha256':sha({'state':case['state'],'questions':questions()})})
        dev={'status':'completed','candidate_step':step,'training_evidence':evidence,'artifact_manifest':manifest,
             'protocol_sha256':ev.PROTOCOL_SHA,'provider':'Kev','split':'dev','questions_sha256':sha(questions()),
             'dataset_sha256':ev.protocol()['dataset_hashes']['dev.json'],'rows':rows,'metrics':ev.metrics(rows)}
        dump(results/f'kev-dev-step{step}.json',dev)
    return training,results


def test_fixed_schedule_selects_earlier_step_and_revalidates_final(experiment):
    training,results=experiment
    selected=ev.selection(results,results/'selection.json',training)
    assert selected['selected_step']==0
    assert ev.verify_selection(results,training)==selected


def test_copied_or_misplaced_report_is_rejected(experiment):
    training,results=experiment
    (results/'kev-dev-step50.json').write_bytes((results/'kev-dev-step25.json').read_bytes())
    with pytest.raises(ValueError,match='candidate_report_evidence_mismatch'):
        ev.selection_result(results,training)


@pytest.mark.parametrize('change',[{'status':'safety_stop'},{'target':50},{'steps':[]},{'base_frozen':False},{'export':{}}])
def test_incomplete_wrong_step_or_unbound_training_is_rejected(experiment,change):
    training,results=experiment;p=training/'stage-0025/report.json';report=json.loads(p.read_text());report.update(change);dump(p,report)
    with pytest.raises(ValueError):ev.selection_result(results,training)


def test_model_config_step_is_verified_independently_of_filename(experiment):
    training,_=experiment
    stage=training/'stage-0050';weights=stage/'weights';config=json.loads((weights/'model_config.json').read_text());config['selected_step']=25;dump(weights/'model_config.json',config)
    manifest=json.loads((weights/'routing_manifest.json').read_text());manifest['public_files']['model_config.json']=file_sha(weights/'model_config.json');dump(weights/'routing_manifest.json',manifest)
    with pytest.raises(ValueError,match='candidate_config_mismatch'):candidate_evidence(training,50,ev.PROTOCOL_SHA)


def test_changed_checkpoint_and_resume_source_are_rejected(experiment):
    training,_=experiment
    (training/'stage-0050/resume.pt').write_text('changed checkpoint')
    with pytest.raises(ValueError,match='artifact_integrity_failure'):candidate_evidence(training,50,ev.PROTOCOL_SHA)


def test_final_recomputes_selection_instead_of_trusting_self_hash(experiment):
    training,results=experiment
    chosen=ev.selection(results,results/'selection.json',training)
    chosen['selected_step']=25;chosen['artifact_manifest']=json.loads((training/'stage-0025/weights/routing_manifest.json').read_text())
    chosen.pop('selection_sha256');chosen['selection_sha256']=sha(chosen);dump(results/'selection.json',chosen)
    with pytest.raises(ValueError,match='selection_evidence_changed'):ev.verify_selection(results,training)


def test_final_cli_rejects_changed_evidence_before_final_data_or_call(experiment,monkeypatch):
    training,results=experiment;ev.selection(results,results/'selection.json',training)
    p=training/'stage-0025/report.json';report=json.loads(p.read_text());report['elapsed_seconds']=100;dump(p,report)
    read_bytes=Path.read_bytes
    def guarded(path):
        if path.name=='final_test.json':pytest.fail('final data reached before provenance check')
        return read_bytes(path)
    monkeypatch.setattr(Path,'read_bytes',guarded)
    monkeypatch.setattr(ev,'jev',lambda state:pytest.fail('Jev called'))
    monkeypatch.setattr(sys,'argv',['discrimination_eval.py','--provider','Jev','--split','final_test','--results',str(results),'--training-dir',str(training)])
    with pytest.raises(ValueError,match='candidate_report_evidence_mismatch'):ev.main()
