# SPDX-License-Identifier: Apache-2.0
import copy,json
import pytest
from discrimination import MODEL_ID,questions
from discrimination_data import build
from discrimination_eval import metrics,promote,protocol
from kev_discriminate import discriminate

PAYLOAD={'task':'requirement: public demonstration','review_kind':'independent_final','candidates':[{'model':'sol','tier':'sol','efforts':['medium','high']}]}

def prediction(need='skip',route='sol_medium'):
    answers={}
    for task,choice in [('need',need),('route',route)]:
        answers[task]={'type':'choice','choice':choice,'confidence':1.,'probabilities':{label:float(label==choice) for label in questions()[task]['criteria']}}
    return {'answers':answers,'identity':{'model':MODEL_ID,'weights_sha256':'0'*64}}

def test_need_skip_preserved_without_authorizing_skip_or_forcing_high():
    result=discriminate(PAYLOAD,lambda state:prediction())
    assert result['ok'] and result['decision']=='skip'
    assert result['automatic_skip_enabled'] is False
    assert result['model_route']['choice']=='sol_medium'
    assert result['model_route']['effort']=='medium'
    assert result['probabilities']['skip']==1.
    assert result['route_probabilities']['sol_medium']==1.

def test_unknown_route_preserves_successful_need():
    result=discriminate(PAYLOAD,lambda state:prediction('review','unknown'))
    assert result['ok'] and result['decision']=='review'
    assert result['model_route'] is None and result['error']['code']=='routing_unknown'

def test_astra_is_rejected_even_if_model_could_predict_sol():
    payload=copy.deepcopy(PAYLOAD);payload['candidates'].append({'model':'astra','tier':'astra','efforts':['high']})
    assert discriminate(payload,lambda state:prediction())['error']['code']=='sol_only_catalog_required'

def test_invalid_distribution_and_identity_fail_closed():
    answer=prediction();answer['answers']['need']['probabilities']['review']=.8
    assert not discriminate(PAYLOAD,lambda state:answer)['ok']
    answer=prediction();answer['identity']['weights_sha256']='not-a-digest'
    assert not discriminate(PAYLOAD,lambda state:answer)['ok']

def test_family_split_and_counterpairs_have_no_auxiliary_label_cue():
    splits=build();families=[]
    for rows in splits.values():
        f={r['family'] for r in rows};assert not any(f&prior for prior in families);families.append(f)
        for family in f:
            quartet=[r for r in rows if r['family']==family]
            without_changes=['\n'.join(line for line in r['state']['task'].splitlines() if not line.startswith('changes:')) for r in quartet]
            assert len(quartet)==4 and len(set(without_changes))==1
            assert len({tuple(r['expected'].items()) for r in quartet})==4

def test_unavailable_is_not_removed_from_recall_denominator():
    rows=[dict(c,status='completed',predicted=c['expected']) for c in build()['dev']]
    assert promote(metrics(rows))
    row=next(r for r in rows if r['pair']=='sensitive');row['status']='unavailable';row.pop('predicted')
    m=metrics(rows)
    assert m['measured']==15 and m['sensitive_downroutes']==1
    assert m['route_macro_recall']<1 and not promote(m)

def test_always_review_high_does_not_meet_promotion():
    rows=[dict(c,status='completed',predicted={'need':'review','route':'sol_high'}) for c in build()['dev']]
    m=metrics(rows)
    assert m['paired_correct_fraction']==0 and not promote(m)

def test_training_contract_frozen_and_every_training_case_consumed():
    p=protocol();assert set(p['training_order'])==set(range(48))
    assert p['questions_sha256'] and set(questions()['route']['criteria'])=={'sol_medium','sol_high','unknown'}


def test_unavailable_profile_preserves_need_without_substituting_effort():
    payload=copy.deepcopy(PAYLOAD);payload['candidates'][0]['efforts']=['high']
    result=discriminate(payload,lambda state:prediction())
    assert result['ok'] and result['decision']=='skip'
    assert result['model_route'] is None
    assert result['error']['code']=='recommended_profile_unavailable'
    assert result['judgment']['probabilities']['skip']==1.
