# SPDX-License-Identifier: Apache-2.0
# Modified for public packaging: relative data/vendor paths, local-only helpers, no dashboard dependency. Training math unchanged.
from pilot_metrics import aggregate,metrics,select
def rows(preds):return [{'scenario_id':'s','family':'f','expected':'review','predicted':p} for p in preds]
def test_valid_tie_becomes_unknown():
    r=aggregate(rows(['review']*4+['skip']*4));assert r['scenario_predictions'][0]['predicted']=='unknown'
def test_invalid_tie_stays_invalid():
    r=aggregate(rows(['review']*4+['invalid']*4));assert r['scenario_predictions'][0]['predicted']=='invalid'
    assert r['scenarios']['per_class']['review']['recall']==0
def test_invalid_counts_as_false_negative():
    r=metrics([{'expected':'review','predicted':'invalid'},{'expected':'skip','predicted':'skip'},{'expected':'unknown','predicted':'unknown'}])
    assert r['accuracy']==2/3 and r['invalid_count']==1 and r['per_class']['review']['recall']==0
def test_equal_results_choose_baseline():
    m={'scenarios':metrics([{'expected':c,'predicted':c} for c in ['review','skip','unknown']]),'rows':{'invalid_rate':0}}
    result,eligible=select([{'step':0,'metrics':m},{'step':50,'metrics':m}]);assert result['step']==0 and eligible==[0,50]
