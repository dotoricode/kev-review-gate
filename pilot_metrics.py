# SPDX-License-Identifier: Apache-2.0
# Modified for public packaging: relative data/vendor paths, local-only helpers, no dashboard dependency. Training math unchanged.
from collections import Counter,defaultdict
LABELS=['review','skip','unknown']
def metrics(rows):
    confusion={g:{p:0 for p in LABELS+['invalid']} for g in LABELS}
    for row in rows:confusion[row['expected']][row['predicted']]+=1
    per={}
    for c in LABELS:
        tp=confusion[c][c];fn=sum(confusion[c].values())-tp;fp=sum(confusion[g][c] for g in LABELS if g!=c)
        precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
        per[c]={'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0.,'support':tp+fn}
    n=len(rows)
    return {'n':n,'confusion':confusion,'per_class':per,'macro_f1':sum(x['f1'] for x in per.values())/3,'balanced_accuracy':sum(x['recall'] for x in per.values())/3,'accuracy':sum(confusion[c][c] for c in LABELS)/n if n else 0.,'review_to_skip':confusion['review']['skip'],'unknown_to_skip':confusion['unknown']['skip'],'review_to_unknown':confusion['review']['unknown'],'abstention_rate':sum(r['predicted']=='unknown' for r in rows)/n if n else 0.,'invalid_count':sum(r['predicted']=='invalid' for r in rows),'invalid_rate':sum(r['predicted']=='invalid' for r in rows)/n if n else 0.}
def aggregate(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['scenario_id']].append(row)
    scenarios=[]
    for ident,group in groups.items():
        assert len(group)==8 and len({r['expected'] for r in group})==1
        counts=Counter(r['predicted'] for r in group);top=max(counts.values());ties=[k for k,v in counts.items() if v==top]
        predicted=ties[0] if len(ties)==1 else 'invalid' if 'invalid' in ties else 'unknown'
        scenarios.append({'scenario_id':ident,'family':group[0]['family'],'expected':group[0]['expected'],'predicted':predicted,'variant_counts':dict(counts),'variant_disagreement':len(counts)>1})
    return {'rows':metrics(rows),'scenarios':metrics(scenarios),'scenario_predictions':scenarios,'variant_disagreement_scenarios':sum(r['variant_disagreement'] for r in scenarios),'per_family':{family:metrics([r for r in scenarios if r['family']==family]) for family in sorted({r['family'] for r in scenarios})}}
def select(evaluations):
    base=evaluations[0];b=base['metrics']['scenarios'];eligible=[]
    for result in evaluations:
        m=result['metrics']['scenarios'];raw=result['metrics']['rows']
        ok=result['step']==0 or (m['review_to_skip']<=b['review_to_skip'] and m['unknown_to_skip']<=b['unknown_to_skip'] and m['macro_f1']>=b['macro_f1'] and m['per_class']['review']['recall']>=b['per_class']['review']['recall'] and m['per_class']['skip']['recall']>=.4 and m['per_class']['unknown']['recall']>=.4 and raw['invalid_rate']<=base['metrics']['rows']['invalid_rate'])
        if ok:eligible.append(result)
    chosen=min(eligible,key=lambda r:(r['metrics']['scenarios']['review_to_skip']+r['metrics']['scenarios']['unknown_to_skip'],r['metrics']['scenarios']['review_to_skip'],-r['metrics']['scenarios']['macro_f1'],-r['metrics']['scenarios']['balanced_accuracy'],r['step']))
    return chosen,[r['step'] for r in eligible]
