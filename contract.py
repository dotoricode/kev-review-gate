"""Fixed review-gate task contract; no network client."""
import json,hashlib,math
def sha(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def validate(answers,questions):
    if set(answers)!=set(questions):raise ValueError('answer_keys')
    for k,q in questions.items():
        a=answers[k];p=a['probabilities']
        if set(a)!={'type','choice','confidence','probabilities'} or a['type']!='choice' or a['choice'] not in q['criteria'] or set(p)!=set(q['criteria']):raise ValueError('answer_contract')
        if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in [a['confidence'],*p.values()]):raise ValueError('finite_probability_required')
        if abs(sum(p.values())-1)>.02 or p[a['choice']]!=max(p.values()):raise ValueError('distribution')
