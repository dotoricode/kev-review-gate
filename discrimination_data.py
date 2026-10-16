# SPDX-License-Identifier: Apache-2.0
"""Fresh paired synthetic evidence, split by component family before any inference."""
import hashlib, json, random
from pathlib import Path
from contract import sha
from discrimination import questions

# Each tuple is a separate semantic component family; no tuple crosses splits.
FAMILIES = {
 'train': [
 ('session badge', 'Fix a session caption typo.', 'Return a dash for an absent display name.', 'Change expired-session rejection to acceptance in concurrent refresh.'),
 ('backup viewer', 'Correct backup heading capitalization.', 'Count empty preview entries as zero.', 'Delete source backups before durable copy acknowledgement.'),
 ('payment label', 'Rename the static payment heading.', 'Format zero cents using two digits.', 'Retry settlement after timeout without checking transaction idempotency.'),
 ('token hint', 'Correct punctuation in token help.', 'Render a missing expiry date as blank.', 'Accept unsigned tokens in a validation fallback.'),
 ('permission tooltip', 'Fix spelling in an admin tooltip.', 'Hide an empty permission list placeholder.', 'Share cached authorization results across tenant identifiers.'),
 ('memory chart', 'Correct the memory chart axis title.', 'Clamp a displayed negative sample to zero.', 'Free a native buffer while a reader thread still holds its pointer.'),
 ('upload counter', 'Correct the upload label grammar.', 'Format single-byte count with singular text.', 'Reuse a file handle after cancellation closes it concurrently.'),
 ('account banner', 'Fix whitespace in account banner text.', 'Omit a display-only suffix for empty usernames.', 'Allow disabled accounts through a secondary login path.'),
 ('file preview', 'Correct a preview empty-state sentence.', 'Return an empty preview for a zero-length string.', 'Resolve uploaded paths without enforcing the storage root boundary.'),
 ('queue panel', 'Fix the static queue title.', 'Render a zero queue length as zero.', 'Remove a lock around shared dequeue and ownership transfer.'),
 ('certificate page', 'Correct certificate help wording.', 'Format an absent issuer as unknown text.', 'Trust an attacker supplied root certificate by default.'),
 ('cache table', 'Correct the cache table caption.', 'Sort an already isolated view of display names.', 'Cache decrypted secrets in a globally shared map without isolation.')],
 'dev': [
 ('archive inspector', 'Fix an archive heading typo.', 'Display an empty archive size as zero.', 'Overwrite the only archive index before validating replacement data.'),
 ('tenant menu', 'Correct menu item punctuation.', 'Remove a trailing separator for a one-item menu.', 'Use another tenant identity when an access lookup fails.'),
 ('retry widget', 'Fix the retry button label.', 'Display attempt zero instead of blank.', 'Mutate retry ownership while cancellation and completion race.'),
 ('key status', 'Correct key status caption casing.', 'Render a missing display timestamp as a dash.', 'Log private key material in an exception fallback.')],
 'final_test': [
 ('audit timeline', 'Correct the timeline legend text.', 'Format an empty event count as zero.', 'Remove integrity verification when importing audit records.'),
 ('restore dialog', 'Fix a restore confirmation typo.', 'Show zero selected files instead of empty text.', 'Erase the recovery key before interruption-safe restore completion.'),
 ('quota meter', 'Correct the quota unit label.', 'Round a display-only fraction to two decimal places.', 'Update shared quota counters without atomic reservation across workers.'),
 ('role selector', 'Correct a role option description.', 'Omit an empty display-only group heading.', 'Treat an unknown role as privileged in a policy fallback.')]
}

def build():
    result={}
    for split,families in FAMILIES.items():
      rows=[]
      for i,(family,inert,bounded,sensitive) in enumerate(families):
        ko=i%2==1
        requirement=(f'{family} 상태를 표시한다.' if ko else f'Display status in the {family}.')
        for variant in ('inert','bounded','sensitive','missing'):
          details={
            'inert': (inert+' Executable conditions and values are unchanged.', 'one literal', '동작 변화 없음' if ko else 'no behavioral change', '실행 경로 동일성 및 문자열 검증 통과' if ko else 'literal assertion and unchanged executable paths verified'),
            'bounded': (bounded+' A single pure display helper changes.', 'one pure display function', '표시 경계값 동작 변경' if ko else 'display boundary behavior changes', '수정 전 경계값 실패, 수정 후 0과 양수 사례 통과' if ko else 'boundary fails before; zero and positive fixtures pass after'),
            'sensitive': (sensitive, 'sensitive enforcement or concurrent lifecycle paths', '민감한 동작 또는 복구 보장 변경' if ko else 'sensitive behavior or recovery guarantees change', '정상 사례만 통과; 실패·경합 미검증' if ko else 'happy path passes; failure and race paths unverified'),
            'missing': (f'Notes conflict about {family}: a literal edit or a sensitive behavior change. No patch supplied.', '', '확인 불가' if ko else 'unestablished', '단언 내용 없는 통과 상태' if ko else 'passing status without tested assertions')}
          changes,scope,behavior,verification=details[variant]
          # Matched auxiliary evidence across each quartet: actual changed behavior must discriminate.
          scope='one component'
          behavior='위 changes 항목 참조' if ko else 'See actual changes above'
          verification=('all stated tests pass; exact coverage unspecified' if i%2==0 else 'no regression test run; static patch inspection only')
          values={'requirement':requirement,'changes':changes,'callers':('컴포넌트의 기존 호출 경로' if ko else 'existing component entry points'), 'verification':verification,'change_scope':scope,'behavior_impact':behavior}
          state={'task':'\n'.join(f'{k}: {v}' for k,v in values.items()),'review_kind':('checkpoint','independent_final','risk_review')[i%3]}
          expected={'need': 'skip' if variant=='inert' else 'unknown' if variant=='missing' else 'review','route':'unknown' if variant=='missing' else 'sol_high' if variant=='sensitive' else 'sol_medium'}
          rows.append({'id':f'{split}-{i}-{variant}','family':family,'pair':variant,'language':'mixed-ko-en' if ko else 'en','state':state,'expected':expected})
      random.Random(722).shuffle(rows);result[split]=rows
    return result

def freeze(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists():raise FileExistsError('protocol_already_frozen')
    hashes={};splits=build()
    for split,rows in splits.items():
      path=output/(split+'.json');path.write_text(json.dumps({'cases':rows},ensure_ascii=False,indent=2)+'\n');hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    protocol={'status':'frozen_before_training_and_inference','seed':722,'steps':[0,25,50], 'questions_sha256':sha(questions()),'dataset_hashes':hashes,'training_order':[i%len(splits['train']) for i in range(50)],
      'selection':'dev joint macro recall, then sensitive downroutes, then earlier step', 'final':'once each provider after dev selection; no retries or tuning from final',
      'promotion':{'minimum_need_macro_recall':.8,'minimum_route_macro_recall':.8,'sensitive_downroutes':0,'missing_false_precision':0,'inert_false_review_max':0,'paired_correct_fraction_min':.75,'all_cases_measured':True},
      'comparison':'actual jev-latest API, same state/questions; public synthetic only; no teacher labels; dev16+final16 requests; errors separate',
      'limits':{'state':384,'packed':512,'updates':50,'available_memory_gib_min':4,'swap_delta_mib_max':512,'segment_seconds_max':900},
      'scope':'Synthetic policy expectations, not measured review defect detection. English/Korean paired cases; family-disjoint splits. Final inputs may be token-length checked but never used in training or selection.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');return protocol

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(freeze(a.output)))
