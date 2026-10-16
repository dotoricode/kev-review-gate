# SPDX-License-Identifier: Apache-2.0
"""Public synthetic routing scenarios; family partitions precede any model call."""
import argparse, hashlib, json, random
from pathlib import Path
from routing_policy import questions, CHOICES

# Each semantic scenario belongs to exactly one split; training paraphrases never cross splits.
SCENARIOS = {
 'sol_medium': [
  'Change only a tooltip spelling. The literal string comparison passes; behavior is unchanged.',
  'Adjust one button margin from 8 to 12 pixels. Screenshot confirms the requested spacing.',
  'Rename an internal local variable in a pure helper. Identical result fixtures pass.',
  'Update a README command example for an existing flag. CLI help confirms its spelling.',
  'Correct one locale translation for a cancel label; no interpolation or behavior changes.',
  'Sort a static list of supported extensions alphabetically; membership remains identical.',
  'Replace a broken documentation anchor with the verified section anchor.',
  'Add one descriptive aria label to an existing icon. DOM assertions pass.',
  'Change the default chart title literal. Serialized series and calculations are unchanged.',
  'Remove a duplicate CSS declaration with the same value. Computed styles match.',
  'Correct the casing of a help heading. No executable code changes.',
  'Update an example fixture comment. Runtime fixture values are unchanged.'],
 'sol_high': [
  'Trace pagination across controller and repository to fix an off-by-one boundary; tests cover first and last pages.',
  'Review retry backoff state across two worker paths; cancellation races are bounded and tested.',
  'Check cache invalidation after renaming a key across writer and reader; no sensitive data involved.',
  'Validate form submission state across keyboard and click paths with duplicate prevention.',
  'Review a multi-file date-filter fix across parsing and query construction, with timezone boundary tests.',
  'Check an async progress indicator cleanup across success, failure and cancellation branches.',
  'Verify sorting stability through table state and URL parameters; behavior is specified.',
  'Trace undo history updates across two editor commands and redo restoration.',
  'Review a bounded streaming parser across chunk boundaries, including empty and split tokens.',
  'Check a local file watcher debounce state machine across rename and duplicate events.',
  'Review cursor movement across wrapped lines in a text viewer; navigation contracts are clear.',
  'Trace download progress aggregation across child tasks; totals and cancellation are specified.'],
 'astra_medium': [
  'Review migration of a public client protocol across several maintained versions; compatibility commitments matter.',
  'Assess a distributed cache consistency redesign affecting several services and stale-read guarantees.',
  'Review cross-platform resource lifecycle changes spanning native wrappers and shared runtime ownership.',
  'Evaluate a broad dependency-injection boundary change affecting initialization order across modules.',
  'Review a database query planner rewrite affecting latency and semantics across many query shapes.',
  'Assess a plugin API redesign with third-party implementations and a multi-release compatibility window.',
  'Review schema evolution across offline clients, synchronization and server compatibility.',
  'Assess a scheduler architecture change spanning task fairness, accounting and multiple execution pools.',
  'Review an event-bus redesign affecting ordering guarantees across many downstream consumers.',
  'Assess serialization format evolution for long-lived archives with multiple readers and writers.',
  'Review ownership semantics across a new foreign-function interface with multiple language bindings.',
  'Assess a compiler pass redesign whose transformations interact across several optimizations.'],
 'astra_high': [
  'Review authentication token refresh changes with concurrent revocation and possible account takeover.',
  'Assess an irreversible production data migration that may delete records; rollback evidence is incomplete.',
  'Review native memory ownership changes with a suspected use-after-free under concurrent shutdown.',
  'Audit authorization enforcement across tenant boundaries after a central policy rewrite.',
  'Review cryptographic signature validation changes where malformed inputs may bypass verification.',
  'Assess a payment settlement change with duplicate debits and uncertain idempotency guarantees.',
  'Audit sandbox escape prevention after changing process privileges and file descriptor inheritance.',
  'Review secrets redaction across logs and error paths where credentials may escape.',
  'Audit certificate trust validation changes involving attacker-controlled chains and fallback rules.',
  'Assess replication failover with possible permanent data loss and ambiguous recovery semantics.',
  'Review a security policy default change allowing untrusted extension code to run.',
  'Audit encrypted backup key rotation where interruption could render all backups unrecoverable.'],
 'unknown': [
  'Review the changes. No patch, requirements or behavior description is supplied.',
  'The change is called minor and critical in conflicting notes; no implementation evidence is available.',
  'Choose a review model for an unspecified task with no risk or scope information.',
  'A reviewer reports a failure but provides no error, affected component or changed behavior.',
  'A patch is mentioned but neither its files nor intended outcome are available.',
  'The request says optimize it, without stating what it is or its constraints.',
  'Two summaries disagree about whether data is deleted; no diff is provided.',
  'The only supplied evidence is a passing status without any tested assertions or change description.',
  'The work item is blank except for a tracking number; no technical content is supplied.',
  'The change summary was truncated before describing behavior; scope cannot be inferred.',
  'One note calls it a label fix and another calls it an authentication rewrite; neither includes a patch.',
  'A binary artifact changed but its purpose, provenance and behavior are not described.']}


def sha(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def build():
 splits={k:[] for k in ('train','dev','final_test')}
 for label,texts in SCENARIOS.items():
  for index,text in enumerate(texts):
   split=('train' if index<4 else 'dev' if index<8 else 'final_test')
   variants=4 if split=='train' else 1
   for variant in range(variants):
    kind=('checkpoint','independent_final','risk_review')[(index+variant)%3]
    task=text if variant==0 else ('Review request. ' if variant==1 else 'Please check this change: ' if variant==2 else 'Change evidence: ')+text
    splits[split].append({'id':f'{split}-{label}-{index}-{variant}','family':f'{label}-{index}',
      'state':{'task':task,'review_kind':kind},'expected':label,'label_basis':'human_auditable_synthetic_policy_expectation_not_Jev_ground_truth'})
 for rows in splits.values(): random.Random(321).shuffle(rows)
 return splits


def freeze(root):
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 if (root/'protocol.json').exists():raise ValueError('protocol_exists_refusing_overwrite')
 splits=build();hashes={}
 for name,rows in splits.items():
  path=root/f'{name}.json';path.write_text(json.dumps({'cases':rows},indent=2)+'\n');hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
 protocol={'status':'frozen_before_training_and_Jev','seed':321,'steps':[0,25,50],
  'selection':'highest_dev_macro_accuracy_then_lower_high_risk_downrouting_then_earlier_step',
  'final_test':'once_only_after_dev_selection; no tuning from final scores','train_rows':80,'dev_rows':20,'final_test_rows':20,
  'Jev':'jev-latest live API; identical state/questions; dev and final 40 calls; no teacher labels; no retries',
  'labels':'Independent synthetic policy expectations; not measured code review quality',
  'training':'original pinned Kev-4B LoRA/head; frozen base bf16; adapter/head fp32; AdamW lr2e-5 weight_decay.01 clip1; warmup10; batch1; no CPU fallback',
  'max_state':384,'max_packed':512,'questions_sha256':sha(questions()),'dataset_hashes':hashes,
  'training_order':list(range(50)),'promotion':'experimental only; any risky downroute or final accuracy below .8 blocks default activation'}
 (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 return protocol

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(freeze(a.output)))
