# Synthetic data card

AI-authored invented diffs/evidence, independently AI consistency-audited.
No employer repositories, customer records or private work changes. This is not
human ground truth. New synthetic task data is Apache-2.0.

| Split | Rows | Semantic scenarios | Families | Rows/class |
|---|---:|---:|---:|---:|
| Train | 576 | 72 | 24 | 192 |
| Dev | 120 | 15 | 5 | 40 |
| Final, now consumed/public | 120 | 15 | 5 | 40 |

Each scenario has8 neutral filename variants, not8 independent examples.
Families were split before execution; exact source states do not repeat across
splits, but decision concepts recur. Balanced labels do not estimate prevalence.
Train is English/Python-heavy (504/576 rows); dev/final have more varied languages
and configurations, confounded with family holdout. This measures transfer
within a constructed task, not unseen-concept or production generalization.

Only state/question/type/ordered criteria enter the model. IDs, labels,
rationales, family/scenario/variant/language/split metadata do not. Targets are
zero-based choice indices, not appended label/rationale text.

Train/dev/protocol/manifest retain frozen source bytes.
`data/final_test.json` is reconstructed after selection/evaluation by joining
blinded120 inputs to independent parent-held scoring labels. It differs from
the original sequestered file: original rationales/language/semantic_group were
not supplied and are not invented. The old final SHA in the original manifest
describes that original author file, not this public reconstruction.
`provenance/PACKAGE_SHA256SUMS` identifies public files. All360 scores were
cross-checked against retained raw responses. The final set is now a consumed
benchmark, unsuitable as a later unseen holdout.

The earlier separate120-case suite was excluded from new training/selection;
its scores cannot be treated as directly comparable to this final suite.
See PROTOCOL_AMENDMENT.md for the explicit venue correction.
