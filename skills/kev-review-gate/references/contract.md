# Fixed input and result contract

Input file/stdin is exactly `{"synthetic_diff": "bounded change evidence"}`.
This historical field name is retained for input compatibility; placing real
diff text there does not make it synthetic or validated. The approved agent
chooses evidence within its existing access authorization. The CLI itself does
not discover files, execute git, read a repository or persist inputs/results.

The CLI wraps it as `state` and constructs the original question:
Does this commit need independent code review? Skip only when supplied evidence
establishes no behavior, interface, security, or data effect. Missing evidence
means unknown.

Ordered criteria remain `review` (behavior/interface/security change), `skip`
(wording/formatting with no behavior effect), `unknown` (insufficient evidence).
Exact text is loaded from the checked-in manifest. No free-form prompt option
is exposed. Metadata/gold labels/rationales never enter the request.

Commands: `doctor`, `judge --input FILE` (or `-`), `fixture`. Optional `--timeout`
is finite, greater than0 and at most60 seconds. The only accepted `--url` is
`http://127.0.0.1:8019`; default is that exact literal. DNS aliases, alternative
ports, IPv6, URL credentials, proxies, redirects and external URLs are not used.
No automatic retries or fallback. Health runs before input POST and identity
is checked again on the model response. Health is service self-report after
startup file verification, not attestation against a malicious local process.

Success returns `ok:true`, `status:review_required`, `decision`, `confidence`,
`probabilities`, safe identity/token usage and input hash. Automatic skip is
false. Confidence measures concentration among the supplied alternatives;
it is not a validated probability that the class or workflow is correct.

Failures return exit2 and `ok:false`, `status:manual`, `decision:null`,
`error.code`, disabled skip. Codes distinguish `service_unavailable`,
`service_timeout`, `redirect_refused`, `inference_rejected`,
`model_identity_mismatch`, `invalid_response`, `input_hash_mismatch`,
`invalid_input`, `input_too_large`, `nonlocal_or_unsupported_endpoint` and
other local protocol errors. They do not echo input, paths or response bodies.
An error must not be relabeled as semantic unknown, nor treated as permission
to skip review. Maintain ordinary manual review.

The exact evaluated model uses the pinned base tokenizer revision
1001bb4d826a52d1f399e183466143f4da7b741b and temperature2.406050072164233.
Model/head/base/tokenizer hashes and MLX bf16/fp32/eval state are in doctor
identity. The judge path uses no candidate-saved tokenizer or dynamic model selection.
The separate route command has its own trained weights and contract; read
`routing.md` when recommending a review model or reasoning effort.
