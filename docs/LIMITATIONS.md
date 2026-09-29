# Limitations

- Frozen historical benchmark records accompany this export. The release pins
  qualified cooperative-MoE bytes; component qualification does not admit a new
  operator boot or transfer historical throughput to the new binary.
- The production-stock policy is integrated and tested offline. Mode-0
  production admission requires fresh hardware qualification of the exact
  shipped configuration. Historical validation-profile measurements do not qualify
  production. The operator clean-room observation below is narrower than full admission.
- First-pass prefill dips and greedy near-ties require the qualification
  protocol's repeat and noise checks. Quick runs are triage only.
- Prefixes just beyond a retained replay boundary may safely recompute;
  full recomputation alone is not a correctness failure.
- Runtime mode switching is unqualified here. Change mode through a restart
  and recompute conversation prefixes.
- Native binaries are excluded. Build/source identity and hardware correctness
  must be established before admission; the offline validator checks source
  pins and expected output metadata without claiming a native rebuild.

No quality-equivalence claim is made for stock mode. Future result tables
must report fresh starts and unresolved variation without scaling edited-mode
measurements into stock estimates. Keep author-reported references separate
from local measurements. Label frozen client replay and distinguish
stream-span decode rates, wave rates, GPU time and wall time.

## Known issues

The first requests after a fresh install can be about 1 s slower once, while GPU kernels (DeepGEMM) compile.

## Fixed known issues

The v1.1.0 single-request width controller could narrow a grammar-constrained
request from seven drafts to three while the grammar masks still described
the wider step. A lone `response_format: json_schema` request or forced/named
`tool_choice` could then stop the engine with a grammar-mask assertion.
The current controller retains the scheduler's width for these requests.
The v1.1.0 applicability is derived from source and a reproduction on a later
build; it was not reproduced on v1.1.0 hardware. v1.0.x lacks this controller.

The source also carries xgrammar speculative-probe and termination fixes from
vLLM pull requests 53046 and 52805. No separate claim about the presence of
that second issue on v1.1.0 hardware is made. Structured-output throughput
must be reported separately because the full-width policy differs from
unconstrained narrow speculation.

## Remaining evidence and installation limits

Greedy near-ties can produce divergent tokens. A parity disagreement requires
a cold-against-cold control on the same prompt before attributing it to prefix
reuse or an optimization. Quick Pi-shaped runs remain triage only.

Recurrent state is checkpointed at retained boundaries. Prefixes just past a
boundary can safely recompute; a delayed first token alone is not a correctness
failure. First-pass prefill and post-hygiene prefill are separate observations.
Stock free-form JSON may be fenced in Markdown and code may use backticks. Request structured output with `response_format` and `json_schema` when bare JSON is required. This formatting behavior is not a quality-equivalence claim.

Operators can build the serving image locally and bind its actual identity with
a receipt; bit-identical image rebuilds are not required or claimed. Build once
and transfer that image across your fleet. Display and coop artifacts must match
across two fresh builds; coop must also match the exact qualified release pin.
Coop is on by default. `prepare_runtime.py --coop-off` and
`build_native.py --display-only` are explicit diagnostic opt-outs.
The shipped qualification command supplies
warmup, page-cache hygiene and admission receipts. See [installation](INSTALL.md)
and [operations](OPERATIONS.md).

One v1.8.2 clean-room fleet passed verify, post-hygiene prefill and fine-hit APC,
and measured 49.05 tok/s median single-stream code decode (three repetitions,
512 tokens, temperature 0, coop off). It did not run full admission. That
historical observation does not describe the v1.8.4 coop-on default. A passing
gate run on your actual fleet is still required. Source metadata retains `hardware_qualified=false` and
`hardware_validation=not-performed` for this export. `publication_authorized`
describes the export stage, not a claim about later release approval.

The v1.8.4 release boot's full admission logs, and three JSON records that
contain private network addresses or paths, are private. They are hash-bound in
its finalization receipt and cannot be replayed by others; the published
receipts, public dependencies and owner attestation are checked by the final
validator. Operators run full admission on their own boots.

v1.8.4 throughput comes from one coop-on serving start and is compared with
v1.8.0 by observed ranges, not a statistical test. The coop on/off comparison is
pending; no uplift is claimed.

Thirds and active TRIAR are deferred to the next release. No TRIAR speedup is claimed.
