# JSpark3 v1.8.0

Changes since v1.1.0 include finer prefix reuse and retained replay windows, decode-floor scheduling and aligned prefill chunks, compact drafter KV, optional display-reserve KV, batched/cooperative MoE paths, adaptive draft width, optional dense-trunk FP8, the stock production profile, and grammar/termination fixes. These are source changes; the tables do not assign a speedup to an unmeasured feature.

Versions v1.2 through v1.7.3 were internal iterations.

Thanks to @unsaltedbutter-ai for the contribution in [PR #9](https://github.com/jakejharris/jspark3/pull/9), superseded by the integrated recipe. Upstream credits and license notices remain in the source tree.

Stock weights (`ABLIT=0`) and `production-stock` are the defaults. Choose stock or edited behavior before launch. Changing modes requires a service restart and recomputes conversation prefixes. Edited weights remain an explicit opt-in supplied separately; no edited checkpoint is redistributed.

The historical measured stock cohort used a validation profile for testing. It does not qualify production admission. This sanitized source distribution retains `hardware_qualified=false`; fresh qualification of the exact shipped configuration is required before production mode-0 admission. Measurements describe the separately identified serving configuration, not a hardware test of this source export.

Thirds and active TRIAR are deferred to the next release. No TRIAR speedup is claimed. The selected source retains TRIAR code resident but inactive under the NCCL graph path. Do not enable it; the startup resident flag is distinct from the runtime OFF state. The full positive inactive-path attestation and native qualification are required.

First-pass canaries are recorded only; quick runs are diagnostic. Post-hygiene canaries block admission, together with the 1100 tok/s prefill floor, correctness, memory/no-swap and zero post-readiness compilation requirements.

Stock free-form JSON may arrive in a Markdown fence, and code may be formatted with backticks. Clients needing bare JSON should request structured output with `response_format` and `json_schema`. Grammar-constrained requests retain the full speculative width. No quality-equivalence claim is made between stock and edited weights.

The single-request grammar-width crash is fixed. Its applicability to v1.1.0 is established by source inspection and reproduction on a later build, not a v1.1.0 hardware reproduction. Greedy near-ties still require controlled parity checks. Prefixes beyond retained replay boundaries may safely recompute.

Native binaries, weights and images are obtained separately. The exact pinned image, native builds, cache preparation and operator hygiene remain installation prerequisites. A fully independent image rebuild has not been demonstrated by this export.

Original code and prose are Apache-2.0. Included derivatives retain AGPL-3.0-only and vendored headers retain MIT. The assembled service is not wholly Apache-2.0; per-file SPDX and REUSE records govern. The draft-model dependency retains its non-commercial research/evaluation restriction.
