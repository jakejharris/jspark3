## Next point release (unreleased)

Fresh installs default to Brandon M. Music's
[EXL3/TR3 checkpoint](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/tree/5ab363a8dcf6405955fd5f99671e01a1c9fb124b).
The pin retains ShapleyMcg License v1.0. The pinned Mia-AiLab copy and JSpark3
mirror remain accepted alternatives, with no re-download for existing installs.
The compatibility directory and serving tensors are unchanged. Validation now
accepts the original snapshot's complete publication-file layout as well as the
mirrors' exact omissions, and checks the pinned license and model-card bytes.
See [checkpoint provenance](docs/CHECKPOINT.md) and
[installation](docs/INSTALL.md). Credits and citations identify Brandon as the
quant author; the retained upstream notices include Victor Cruz's MIT notice.

This source change has offline verification only. It adds no serving measurements
or hardware qualification; the v1.8.4 results below remain historical. Release
packaging, new receipts and publication are still pending.

# JSpark3 v1.8.4

Cooperative MoE is on by default. Default builds compile display and coop twice
and require coop to match the exact native bytes qualified and sealed for this
release. The [release pin](recipe/config/coop-release.json) and
[BUILD record](recipe/overlays/v16/coop/BUILD.json) identify the native, component
seal and measured policy. `prepare_runtime.py --coop-off` and
`build_native.py --display-only` are explicit diagnostic opt-outs.

Serving throughput is at parity with v1.8.0. Of 12 measured cells from one
admitted coop-on boot, 8 overlap the frozen v1.8.0 ranges, 2 are above and 2 are
below ([measurements](release/MEASUREMENTS-v1.8.4.md)). The coop on/off
comparison is pending; no uplift is claimed.

The component campaign includes integration/H1, memcheck, racecheck, profile
and policy checks. Operators must still admit their own exact boot before
traffic. The release boot's full admission logs are private and hash-bound in
its [finalization receipt](release/v1.8.4-admission/finalize.json). See the
[release notes](release/RELEASE-NOTES.md) for filled-in seal identities and the
[v1.8.4 record](release/results-v1.8.4.json) for its serving measurements.
Frozen v1.8.0 results are unchanged.

# JSpark3 v1.8.3

Runnable operator admission with all-rank hygiene, fine-hit prefix-cache gates,
readiness waiting and native TRIAR-inactive proof. Complete install/upgrade
instructions and explicit coop-on/coop-off measurement scope. Credentials are
redacted before failure diagnostics are saved. Publication resumes exact approved
refs and assets after partial failures; standalone HF clients default Xet off.
Experimental coop build containers are removed on cancellation.

# JSpark3 v1.8.2

Local native builds and receipt-verified runtime preparation. Operator builds
run coop off and require only the display library/probe, with two-build identity.

# JSpark3 v1.8.1

Outside operators can build the pinned Dockerfile, verify InstantTensor bytes,
and record their own local image identity. Runtime preparation binds this receipt
to the source recipe; preflight, lifecycle commands and patch installers use it
consistently. The unpublished reference image is no longer a prerequisite for
new installations. Images remain local and are not redistributed. Native artifacts
can also be built locally and staged with receipts binding their source, fixed
build recipe, image and observed output hashes. The default native build requires
only the display library and probe, each with two-build byte identity. Coop-MoE
is an experimental opt-in build because its output was not reproducible on a
native DGX Spark. Historical pins remain the default
without a native receipt. Native-receipt preparation explicitly selects coop off
in the prepared environment; operator coop-on support needs a future implementation
change. The historical coop-on gate remains intact. Hardware admission gates and
v1.8.0 results are unchanged; those results do not qualify the new coop-off setup.

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
