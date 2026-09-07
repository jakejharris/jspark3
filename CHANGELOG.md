# Changelog

All notable changes to JSpark3 are recorded here. Versions follow semantic
versioning; the serving envelope, pinned inputs, and transform contract are
part of the public interface.

## v1.1.0 (Cadence) - Unreleased

Staged for publication; the `v1.1.0` tag and release are created at
publication and none of the items below are claimed as released before then.
Weights, checkpoints, and image digests are unchanged; the historical v1.0.0
benchmark results keep their existing identity and are not remeasured or
merged with the v1.1 evidence.

- **Cadence serving features.** The measured launch adds a group-128 INT8
  shadow for the QKV projections on small-batch pure decode and a
  request-local speculative width controller that narrows speculation from
  seven to three tokens on high acceptance. Batches of two or more requests
  and prefill fall back to the wide path. The stock indexer workspace, the
  original drafter, and the sampler are retained.
- **Measured scope.** Paired within-instance single-stream effects across two
  independent serving starts: prose +16.18% and +19.17%, structured count
  +7.66% and +7.61% (95% CIs exclude zero); code +9.71% then +3.23% with the
  second interval spanning zero, so the paired code gain is not replicated.
  A fixed quality battery showed candidate-only delivered-answer failures and
  remains inconclusive on population-level semantic parity. A short-prompt
  concurrency burst (183 requests, up to 24 concurrent streams, aggregate
  263.39 tok/s, per-stream 15.38 tok/s) is finite-burst evidence and certifies
  neither sustained service, per-stream fairness, the 32-sequence envelope,
  nor large-context capacity.
- **Long-context kernel fix carried.** The v1.0.1 transform-level disable of
  vLLM's `persistent_topk` kernel is part of the v1.1 transform contract; the
  transform emits the exact file the measured arms executed. A live
  single-stream witness above 32,768 tokens on the assembled public build
  remains a pending release-verification item and is not claimed as passed.
- **Documentation corrections carried into this release:** the unverified
  stub-`libcuda` causal claim in the v1.0.1 install notes is reworded as
  unconfirmed; the v1.0.1 operations wording implying proven 1,000,000-token
  reachability and whole-tree byte identity is corrected to what the evidence
  shows (known abort removed; the indexer transform output matches the
  measured bytes); the Hub mirror revision is described as the verified
  weights-mirror revision; and the v1.0.0 gate page's inaccurate "identifier
  renames only" derivation claim is explicitly corrected by a provenance note
  while the historical record stays unmodified.
- Recipe-side packaging (modules, manifests, checksums, verification tooling)
  lands with the recipe lane and is reconciled at integration.

## v1.0.1 - 2026-09-06

Documentation and recipe-contract fix for two issues found after v1.0.0 by a
community bug report from @BTCXoomer on X. Weights and benchmarks are
unchanged: no number was remeasured, and every published figure already came
from the disabled-kernel path.

- The v1.0.0 transform contract pinned `sparse_attn_indexer_kpool.py` with
  vLLM's `persistent_topk` kernel enabled. On GB10 that kernel aborts on any
  single-stream request whose context passes 32,768 tokens (85 cooperative
  CTAs against 48 resident; the 128 KiB fallback exceeds the device's
  101,376-byte shared-memory opt-in). v1.0.1 changes the transform to emit
  the one-line GB10 disable, the exact file every measured arm ran, and
  re-pins the patch contract, `SHA256SUMS`, and the recipe manifest.
- `docs/INSTALL.md` now names `NCCL_IB_SUBNET_AWARE_ROUTING=1` in the install
  steps. New in NCCL 2.30.7 and off by default, it is required for a
  switchless three-node triangle; without it NCCL pairs NICs by index and
  routes rank 0 to rank 2 over rank 1's leg. The controller always set it;
  the docs never said so. The full controller-injected fabric environment is
  now listed in step 5, and the docs state that preloading or substituting a
  different NCCL build is outside the verified recipe.
- Warnings for both issues moved into the install path (README, INSTALL,
  OPERATIONS) instead of living only in LIMITATIONS.
- No weights, checkpoints, image digests, or benchmark figures changed. The
  serving envelope (TP3, EP3, DFlash2 k=7, FP8 KV, 1,000,000-token
  configured context) is unchanged.

## v1.0.0 - 2026-09-02

First public release:
<https://github.com/jakejharris/jspark3/releases/tag/v1.0.0>.

- Three-DGX-Spark serving recipe for GLM-5.3 Flash: tensor parallel 3 and
  expert parallel 3 over a two-leg RoCE-v2 triangle, EXL3/TR3 4-bpw target,
  DFlash2 k=7 draft, FP8 KV cache, prefix caching, 1,000,000-token configured
  context.
- Fail-closed lifecycle controller: per-rank preflight bound to a checksum,
  hash-bound release manifest, deterministic container names, host-minted image
  receipt, verify path with shard, graph, and witness checks.
- Five hash-gated in-container runtime transforms reconstructed from pinned
  upstream revisions, plus the selective W8A16 Marlin trunk overlay
  (169 runtime modules / 225 logical tensors; KDA f/g modules excluded).
- Sanitized, machine-readable evidence set under `results/` with a claim
  reconciliation map, and a release validator that regenerates checksums,
  scans for private data, and runs the lifecycle dry-runs.
- Local-only runtime image build definition, CI validation, SBOM generation,
  and reproducible release-asset packaging. No JSpark3 GHCR image is published
  for v1.0.0; the recipe uses the exact upstream image by digest.

Not included: model weights, tokenizer files, container layers, or any patched
vLLM tree. See `docs/LICENSING.md` for the third-party terms that apply.
