# Changelog

All notable changes to JSpark3 v1 are recorded here. Versions follow semantic
versioning; the serving envelope, pinned inputs, and transform contract are
part of the public interface.

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
