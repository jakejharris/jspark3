# JSpark3 v1.0.0

Released 2026-09-02:
<https://github.com/jakejharris/jspark3/releases/tag/v1.0.0>

## Summary

JSpark3 v1 is a reproducible, fail-closed recipe that serves GLM-5.3 Flash
across three NVIDIA DGX Sparks as one OpenAI-compatible endpoint: tensor
parallel 3, expert parallel 3, a two-leg RoCE-v2 triangle, EXL3/TR3 4-bpw
target weights, a DFlash2 k=7 draft, FP8 KV cache, prefix caching, a
1,000,000-token configured context, and a selective INT8 (W8A16 Marlin)
overlay for the model trunk that frees 1,595,392,320 bytes per rank.

The GitHub tree and release assets contain the recipe, documentation, evidence,
and checksums—not checkpoint weight objects. The separate public Hugging Face
repository carries the attributed, byte-identical target mirror. Its exact
123-file LFS payload and completion receipt were remotely verified before
maintainer merge into immutable public main revision
`e7c34dba923916754cfcb0bdf6c2c75a9b7ff1fc`. Every serving input is pinned by
revision, digest, or hash and verified before the fleet starts.

## How it compares with what was already public

The reference rows below are **author-reported**: measured by each recipe's own
author, on that author's hardware, with that author's harness. The JSpark3 v1
row is our own local measurement, and the Basis column states the conditions of
every row. **No delta or ranking is computed between any two rows**; node
counts, quantization lanes, speculation, context, clocking, and estimators
differ.

| Recipe | Nodes | Lane | Context | Decode (tok/s) | Basis |
|---|---:|---|---:|---|---|
| **JSpark3 v1** `v1.0.0` | 3 | EXL3/TR3 4 bpw, DFlash2, W8A16 trunk overlay | 1,000,000 | structured count 81.962; code 66.257; prose 29.049 | local; frozen 24-request screen, thinking off, temperature 0, 400 max tokens, warm server; medians of three batteries; per-stream estimator |
| FlyCockpit TP3 `9093765c` | 3 | EXL3/TR3 4 bpw, DFlash2 | 1,000,000 | structured count 69.0 / 68.5 / 71.2; code 52.3 / 58.7 / 58.2 | author-reported |
| [neko-legends TP4](https://github.com/neko-legends/spark-bench/blob/a1d8daffad44ad69d8f9e27e621b5f4afc4157fe/README.md#L81-L188) | 4 | EXL3/TR3 4 bpw, DFlash2 | 1,000,000 | code 64.5; structured 100.9; math 77.8; prose 23.1; C4 aggregate 253 | author-reported; four DGX Sparks, warm client-wall `bench_exl3.py`, thinking off |
| Mia TP2 `c190db1a` | 2 | EXL3/TR3 4 bpw, DFlash2 | 1,000,000 | sparkDash C1 62.9; lab structured 65.1; lab prose 27.1 | author-reported |
| jetnet TP3 `bfc820ec` | 3 | NVFP4 with Marlin W4A16, MTP-4 | 512K | 35.2; DFlash2 lane, thinking on, 47.2 | author-reported |

Three of those recipes were run on this fleet as adapted local reproductions,
never as exact ones: `mia-tp2-historical-0e2e78f` (site/safety-adapted) at
24.913 tok/s, `mia-tp2-current-c190db1a-adapted` (compatibility-adapted) at
24.728 tok/s, and `fly-derived-9093765c-adapted`
(minimal-correctness/safety-adapted) at 29.042 tok/s, each on a same-prompt
agent trajectory that is product evidence rather than an engine-rate
comparison. No literal FlyCockpit run and no jetnet run exists here.

## What the overlay changed, internally

Measured against the matched three-Spark control (same recipe, overlay
disabled), an unreleased internal development build, not a market comparison:

- Single-stream decode medians on the frozen 24-request plan: code 63.861 to
  66.257 tok/s (+3.75%), structured count 77.510 to 81.962 (+5.74%), prose
  28.308 to 29.049 (+2.62%). Strict same-day pairing: +7.27%, +6.63%, +8.35%.
- Token pacing (paired): median interval 98.645 to 91.912 ms (-6.83%), p99
  120.472 to 108.105 ms (-10.27%), worst interval 364.416 to 148.344 ms
  (-59.29%).
- Concurrency (aggregate service throughput): C12 +0.16%, C24 +1.21%, C48
  229.966 to 237.946 tok/s (+3.47%).
- Long prefill (113,908 tokens): 1277.443 to 1234.246 tok/s (-3.38%), a
  measured regression.
- Three-stream waves were variable and lost the strict pairing (-21.20%).
- Two internal promotion gates were missed and are disclosed: the campaign
  code median (66.257 tok/s) fell 0.743 tok/s (1.11%) short of a 67.0 floor,
  and the agent demonstration's longest uncompensated slow run was 14 against
  a limit below 5. No correctness, stability, or safety failure occurred in
  any run.

Conditions, estimators, minimum fields, and receipts: `docs/BENCHMARKS.md` and
`results/results.json`.

## Contents

- `recipe/`: lifecycle controller, per-rank preflight, fail-closed entrypoint,
  five transactional hash-gated runtime transforms, the W8A16 overlay,
  checkpoint validation and runtime-view preparation, `SHA256SUMS`.
- `docs/`: architecture (with diagram), technical report, benchmarks,
  installation, operations, limitations, reproducibility, licensing.
- `results/`: machine-readable results and sanitized evidence.
- `manifests/`: pinned dependencies, release metadata, derivation record,
  CycloneDX SBOM.
- `huggingface/`: the exact metadata, attribution, provenance, target-mirror
  contract, and completion receipt for the remotely verified public mirror.
- `docker/`, `.github/workflows/`: local-only image reproducibility definition
  and CI validation. No JSpark3 GHCR image is published for v1.0.0; the recipe
  uses the exact upstream image by digest.

## Release assets

- `jspark3-recipe-1.0.0.tar.gz`: the self-contained recipe directory.
- `jspark3-results-1.0.0.tar.gz`: results and evidence.
- `jspark3-1.0.0.sbom.cdx.json`: SBOM of the pinned inputs.
- `SHA256SUMS`: checksums of the assets above; verify before use.

Assets are built by `tools/build_release_assets.sh` with reproducible
archives and attested by the release workflow.

## Known limitations

See `docs/LIMITATIONS.md`. In brief: exactly three DGX Sparks; every input
pinned; prefill slower than the matched control; C3 variability; 96.722 s p90 time to
first token at 48 streams; single-fleet evidence; no public accuracy
benchmark; no endpoint authentication.

## Licenses

Original code and prose: Apache-2.0. Target checkpoint: ShapleyMcg License
v1.0, attribution-required. DFlash2 draft: CC BY-NC-ND 4.0, research and
evaluation use, commercial use requires permission from Inco AI. The
assembled endpoint is neither unrestricted open source nor commercial-ready.
See `docs/LICENSING.md`, `THIRD_PARTY_NOTICES.md`, and
`REQUIRED_ATTRIBUTION.md`.
