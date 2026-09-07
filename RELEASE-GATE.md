# Release state: JSpark3

Status: **v1.1.0 (Cadence) staged for publication — not tagged, not released.
v1.0.0 was released 2026-09-02; its record below is historical.**

## Current release state: v1.1.0 (Cadence)

| Item | State |
|---|---|
| Identity | JSpark3 v1.1, release name Cadence, machine/tag `v1.1.0`. The tag, GitHub release, and Hub push are serialized publication actions that have not happened; install references name the tag and fail until it exists. |
| Content | The v1.0.1 construction (kernel disable in the transform; fabric documentation) plus the two measured Cadence features: the QKV decode shadow and the speculative width controller, with the stock indexer workspace, original drafter, and sampler retained. |
| Measured evidence | Paired single-stream effects with scoped claims: prose +16.18% / +19.17% and structured count +7.66% / +7.61% across two independent serving starts — first-start figures are diagnostic only (their sham control failed the predeclared resolution margin), the second start's predeclared sham passes; the paired code gain is not replicated; candidate-only quality losses are preserved; a short-prompt concurrency burst is explicitly not a capacity certification. Machine-readable source of record: [results/evidence/candidate/cadence-v11/CLAIMS.json](results/evidence/candidate/cadence-v11/CLAIMS.json). See [docs/BENCHMARKS.md](docs/BENCHMARKS.md#v11-cadence-evidence). |
| GHCR | No JSpark3 GHCR image is published for v1.1.0; the v1.0.0 license audit's NO-GO carries forward and the recipe pins the exact upstream image by digest. |
| Reconstruction | A fresh three-rank build from the recipe contracts completed construction, route audits, and the frozen batteries without a serving repair, and the exact prior arm was restored and re-verified afterward — evidence that the recipe reconstructs the construction, not a new performance claim. |
| Kernel fix | Carried in the transform contract. An independent offline construction test reproduced the pinned sparse-attention kernel transform's expected output byte-identically (`00e32052b781723500987a463814116634c4d00b3b61066915f8cc70780c931e`) and confirmed the already-applied check. |
| Pending verification | (1) A live single-stream witness above 32,768 tokens on the assembled public build has not run; the transform-level kernel-file change is verified, long-context operation is not. (2) Verifier defect F-1 is fixed in this tree: `verify` now requires the pinned Cadence path's full-bank capture evidence per rank (at least 16 hash-matching capture receipts and 8 non-empty serving graph dumps — the boot41/43 measured-arm startup receipts) instead of the stock capture progress bar the pinned path never renders; the fix still awaits live confirmation on hardware. (3) Integrated regeneration of manifests, checksums, and release assets across recipe and docs is complete in this tree, redone after the F-1 fix integrated. (4) Reconciliation of the v1.1 measured numbers into the machine-readable results and the validator's claim map is done via the current-claims evidence export; the frozen historical results are untouched. |
| Provenance correction | The v1.0.0 record below describes the recipe as derived by "identifier renames only"; that was inaccurate for v1.0.0, whose transform contract missed the start-time kernel disable. The historical record is preserved unmodified; [docs/LIMITATIONS.md](docs/LIMITATIONS.md#kernel-disable-provenance-and-the-32768-token-single-stream-boundary-v100) carries the correction. |

## Historical record: v1.0.0

The public GitHub release is
<https://github.com/jakejharris/jspark3/releases/tag/v1.0.0>. The public,
ungated, enabled Hugging Face repository is
<https://huggingface.co/jakejharris/jspark3>. Its attributed target mirror and
exact completion receipt were remotely verified before maintainer merge into
the verified weights-mirror revision
`e7c34dba923916754cfcb0bdf6c2c75a9b7ff1fc`. No JSpark3 GHCR image is
published for v1.0.0; the recipe uses the exact upstream image by digest.

## Final GitHub release state

| Item | State |
|---|---|
| Public release | `v1.0.0`, released 2026-09-02 at the deterministic URL above. The source and `dist/` assets describe this same terminal state. |
| Public tree | One product, `JSpark3 v1`, slug `jspark3`. No internal experiment labels, private paths, hosts, addresses, container identities, captures, or authorization material on the public surface; the validator scans every file, including binaries. |
| Recipe | Derived from the measured recipe by identifier renames only; overlay, loader-hook, and patched-loader hashes re-pinned and proven from the pristine loader. All Python parses, all shell passes `bash -n`, and 13 lifecycle and wrapper dry-runs render without host contact. |
| Evidence | Sanitized machine-readable evidence under `results/`, with campaign rejection and demonstration pacing failure preserved. Every public numeric claim reconciles against the results display map. |
| Comparison taxonomy | Published reference recipes remain author-reported with no cross-recipe deltas. Adapted local reproductions carry fidelity qualifiers. The matched overlay control remains an unreleased internal development build and is never presented as a competitor. |
| Licensing | Apache-2.0 covers original work only. ShapleyMcg attribution and DFlash2's non-commercial restriction remain explicit. The endpoint is neither unrestricted open source nor commercial-ready. |
| Container | The redistribution audit is a binding NO-GO. v1.0.0 publishes no JSpark3 GHCR image, runs the exact upstream image digest, and retains `docker/` for local reproducibility only. |
| Hugging Face | The public, ungated, enabled repository carries the remotely verified, attributed, byte-identical target mirror at `e7c34dba923916754cfcb0bdf6c2c75a9b7ff1fc`, plus the exact completion receipt. No DFlash2 byte is mirrored. |
| Release assets | Reproducibly built into `dist/` by `tools/build_release_assets.sh`: recipe archive, results archive, CycloneDX SBOM, and checksums. |

`python3 tools/validate_release.py . --report validation.json` must return
`VERDICT PASS` with all 16 checks after any edit to the historical tree. For
the v1.1 state, the same 16 checks attest hash-cascade, privacy, and claim
integrity of the tree; a PASS on this branch does not attest that the v1.1.0
release exists, and the new v1.1 numbers do not reconcile until the
integration step extends the machine-readable results.

## Completed independent Hugging Face mirror

The independent mirror boundary is complete. The resumable upload targeted an
existing pull-request revision, never `main`; `tools/mirror_weights.py
remote-verify` then proved the exact 123-file LFS allowlist, sizes, hashes,
unchanged metadata parent, and DFlash2 exclusion. The exact verified receipt at
`huggingface/jspark3/MIRROR-COMPLETION.json` was added and checked on the same
review revision before the maintainer merged it.

The ShapleyMcg License v1.0, its full attribution, upstream notices, and pinned
provenance travel with the mirror unchanged. Operators may fetch the public
JSpark3 mirror at terminal revision
`e7c34dba923916754cfcb0bdf6c2c75a9b7ff1fc` or the pinned upstream checkpoint;
the recipe validates the same byte contract either way.

## Closed container decision

Do not publish the prepared derivative. It retains NVIDIA-derived upstream
layers, and adding labels and notices does not satisfy the NGC derived-container
redistribution grant. `manifests/dependencies.json`
`owned_runtime_image.digest` and `manifests/release.json`
`live_links.ghcr_digest` stay null. Do not redistribute a local build without
independently satisfying NVIDIA and every applicable upstream term.

## Post-release evidence work

A fresh clean-room proof from the public recipe archive on three Sparks remains
useful, but it is not represented as completed here. Until an independent
reproduction exists, the evidence grade remains `ENGINEERING-EVIDENCE`.
Recommended follow-up also includes investigating C3 variability and a
prefill-neutral overlay variant. No Spark was contacted while assembling this
source amendment.
