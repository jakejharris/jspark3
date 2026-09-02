# Release state: JSpark3 v1.0.0

Status: **v1.0.0 released 2026-09-02; Hugging Face metadata live; weight
mirror in progress on a separate review branch and not merged.**

The public GitHub release is
<https://github.com/jakejharris/jspark3/releases/tag/v1.0.0>. The Hugging Face
main revision remains the exact 29-file metadata allowlist at
`e9cbbafaf9ae4ab64f385c2f68e7fe2f06d78676`; the 175,715,854,754-byte target
mirror is not part of main until its review branch passes remote verification
and a maintainer merges it. No JSpark3 GHCR image is published for v1.0.0; the
recipe uses the exact upstream image by digest.

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
| Hugging Face | Metadata is live. The attributed, byte-identical target-weight transfer is authorized and in progress on a separate review branch; it is not merged into main. No DFlash2 byte is mirrored. |
| Release assets | Reproducibly built into `dist/` by `tools/build_release_assets.sh`: recipe archive, results archive, CycloneDX SBOM, and checksums. |

`python3 tools/validate_release.py . --report validation.json` must return
`VERDICT PASS` with all 16 checks after any edit.

## Remaining independent Hugging Face work

The GitHub release is complete; the weight mirror has its own completion
boundary. The active transfer targets an existing Hugging Face pull-request
revision, never `main`.

1. Finish the resumable transfer in its durable mirror directory and existing
   review branch. Do not start a second fetch or uploader.
2. Run `tools/mirror_weights.py remote-verify` against that pull-request
   revision. A successful uploader exit is not proof.
3. Add `jspark3/MIRROR-COMPLETION.json` only after remote verification, then
   verify the receipt bytes on the same review revision.
4. A maintainer may merge only after the exact 123-file LFS allowlist, sizes,
   hashes, unchanged metadata parent, and DFlash2 exclusion all pass. Merge is
   deliberately not automated.
5. After merge, record the immutable Hub main revision in a later metadata
   amendment. The tagged v1.0.0 state remains honest about the weights being
   unmerged on release day.

The ShapleyMcg License v1.0, its full attribution, upstream notices, and pinned
provenance travel with the mirror unchanged. Operators should keep using the
pinned upstream checkpoint until the JSpark3 Hub main revision contains the
verified objects.

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
