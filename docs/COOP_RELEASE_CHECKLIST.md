# Completing the v1.8.4 source release

The prepared source already selects `JSPARK3_V16_COOP=1` in
[`recipe/.env.example`](../recipe/.env.example).
[`prepare_runtime.py`](../tools/prepare_runtime.py) refuses its default path
until the release-pinned component seal verifies. `--coop-off` remains an
explicit diagnostic option. Do not remove that refusal or turn a pending record
into a qualification claim by changing a flag.

The completed component campaign supplies evidence for the native kernel and
measured policy. Publication also needs final-source reproduction, archive/clone
parity, stock admission and the measured record required by
[`tools/_qualification.py`](../tools/_qualification.py). A matched coop on/off
comparison is a separate measurement after sealing. Historical result files
remain unchanged.

## Integrate the reviewed component seal

1. Validate the owner's completed seal with `qualify_coop.py --check-seal` and
   review the underlying campaign, sanitizer controls and independent build
   receipts. Copy the publishable `BUILD.json`, `QUALIFICATION.json`, selected
   bundle manifest and text/source/license assets into
   `recipe/overlays/v16/coop`. Exclude the native library, compiler output,
   saved sanitizer records, private diagnostics and model shards.
2. Copy the selected `dispatch_policy.json` into both `source/` and `bundle/`.
   Refresh its entry in `SOURCE_MANIFEST.json`, the `SOURCE_MANIFEST_SHA256`
   constant in `recipe/scripts/apply_coop_moe.py`, and the source-manifest pin in
   `INSTALL_CONTRACT.json`. Keep stage-8/footer pins unchanged when their bytes
   are unchanged.
3. Rebind BUILD's final `source_manifest_sha256`; preserve its original
   `qualification_source_manifest_sha256`, gate index and qualification build
   identities. The compiled-input map must remain exact. Use the canonical
   BUILD digest computed by `_coop_qualification.digest_value`, not a hash of
   arbitrarily reformatted JSON.
4. Set `recipe/config/coop-release.json` to `status: qualified` with the reviewed
   native, canonical BUILD and gate-index digests. Update the cooperative native
   entry in `manifests/binaries.json` to that exact native digest. Do not replace
   the compiled target in the verifier to accept a different result.
5. Rebuild the final source twice on the first physical machine and independently
   on the second. All cooperative libraries must match the qualification target.
   Retain the original pre-policy campaign and the final-source build witnesses
   as distinct evidence. The sealed native target is defined in
   [`_coop_qualification.py`](../recipe/scripts/_coop_qualification.py).

## Bind admission and measurement evidence

- Populate `release/results-v1.8.4.json` with the actual candidate commit,
  component seal, policy and measured stock cohort. Its required decode sweeps,
  prefill turns and quality evidence are defined by the existing final validator.
  Keep edited-mode observations separate. Do not copy historical values into
  the new record.
- Retain publishable same-boot admission inputs under
  `release/v1.8.4-admission/`, including `first-prompt.json` and `finalize.json`.
  Re-run the actual admission validator; a copied PASS label is insufficient.
- Fill the v1.8.4 `manifests/final-binding.json` component, policy, results and
  admission digests, set its state to `bound`, and update only the v1.8.4 entry
  in `manifests/final-catalog.json`. Preserve `final-binding-v1.8.3.json` and the
  historical catalog entries. Source `hardware_qualified` remains false; the
  actual receipts qualify their exact boot.
- Set `manifests/release.json` to the final stage while preserving its source-only
  distribution and publication-authorization semantics. Set each toggle's
  `release_ship_value` to the admitted prepared setting, with coop on and
  unedited production-stock defaults. No extra default flip is needed.
- Record the matched coop on/off A/B in a separate source-local evidence file.
  Keep weights, workload, concurrency, sampling, hardware and measurement
  definitions matched; use separate serving starts. Publish the actual result,
  including a neutral or negative result, with its conditions and evidence.

## Finish the public copy and source inventories

Replace pending language in the README, release notes, changelog,
`release/MEASUREMENTS-v1.8.4.md`, installation, operations, limitations and
reproducibility documents. Cite the real seal, pinned native bytes, completed
memcheck/racecheck campaign and matched comparison. Credit Mia's AI Lab
(MiaAI-Lab). Describe shared cudart linkage alongside the existing AGPL/MIT
component boundaries and model restrictions; do not imply that linkage changes
the licenses or that the complete service is Apache-2.0.

Credit ShapleyMcg near the top of the HF card, link its canonical repository,
include the Schedule B BibTeX citation and add the `shapleymcg` metadata tag.
Include the credit, link, notice and citation in the release notes, and retain
the citation in [`TECHNICAL-REPORT.md`](TECHNICAL-REPORT.md). Jake treats these
additions as goodwill: the publication kit warns during preparation and dry-run
planning but does not block staging, approval or publication over attribution.
Keep `REQUIRED_ATTRIBUTION.md` and its v1.0 pin unchanged unless Jake decides
to adopt different terms.

Refresh `recipe/SHA256SUMS`, the file SBOM, per-file license/REUSE inventory for
new evidence, executable-output audit for changed writers, derivation records,
the v1.8.4 documentation hashes and root `SHA256SUMS`. The delivery source-recipe
binding must describe the final bytes. Keep the historical results and their
original provenance records unchanged.

Validate an archive and an ordinary clone of the exact final commit with
`python3 -B tools/validate_release.py . --require-final`, checksum checks and
the component/admission tests. Check the prepared environment from both paths.
Package only that reviewed commit. Approval of publication applies to the exact
tag, archive, release notes, HF card and landing diff presented for review.
