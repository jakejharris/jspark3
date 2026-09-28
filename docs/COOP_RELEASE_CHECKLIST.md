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

The mechanical fill-in command implements steps 1–4 below. Run it from a clean
validated clone or archive of this source (without private task files), after
reviewing the actual campaign. `SEALED_OUTPUT` is the original output of
`qualify_coop.py --seal`; `SERVING_SOURCE` must be a new directory outside both
the source and seal:

```sh
python3 -B tools/integrate_coop_seal.py \
  --sealed-output "$SEALED_OUTPUT" --output "$SERVING_SOURCE"
python3 -B "$SERVING_SOURCE/tools/validate_release.py" "$SERVING_SOURCE"
```

The tool verifies the complete seal and native bytes against this source, copies
only the listed text assets, and validates the new tree before exposing it.
The original seal remains unchanged. The original source manifest is retained
as `QUALIFICATION_SOURCE_MANIFEST.json`; BUILD keeps its qualification-source
hash and exact compiled-input map while its final source-manifest hash changes.
The canonical rebound BUILD digest becomes the integration seal identity.

Review and commit the generated source as the integration commit before the
final-source builds below. The binding and release stage become
`component-qualified`; the new results stay `pending-measurement`, and admission
and results digests stay null. Source `hardware_qualified` remains false.
This state allows private serving and measurement. `--require-final` refuses it
regardless of stage labels. Publication staging still runs that final validator.
The publication preparation handoff therefore has an intermediate delivery step:
prepared → component-qualified → bound/final. Its remaining boot, result, copy,
inventory and publication requirements still apply.

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

Build the operator image for the integration source, then use `build_native.py`
for two fresh native builds on the first physical machine and an independent
build on the second. Every cooperative result must retain the target digest.
Image and native receipts from the pre-integration source cannot describe these
new source bytes. With the first machine's final-source receipts and build tree,
prepare the single coop=1 runtime shared by the matched A/B arms:

```sh
python3 -B "$SERVING_SOURCE/tools/prepare_runtime.py" \
  --binary-root "$FINAL_BUILD_ROOT" \
  --image-receipt "$FINAL_IMAGE_RECEIPT" \
  --native-receipt "$FINAL_BUILD_ROOT/native-build-receipt.json" \
  --output "$PRIVATE_RUNTIME"
```

The runtime receipt carries the verified schema-2 component identity and
`hardware_qualified=false`. Both arms use this same prepared recipe; choose
coop on/off in their separate boot environments and restart between arms.
Do not use preparation's diagnostic `--coop-off` option to construct the ON arm.
The stock release measurement and admission boot remain required before final
binding, independently of the matched comparison.

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

The prepared [documentation patch](../tools/v184_post_seal_docs.patch) supplies
the release-time copy for README, INSTALL, OPERATIONS, CHANGELOG, benchmark and
limitation guidance, reproducibility and maintainer handoff docs, release notes
and the measurement guide. `integrate_coop_seal.py` does not apply it or edit
public prose. Apply it after the reviewed seal and required serving evidence
are integrated, before the final documentation render and inventory refresh:

```sh
git -C "$FINAL_TREE" apply --check "$FINAL_TREE/tools/v184_post_seal_docs.patch"
git -C "$FINAL_TREE" apply "$FINAL_TREE/tools/v184_post_seal_docs.patch"
```

The patch leaves the seal, native, policy, campaign link and matched A/B slots
in README and release notes for the publication renderer. Fill them from the
actual integrated records, then copy the rendered README and release notes
into the source. The HF card uses the same fill-in values. No placeholder may
reach the tag. If the patch is already applied, review its complete file list
instead of forcing it over later documentation changes.

This changes only documentation. Preserve every recipe byte, including the
sealed toolchain metadata and `recipe/SHA256SUMS`, and preserve historical result
records. Refresh documentation hashes, root derivation and root `SHA256SUMS`
after applying and rendering; source validation must pass again. Review all
current installation claims for the qualified coop-on default and both explicit
diagnostic opt-outs: `prepare_runtime.py --coop-off` and
`build_native.py --display-only`. Historical coop-off measurements and conditional
refusals for invalid inputs remain correctly scoped.

Replace pending language in the README, release notes, changelog,
`release/MEASUREMENTS-v1.8.4.md`, installation, operations, limitations and
reproducibility documents. Cite the real seal, pinned native bytes, completed
memcheck/racecheck campaign and matched comparison. Credit Mia's AI Lab
(MiaAI-Lab). Describe shared cudart linkage alongside the existing AGPL/MIT
component boundaries and model restrictions; do not imply that linkage changes
the licenses or that the complete service is Apache-2.0.

Group upstream authors in one `Credits` section near the end of the release
notes, README and HF card, naming each once in the credit prose and linking
`THIRD_PARTY_NOTICES.md`. Keep the exact ShapleyMcg notice there, its canonical
link, the HF `shapleymcg` metadata tag and Schedule B BibTeX citation. Preserve
the citation in [`TECHNICAL-REPORT.md`](TECHNICAL-REPORT.md) and every source
license notice, including notices in AGPL components. Jake treats these
additions as goodwill: the publication kit warns during preparation and dry-run
planning but does not block staging, approval or publication over attribution.
The optional HF tag edit parses and serializes YAML, including flow lists with
a trailing comma. If it cannot safely add the tag, it preserves the captured
frontmatter and warns; tag insertion must not abort staging or publication.
Keep `REQUIRED_ATTRIBUTION.md` and its v1.0 pin unchanged unless Jake decides
to adopt different terms.

Unfilled publication placeholders block release. The kit checks every defined
fill-in slot and other bracketed tokens, including digits, lowercase, spaces
and line breaks, after substitution and again in staged publication payloads.
Valid Markdown links and structured data arrays remain supported. The check
includes free-form A/B text, source and landing READMEs, the HF card, release
notes, titles and published JSON/measurement assets. Do not approve or publish
a package containing unresolved placeholders, even if its hashes were refreshed.
Keep the ShapleyMcg notice verbatim in the rendered release notes and standalone
GitHub release body; the kit's regression checks compare them to the source
notice in `REQUIRED_ATTRIBUTION.md`.

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
