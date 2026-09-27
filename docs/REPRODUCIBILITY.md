# Reproducing the export

The maintainer's offline exporter consumes a sealed source snapshot and its
file inventory. It verifies input hashes, selects the recipe and an explicit
gate/build-tool allowlist, applies a reviewed rename map, replays affected
transforms against pinned local image sources, and updates dependent hashes. It composes the sealed stock production policy,
applies the approved per-file license audit, and excludes native binaries.
Unknown payloads and privacy findings stop packaging.

The [derivation manifest](../manifests/derivation.json) records input and
output hashes for every payload file, the builder and policy hashes, and the
source-inventory identity. The [composition report](../manifests/composition.json)
records target hashes and comparison of executable Python syntax trees after
comment and docstring sanitization. These are offline checks, not hardware
qualification. Generated manifests and replacement documentation identify
their origin separately from sealed-source files.

The source snapshot and private rename map are maintainer inputs. This export
does not claim that a reader lacking those inputs can rederive the original
private snapshot. Readers can independently verify the payload, source
contracts, binary-exclusion inventory, file and dependency SBOM and lifecycle command rendering:

```sh
sha256sum -c SHA256SUMS
python3 -B tools/validate_release.py . --report ../validation.json
```

Archives have sorted paths, fixed zero timestamps, numeric root ownership,
normalized modes and timestamp-free gzip. Root `SHA256SUMS` identifies the
complete public tree. Distribution `SHA256SUMS` identifies the archive and
SBOM. Reports live beside the export so validation never changes hashed bytes.

The validator uses shared inventory helpers, syntax checks, link checks and
lifecycle dry-runs from the latest local release-validator lineage, whose
hash is recorded in the derivation manifest. This recipe-only profile has
seventeen checks. Historic weight-mirror, model-card and result-table checks
belong to the separate final publication validation and are not claimed here.

## Final source binding

A source checkpoint may leave the final recipe unselected. A prepared variant is a
rehearsal, not a selected release. Validate an eventual final with:

```sh
python3 -B tools/validate_release.py . --require-final --report ../final-validation.json
```

Final validation refuses an unbound recipe, a mismatched delivery/source/profile,
a mixed recipe inventory, or example/toggle values that differ from the bound
startup settings. The binding records runtime epoch choices separately from
resident startup switches. A seal and a passing offline validator do not confer
hardware admission: the exact shipped configuration still needs fresh hardware
qualification. Historical validation-profile results remain a separate measured cohort.
