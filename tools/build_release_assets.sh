#!/usr/bin/env bash
# Build the release assets for the current tag into a directory (default dist/).
#
#   jspark3-recipe-<version>.tar.gz       the self-contained recipe/ tree
#   jspark3-results-<version>.tar.gz      results.json, SUMMARY.md, sanitized evidence
#   jspark3-<version>.sbom.cdx.json       CycloneDX SBOM of the pinned inputs
#   SHA256SUMS                             checksums of the files above
#
# Archives are reproducible: sorted entries, fixed mtime, numeric root owner,
# gzip without a timestamp. Run tools/validate_release.py first.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
out=${1:-$here/dist}
version=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tag"].lstrip("v"))' "$here/manifests/release.json")
mkdir -p "$out"
archive() {
  local name=$1; shift
  tar --sort=name --mtime='2026-01-01 00:00:00Z' --owner=0 --group=0 --numeric-owner \
      --exclude='__pycache__' --exclude='*.pyc' -C "$here" -cf - "$@" | gzip -n -9 > "$out/$name"
}
python3 "$here/tools/build_sbom.py" --dependencies "$here/manifests/dependencies.json" \
  --release "$here/manifests/release.json" --output "$here/manifests/sbom.cdx.json" >/dev/null
archive "jspark3-recipe-$version.tar.gz" recipe
archive "jspark3-results-$version.tar.gz" results
cp "$here/manifests/sbom.cdx.json" "$out/jspark3-$version.sbom.cdx.json"
(cd "$out" && sha256sum "jspark3-recipe-$version.tar.gz" "jspark3-results-$version.tar.gz" "jspark3-$version.sbom.cdx.json" > SHA256SUMS)
echo "release assets in $out:"
cat "$out/SHA256SUMS"
