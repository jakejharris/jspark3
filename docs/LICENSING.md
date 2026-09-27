# Licensing

Root `LICENSE` covers original JSpark3 code and prose under Apache-2.0.
Upstream code keeps its per-file license. SPDX headers identify source files;
`REUSE.toml` records metadata and pinned cooperative-MoE files without inserting
headers into their source. Vendored ExLlamaV3 headers retain their MIT notices
and original bytes. The replay-window patch remains AGPL-3.0-only.

The assembled serving process includes AGPL-3.0-only components and is not
represented as wholly Apache-2.0. If you modify an AGPL component and let users
interact with the modified service over a network, AGPL section 13 requires
offering those users its Corresponding Source.

[Third-party notices](../THIRD_PARTY_NOTICES.md) credit the upstream revisions
and contributors. License texts are retained under `third_party/licenses/`
and beside the components. [Dependencies](../manifests/dependencies.json),
[per-file terms](../manifests/license-review.json) and the
[SBOM](../manifests/sbom.cdx.json) record the source package's posture.

No native binaries, weights, tensor exports or container images are included.
Obtain those separately under their own terms. DFlash2 is restricted to
non-commercial research/evaluation absent separate permission. Preserve the
[required ShapleyMcg attribution](../REQUIRED_ATTRIBUTION.md). InstantTensor's
native module has additional linked-library obligations; neither that module
nor the serving image is redistributed here.
