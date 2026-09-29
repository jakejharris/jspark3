# Licensing

Root `LICENSE` covers original JSpark3 code and prose under Apache-2.0.
Upstream code keeps its per-file license. SPDX headers identify source files;
`REUSE.toml` records metadata and pinned cooperative-MoE files without inserting
headers into their source. Vendored ExLlamaV3 headers retain their MIT notices
and original bytes. Fifty-two files retain AGPL-3.0-only terms, including scheduler, prefix-cache,
display-KV and cooperative-MoE code; see the per-file manifest.

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
Obtain those separately under their own terms. The controller always uses the DFlash2 draft model. Its pinned
[model card](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/blob/dc77ff1c99eeb2df044ee3d4f0094eb033fee410/README.md)
licenses it under CC BY-NC-ND 4.0 for non-commercial research/evaluation;
commercial licensing requires separate permission from Inco AI. The recipe's
Apache-2.0 license does not remove that required dependency's restrictions.
The target checkpoint, Brandon M. Music's EXL3/TR3 quant fetched from Mia-AiLab's
byte-identical re-host, retains the [ShapleyMcg License v1.0](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/blob/5ab363a8dcf6405955fd5f99671e01a1c9fb124b/LICENSE),
including attribution and its named exclusion. Preserve the
[required ShapleyMcg attribution](../REQUIRED_ATTRIBUTION.md). InstantTensor's
native module has additional linked-library obligations; neither that module
nor the serving image is redistributed here.
