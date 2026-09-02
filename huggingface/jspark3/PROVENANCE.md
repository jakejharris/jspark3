# Provenance of the mirrored target weights

This repository re-hosts a checkpoint that JSpark3 did not create. This file
records where every byte came from, how every hash in
[`WEIGHTS-MANIFEST.json`](WEIGHTS-MANIFEST.json) was obtained, one discrepancy
found while checking them, and the boundary between the mirrored work and
JSpark3's own.

## The chain

| Link | Who | What |
|---|---|---|
| Base model | Z.AI | `zai-org/GLM-5.3-Flash`, the model these weights quantize |
| Quantization | Brandon M. Music | The EXL3/TR3 4-bpw checkpoint, produced with ShapleyMcg, published as `brandonmusic/GLM-5.3-Flash-tr3-4bpw` at revision `5ab363a8dcf6405955fd5f99671e01a1c9fb124b` |
| Re-host | Mia's AI Lab | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` at revision `25a44fdbf16862a46b7cc9921142c6c81350af2f`, which its own card describes as a byte-identical redistribution of the revision above |
| Mirror | JSpark3 | This repository, an exact hash-verifiable mirror of the Mia revision above |

JSpark3 did not train the base model, did not perform the quantization, did not
modify a single weight byte, and claims no authorship of any of it. What JSpark3
contributed is the three-DGX-Spark TP3 integration, the selective W8A16 Marlin
overlay, the runtime recipe, the tooling, the validation, and the evidence — all
of which live in the GitHub repository, not in these weights. The DFlash2
speculative draft is a separately pinned dependency under its own license and is
deliberately not mirrored here.

## What is in this repository and what is not

The mirror describes 144 files totalling 175,715,854,754 bytes at the pinned
revision. 123 of those are Git LFS objects: the 120 `.safetensors` shards plus
`model.safetensors.index.json`, `quantization_config.json`, and `tokenizer.json`.
**No LFS object is committed in the JSpark3 source tree.** The 21 remaining
small files are carried here verbatim at their upstream paths, with one
disclosed exception: the upstream `README.md` is carried as
[`UPSTREAM_MODEL_CARD.md`](../UPSTREAM_MODEL_CARD.md) so that this repository can
hold its own card. Every upstream path is otherwise preserved so that a
checkpoint contract validating a download from this mirror validates exactly as
it does against the upstream repository.

## How every hash was obtained

- **LFS files (123).** The SHA-256 in the manifest is the Hub's LFS object id
  for that file at the pinned revision, read from the repository tree listing.
  Each one was then compared against that file's entry in the upstream
  `SHA256SUMS`. All 123 agree. `hash_source` is
  `hub-lfs-oid+upstream-sha256sums`.
- **Small files (21).** The SHA-256 in the manifest was computed locally over
  the bytes actually fetched from the Hub at the pinned revision, which are the
  same bytes shipped in this repository. `hash_source` is
  `local-sha256-of-fetched-bytes`.
- Sizes come from the same tree listing and were confirmed against the fetched
  bytes for every file present here.

Nothing in the manifest is estimated, inferred, or reconstructed. No digest was
invented, and no hash was copied from a card or a README rather than measured.

## A discrepancy in the upstream `SHA256SUMS`, recorded as found

The upstream `SHA256SUMS` file carries 328 entries. It originates with the
quantization author's repository and therefore lists files that the re-host does
not carry. Of the 21 small files present at the pinned revision:

- 12 appear in `SHA256SUMS`. 10 of those match the fetched bytes exactly.
- **`LICENSE` and `README.md` do not match their `SHA256SUMS` entries.** For
  `LICENSE` the listed digest is `30b85b6b9659f2e7…` while the fetched bytes
  hash to `9a354667162e40201fa556e29ae7a327cdb112eacaa8ef100106e6063635e28a`.
  For `README.md` the listed digest is `a24ed04666047362…` while the fetched
  bytes hash to `ad818d1fb7c02d6d…`.
- 9 are additions made by the re-host (`.gitattributes`, `MANIFEST.json`,
  `MIRROR.json`, `ORIGINAL_MODEL_CARD.md`, `PROVENANCE.md`, `SHA256SUMS`
  itself, `THIRD_PARTY_LICENSES/B12X-APACHE-2.0.txt`,
  `THIRD_PARTY_NOTICES.md`, and `runtime-results-v44.json`) and have no entry.

The `README.md` difference has an obvious explanation on the face of the
repository: the re-host states that it replaced the card with its own. The
`LICENSE` difference does not: the re-host's card says "The `LICENSE` file is
the upstream file, unmodified", and the fetched `LICENSE` bytes nevertheless do
not hash to the digest listed for `LICENSE` in the same repository's
`SHA256SUMS`.

**JSpark3 does not resolve this by assertion.** Both facts are recorded above
exactly as measured. This mirror carries the `LICENSE` bytes as fetched at the
pinned revision, and the manifest records the hash of those bytes, not the
listed one. Any reader who needs the point settled should raise it with the
upstream maintainers; it is a question about their repository, not about this
one. No weight shard is affected: all 123 LFS digests agree with the same
`SHA256SUMS` file.

## Verifying this mirror yourself

`tools/mirror_weights.py` in the GitHub repository fetches the pinned revision
and verifies every entry in `WEIGHTS-MANIFEST.json` by size and SHA-256,
refusing on any drift, missing file, or extra file. It performs no network call
during release validation, and its upload path is disabled by default.

## Licensing of the mirrored bytes

The checkpoint is licensed under the ShapleyMcg License v1.0, reproduced
verbatim as [`LICENSE`](../LICENSE) at the root of this repository. It is a
source-available, attribution-required license; it is not an OSI-approved open
source license, and the license text itself says so. Downstream copies of the
Work stay under it. The required attribution notice appears in this repository's
card, verbatim, as the license requires. JSpark3's own recipe, tooling, and
documentation are Apache-2.0 and are covered by
[`RECIPE-LICENSE`](RECIPE-LICENSE); that license covers none of the weights.

The maintainer authorized the attributed mirror. Its verified transfer has not
started, so the public repository contains no model bytes yet.
