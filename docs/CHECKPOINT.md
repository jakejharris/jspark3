# Checkpoint source and accepted copies

Fresh installs default to Brandon M. Music's EXL3/TR3 checkpoint at
[`brandonmusic/GLM-5.3-Flash-tr3-4bpw@5ab363a8dcf6405955fd5f99671e01a1c9fb124b`](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/tree/5ab363a8dcf6405955fd5f99671e01a1c9fb124b).
This is an unreleased change after v1.8.4. It changes the download source,
not the serving tensors or benchmark results.

Two pinned re-hosts remain accepted without a new download:

- [Mia-AiLab at `25a44fdbf16862a46b7cc9921142c6c81350af2f`](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw/tree/25a44fdbf16862a46b7cc9921142c6c81350af2f).
- [JSpark3 at `e7c34dba923916754cfcb0bdf6c2c75a9b7ff1fc`](https://huggingface.co/jakejharris/jspark3/tree/e7c34dba923916754cfcb0bdf6c2c75a9b7ff1fc).

The [checkpoint contract](../recipe/config/checkpoint-contract.json) names the
default, alternates and compatibility directory. [Installation](INSTALL.md)
uses that same directory for every source. Its historical Mia-AiLab name is a
local storage convention used by preflight, runtime views and container mounts;
it does not select a download repository. Reuse an existing pinned snapshot and
its sibling view. Never overlay a different snapshot onto a populated directory.

## Verification scope

The three pinned repositories publish the same 123 LFS SHA-256 hashes and sizes:
120 safetensors shards (175,642,157,752 bytes), the model index, tokenizer and
quantization config. The shared Git files have matching blob IDs except README.
The source files named by the ledger were fetched and checked by SHA-256; LFS
objects were compared through published metadata, not downloaded and rehashed
for this source change. The runtime validator still hashes every serving shard.

All three publish the same 328-entry `SHA256SUMS`, SHA-256
`cb0da1f97a53aebc3fbc5478f19c82b25586b0bf8533c99fb4ed5321a48f5342`.
Brandon's snapshot includes every listed file. Both mirrors omit exactly 120
`.materialization/shards/` receipts and 72 `runtime/` files. The validator requires
the complete source layout or that exact mirror omission set, selected by the
pinned README hash. Partial omission sets refuse. Every present ledger entry is
hashed, and the shard inventory, index, byte totals and runtime links stay checked.

The ledger predates the snapshots' README and LICENSE updates. Those two hashes
are stale in all three copies. Validation requires these exact replacement hashes:

| File | Snapshot | SHA-256 |
|---|---|---|
| LICENSE | All three | `9a354667162e40201fa556e29ae7a327cdb112eacaa8ef100106e6063635e28a` |
| README.md | Brandon | `f7adf502e7ac90e752a5bd0dfbced0aea619c8ce9b66e612658a285f092a2d88` |
| README.md | Mia-AiLab | `ad818d1fb7c02d6d10c1cd9fa0a378f6e636879bded387d3d4d90037a726235f` |
| README.md | JSpark3 | `ca1aa1885f861edd621c7d10f685fc88a0464854ac3859f650303f263c9d7a58` |

`publication_ledger_complete` stays false: a serving pass is not a claim that the
upstream publication ledger is current or that all repository files are listed.
Additional documentation outside the ledger is not authenticated by this gate.

## License and component qualification

The selected source pin carries ShapleyMcg License v1.0, byte-identical to both
accepted mirrors. Its LICENSE hash matches [REQUIRED_ATTRIBUTION.md](../REQUIRED_ATTRIBUTION.md).
Revision `a5fee929cf4888b1824323e33e8a19b60129e025` carries the Local Inference Lab
Attribution License v1.0 instead. Do not substitute that revision or a branch name.
The pinned attribution text and Schedule B citation remain unchanged.

The cooperative-MoE component campaign retains its original
[checkpoint subset pin](../recipe/config/coop-checkpoint.json) and Mia snapshot
cache mount. That pin is now checked against the accepted alternates. The
campaign's environment reads its own subset pin instead of combining a new
source revision with a hardcoded Mia cache path. This preserves historical
qualification evidence and does not claim a new component campaign.

Offline tests cover both layouts, all three cards, corruption, missing or extra
shards, partial omissions, changed licenses and broken runtime links. A complete
local rehash, runtime preparation, preflight and serving admission on the target
hardware remain release follow-ups. No new hardware or performance result is
claimed here.
