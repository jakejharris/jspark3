# JSpark3 v1 recipe

This directory is the runnable part of JSpark3 v1: a fail-closed serving recipe
that brings up one GLM-5.3 Flash endpoint across three NVIDIA DGX Sparks with
tensor parallel 3 and expert parallel 3, an EXL3/TR3 4-bpw target, a DFlash2
k=7 draft, FP8 KV cache, prefix caching, a 1,000,000-token configured context,
and a selective W8A16 Marlin trunk overlay.

It contains no weights, no tokenizer, no image layers, and no patched vLLM
tree. Everything it launches is pinned by revision, digest, or hash and
verified before start.

## Layout

| Path | Purpose |
|---|---|
| `config/profile.json` | The serving envelope: image, runtime arguments, environment, overlay identity, hash gates. |
| `config/checkpoint-contract.json` | Target and draft checkpoint identity: revisions, config hashes, shard inventory, byte totals. |
| `config/patch-contract.json` | The five runtime transforms with full before/after hashes and seam cardinalities. |
| `scripts/fleetctl.py` | Lifecycle controller: `preflight`, `start`, `status`, `stop`, `verify`, each with `--dry-run`. |
| `scripts/*.sh` | Thin wrappers (`clean-room-setup`, `preflight`, `start`, `health`, `status`, `verify`, `stop`, `rollback`). |
| `scripts/container_entry.sh` | In-container entrypoint: cgroup and environment refusals, receipt binding, transforms, overlay, serve. |
| `scripts/apply_*.py`, `scripts/_atomic.py`, `scripts/_contracts.py` | Crash-recoverable transform programs and their transaction layer. |
| `overlays/trunk_w8a16.py`, `overlays/patch_base_loader_hook.py` | The W8A16 overlay module and the loader hook patcher. |
| `scripts/prepare_runtime_views.py`, `scripts/validate_checkpoint.py` | Build and validate the TP3 runtime views of the downloaded checkpoints. |
| `scripts/api_smoke.py`, `scripts/focused_witness.py` | Endpoint smoke test and the fixed health witness used by `verify`. |
| `transforms/README.md` | Human-readable transform inventory with every hash. |
| `SHA256SUMS` | Checksums of every recipe file; checked before any preflight. |

## Required inputs

- Target: `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` at revision
  `25a44fdbf16862a46b7cc9921142c6c81350af2f`.
- Draft: `incoai/GLM-5.3-Flash-DFlash2` at revision
  `dc77ff1c99eeb2df044ee3d4f0094eb033fee410`.
- Image: `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58`.
- FlyCockpit sources at commit `9093765c757bd1976372196e44af84a67cf86bad`,
  mounted read-only; the transforms verify the files they read.
- The overlay files must hash to the values pinned in `config/profile.json`
  and `scripts/container_entry.sh`; the loader transition is pinned from
  `a7e925f232ad3eebbee7ab37d3aba724c24465c3078da29489da0438664c6b08` to
  `3205bff77aac34785167f5b21306048b9dc916b2c0691bf774bb3d9202bbd8da`.

The installation walk-through, including checkpoint download and runtime view
preparation, is in the repository's `docs/INSTALL.md`.

## Lifecycle

```bash
cp .env.example .env                 # replace every placeholder
./scripts/clean-room-setup.sh --env-file .env --output preflight.json
preflight_sha=$(sha256sum preflight.json | cut -d' ' -f1)
./scripts/start.sh --env-file .env --preflight preflight.json \
  --preflight-sha256 "$preflight_sha" --confirm START-JSPARK3
./scripts/health.sh --env-file .env --manifest jspark3-release-manifest.json
./scripts/verify.sh --env-file .env --manifest jspark3-release-manifest.json \
  --output verify.json --log-output verify-rank0.log
```

Stop while preserving the exact containers, or remove them:

```bash
./scripts/rollback.sh --env-file .env --manifest jspark3-release-manifest.json
./scripts/stop.sh --env-file .env --manifest jspark3-release-manifest.json \
  --confirm STOP-JSPARK3 --remove --remove-confirm REMOVE-JSPARK3
```

Every command accepts `--dry-run`, which renders the exact remote commands
without contacting any host. The release validator runs those dry-runs.

## Guarantees and refusals

The controller refuses to start when any rank's preflight row differs from
the expected row, when the preflight checksum does not match, when a container
with the release name already exists, or when the confirm token is wrong. The
entrypoint refuses on cgroup limits other than 64 GiB with swap off, on any
NCCL protocol, algorithm, or address-range override, on environment drift, on
a missing or unbound image receipt, and on any overlay or loader hash drift.
Transforms are transactional: an interrupted apply is completed or rolled back
from observed hashes on the next run.

See `docs/REPRODUCIBILITY.md` and `docs/LIMITATIONS.md` in this directory.
Licenses: `LICENSE`, `THIRD_PARTY_NOTICES.md`, `REQUIRED_ATTRIBUTION.md`.
