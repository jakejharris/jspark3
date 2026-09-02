# Reproducibility contract

The recipe reproduces an exact construction and its guards. It does not
guarantee identical token rates: ambient temperature, clocks, firmware, fabric
placement, request shape, cache state, and scheduler state all remain material.

## What is pinned

- Target and draft checkpoints by repository, revision, config hashes, shard
  count, and byte totals (`config/checkpoint-contract.json`,
  `scripts/validate_checkpoint.py`).
- The serving image by manifest digest and config digest, re-verified on every
  rank by the preflight and bound into a host-minted receipt at start.
- Five runtime transforms by before and after hashes of every touched file
  (`config/patch-contract.json`, `transforms/README.md`), applied inside the
  disposable container.
- The W8A16 overlay module, the loader hook patcher, and the loader file's
  before and after hashes (`config/profile.json`, `scripts/container_entry.sh`).
- The full serving argument set and environment (`config/profile.json`).
- Every recipe file by `SHA256SUMS`, which `scripts/clean-room-setup.sh`
  checks before it will run a preflight.

## What must be true on your hardware

- Exactly three NVIDIA DGX Sparks (aarch64, GB10, SM 12.1).
- Two RoCE-v2 legs per rank at MTU 9000 forming a pairwise triangle, with one
  common RoCE-v2/IPv4 GID index on all six HCAs. The measured fleet observed
  index 3; verify yours, do not assume it.
- A management network for SSH, the API, and Gloo/TP sockets.
- At least 72 GiB of available host memory per rank at preflight and at least
  8 GiB free on the model and work filesystems.
- Docker with the NVIDIA runtime, `/dev/infiniband`, and cgroup v2.

## The verification path

1. `sha256sum -c SHA256SUMS` (done by `clean-room-setup.sh`).
2. Preflight on all three ranks; the output file's SHA-256 binds the start.
3. Ordered start (rank 2, then 1, then 0) with the confirm token, which writes
   the release manifest bound to the preflight and recipe hashes.
4. Health, then `verify`, which checks shard identity, CUDA graph capture, and
   runs the fixed one-warmup, three-score focused witness against the API.

A run whose entrypoint printed `JSPARK3_STARTUP_PATCH_PASS` and whose
`verify.json` reports `PASS` reproduced the construction. Compare rates only
under the measurement conventions in the repository's `docs/BENCHMARKS.md`.
