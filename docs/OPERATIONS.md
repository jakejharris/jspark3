# Operations

How JSpark3 v1 behaves once it is up, what to watch, and how to change it
without losing the guarantees.

## The running shape

- Three containers, one per Spark, deterministic names, `--restart no`. Rank 0
  owns the API. Ranks 1 and 2 are headless workers.
- Each container: host network, host IPC, all GPUs, `/dev/infiniband`,
  `IPC_LOCK`, 32 GiB shared memory, a private cgroup with `memory.max` at
  64 GiB and swap disabled. The entrypoint refuses any other limit.
- Read-only mounts: model root, recipe root, FlyCockpit sources. Writable:
  the work root (evidence, compiler and runtime caches).
- Serving envelope (from `recipe/config/profile.json`): TP 3, PP 1, EP on,
  multiprocessing executor, EXL3 target, DFlash2 with 7 speculative tokens,
  FP8 KV cache, prefix caching, 1,000,000-token maximum model length, 32
  sequences, 8,192 batched tokens, full-decode-only CUDA graphs at 8, 16, 24,
  32 and 48, GPU memory utilization 0.83, `glm47` tool parser, `glm45`
  reasoning parser, thinking disabled by default, served name
  `glm-5.3-flash`.
- No FlashInfer autotune, no `NCCL_PROTO`, `NCCL_ALGO`, or
  `NCCL_IB_ADDR_RANGE` overrides. The fabric settings the controller injects
  are listed in `recipe/scripts/fleetctl.py` and are not operator-tunable.

## Daily checks

```bash
./scripts/status.sh --env-file .env --manifest jspark3-release-manifest.json
```

Status reports each rank's container state against the release manifest and
the API readiness on rank 0. Run `verify.sh` after any host change, reboot, or
driver update; it re-runs the fixed witness and the fabric counter checks and
writes a fresh `verify.json`. Keep `verify.json` and the release manifest with
your change records. They contain your hostnames, addresses, and container
identities, so treat them as private operational data.

## What to expect from the workload

The numbers in `docs/BENCHMARKS.md` are the only ones the project stands
behind, and they carry their measurement conditions. Two operational
consequences are worth stating plainly:

- Long prompts are expensive on first sight. A 113,908-token prompt took
  92.290 s to first token in the matched prefill measurement. Prefix caching
  makes repeats cheap; cold prompts are not.
- High concurrency raises time to first token sharply. In the C48 wave, the
  p90 time to first token was 96.722 s even though aggregate throughput rose.
  If you serve interactive traffic, cap concurrency well below 48 or add an
  admission layer in front of the endpoint.

## Security posture

The endpoint has no authentication, no TLS, and binds to the address you set
in `.env`. Place it behind a gateway that terminates TLS, authenticates, and
rate-limits. The privileged container settings are required for RDMA and
cannot be relaxed by configuration; run the fleet on a network you control.

## Changing things

Any change to the recipe directory changes its manifest hash, and the running
containers were bound to the previous hash. The supported path is: stop,
change on the controller, re-verify `SHA256SUMS`, re-sync, preflight, start.
There is no live reload and no partial re-pin.

Changing the image, a checkpoint revision, or the overlay is a new release,
not an operation. `docker/README.md` documents the re-pin procedure and the
Spark verification it requires.

## Incident handling

- `rollback.sh` stops all ranks and preserves the containers and their logs
  for inspection. Removal requires the separate `REMOVE-JSPARK3` token.
- A container that exits at start prints exactly one `REFUSE:` line naming
  the failed gate. Fix the cause; do not bypass the gate.
- Out-of-memory, swap use, or restart events show up in the cgroup counters
  the preflight and verify paths read. The measured campaign recorded zero of
  each; if you see any, treat the run as invalid evidence and investigate the
  host.

## Upgrading

Watch the repository's releases. A new tag with a changed transform contract,
overlay hash, or serving envelope will ship with a fresh three-node
verification receipt. Do not mix recipe versions across ranks; the preflight
refuses it, and it would not work anyway.
