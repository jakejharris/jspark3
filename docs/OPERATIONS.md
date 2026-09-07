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
  `glm-5.3-flash`. v1.1 adds the width controller and the QKV decode shadow
  to the measured launch; the recipe's pinned profile and launch contracts
  are authoritative for the exact envelope.
- No FlashInfer autotune, no `NCCL_PROTO`, `NCCL_ALGO`, or
  `NCCL_IB_ADDR_RANGE` overrides. The fabric settings the controller injects
  are not operator-tunable. Since v1.0.1 they are also documented in
  [installation step 5](INSTALL.md#5-fabric-checks); the full per-rank set is
  `NCCL_NET=IB`, `NCCL_NET_PLUGIN=none`, `NCCL_IB_DISABLE=0`,
  `NCCL_IB_HCA` (both of the rank's HCAs), `NCCL_IB_GID_INDEX`,
  `NCCL_IB_ROCE_VERSION_NUM=2`, `NCCL_IB_ADDR_FAMILY=AF_INET`,
  `NCCL_IB_SUBNET_AWARE_ROUTING=1`, `NCCL_CROSS_NIC=0`,
  `NCCL_IB_MERGE_NICS=0`, `NCCL_NVLS_ENABLE=0`, `NCCL_CUMEM_ENABLE=0`,
  `NCCL_IGNORE_CPU_AFFINITY=1`, and the shared management interface for
  `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, and
  `MN_IF_NAME`. `NCCL_IB_SUBNET_AWARE_ROUTING` is new in NCCL 2.30.7 and
  defaults to off; with it off, NCCL pairs NICs by index and routes rank 0
  to rank 2 over rank 1's leg, which breaks a switchless triangle. A
  preloaded or substituted NCCL build is outside the verified recipe.

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
- In v1.1, single-stream decode may run in the narrow speculative mode when
  the width controller's acceptance checks pass, and silently falls back to
  the wide path for any batch of two or more requests, prefill, or guard
  condition. The fallback is normal behavior, not an error, and the measured
  single-stream gains carry no promise for a busy multi-stream service.
- Single-stream long context was a boundary in v1.0.0. The v1.0.0 recipe
  aborted on any single decoding stream past 32,768 tokens of context (the
  `persistent_topk` kernel; see the
  [install-path warning](INSTALL.md#known-issue-in-v100-single-stream-requests-past-32768-tokens)).
  v1.0.1 disables that kernel in the transform, so the known deterministic
  single-stream abort is removed and the transform emits the exact kernel
  file every measured run executed. That is not a demonstration of end-to-end
  reachability: the configured 1,000,000-token context remains a
  configuration value and does not certify this candidate's operating
  envelope, and a live single-stream witness above 32,768 tokens on the
  assembled public build is a pending verification item. All published
  numbers, including the long-prefill figures above, were measured with that
  disable applied and are unchanged.
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
- A server abort containing `persistent_topk would oversubscribe` means the
  v1.0.0 known issue was hit: a single decoding stream passed 32,768 tokens
  of context on the v1.0.0 construction. Move to the v1.0.1 recipe; the
  [install path](INSTALL.md#known-issue-in-v100-single-stream-requests-past-32768-tokens)
  lists the unsupported v1.0.0 workarounds.
- Out-of-memory, swap use, or restart events show up in the cgroup counters
  the preflight and verify paths read. The measured campaign recorded zero of
  each; if you see any, treat the run as invalid evidence and investigate the
  host.

## Upgrading

Watch the repository's releases. A new tag with a changed transform contract,
overlay hash, or serving envelope will ship with a fresh three-node
verification receipt. Do not mix recipe versions across ranks; the preflight
refuses it, and it would not work anyway.
