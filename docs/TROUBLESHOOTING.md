# Troubleshooting JSpark3

Every gate in JSpark3 is fail-closed: when a check fails, the lifecycle prints
a `REFUSE:` line (or a `Refusal` detail), **exits with status 9**, and leaves
the fleet unchanged. That is by design — the refusal text tells you exactly
which input drifted. This guide maps every refusal emitted by v1.1.0 to its
cause and fix. Refusal sources: `recipe/scripts/fleetctl.py` (controller),
`recipe/scripts/remote_preflight.py` (per-rank preflight),
`recipe/scripts/validate_checkpoint.py` (checkpoint bytes),
`recipe/scripts/container_entry.sh` (in-container startup).

First, three habits that solve most problems:

1. **Read the refusal text literally.** It names the exact key, path, or hash
   that drifted. Do not edit recipe scripts to work around a refusal; the
   recipe is hash-pinned and any edit changes its manifest, which every later
   step verifies (`rank{N} recipe changed after preflight`).
2. **Reproduce without side effects.** Add `--dry-run` to any command to print
   the exact remote commands without contacting a host. Use
   `./scripts/status.sh --env-file .env` to see fleet state at any time.
3. **Fix inputs on the controller, then re-sync.** Never edit the recipe in
   place on a rank. Change it on the controller, re-verify
   (`sha256sum -c SHA256SUMS`), and `rsync` to all three ranks again.

Recovery after any failed `start`: `rollback.sh` stops ranks and preserves the
exact containers for inspection. `stop.sh --remove` needs its own confirm
token (`REMOVE-JSPARK3`). Removal is required before a fresh `start` because
container names are deterministic.

---

## Stage 1 — `.env` parsing and validation (controller, before any SSH)

These fire before anything touches your ranks. Exit status 9, no host contact.

| Refusal | Cause and fix |
|---|---|
| `cannot read env file: ...` | `.env` missing or unreadable at the path you passed to `--env-file`. Check the path and permissions. |
| `invalid env syntax at line N` | Line N is not `KEY=value` with an uppercase key (`[A-Z][A-Z0-9_]*`). Quotes are stripped only if they wrap the whole value; the value cannot contain `$`, backticks, or newlines (`shell expansion is not supported`). |
| `duplicate env key: KEY` | KEY appears twice in `.env`. Delete one. |
| `missing env keys: A,B,...` | Those required keys are absent or empty. All 27 keys in `.env.example` must be filled. |
| `unknown env keys: A,B,...` | You added a key that is not in the contract (e.g. `JSPARK_NCCL_DEBUG`, `JSPARK_EXTRA_...`). Remove it — extra tuning knobs are deliberately not supported. |
| `forbidden inherited NCCL override` | `NCCL_PROTO`, `NCCL_ALGO`, or `NCCL_IB_ADDR_RANGE` is set in the controller's own environment. `unset` all three in the shell you run the controller from. |
| `three distinct SSH hosts and management addresses are required` | Two ranks share a `JSPARK_RANK{N}_HOST` or `JSPARK_RANK{N}_ADDR`. Each rank needs its own. |
| `SSH host labels must use a safe non-shell form` | A HOST value contains characters outside `[A-Za-z0-9_.:@-]`. Use plain hostnames or IPs — no spaces, quotes, or shell metacharacters. |
| `master address must equal rank 0 management address` | `JSPARK_MASTER_ADDR` must be exactly `JSPARK_RANK0_ADDR`. |
| `management addresses must be IPv4` / `fabric addresses must be IPv4` / `peer management addresses must be IPv4` | IPv6 or hostnames are not accepted in ADDR/CIDR fields. Use dotted-quad IPv4. |
| `rank{N} fabric interfaces must contain two distinct comma-separated values` (same for HCAs, fabric addresses) | Each rank has exactly two fabric legs. `FABRIC_IFACES_n`, `FABRIC_ADDRS_n`, and `HCAS_n` are two values, comma-separated, distinct, no spaces around commas (the parser also rejects `"a, b"` because the rejoined string must match exactly). |
| `rank{N} fabric legs share one network` | Both CIDRs of rank N are in the same subnet. Each leg needs its own IPv4 network — that is what makes the switchless triangle route correctly. |
| `fabric address reused` | The same fabric IP appears on more than one leg. Six distinct fabric IPs across the fleet. |
| `fabric CIDRs do not form a three-edge pairwise triangle` | The six legs must form exactly three /networks (subnets), each shared by exactly one pair of ranks. Re-cable or re-address so rank0–rank1, rank1–rank2, and rank0–rank2 each have a dedicated subnet. |
| `invalid common GID index` | `JSPARK_IB_GID_INDEX` must be an integer 0–255. |
| `JSpark3 v1 requires master port 29533 and API bind 0.0.0.0:8000` | `JSPARK_MASTER_PORT`, `JSPARK_API_PORT`, and `JSPARK_API_BIND` are pinned. Change nothing here. |
| `JSPARK_{MODEL,WORK,RECIPE,FLY}_ROOT must be an absolute normalized path` | Roots must be absolute, contain no `..`, and have at least three path components (so `/srv/models`, not `/models`). |
| `model, work, recipe, and Fly roots must be distinct` | The four roots must be four different directories. |
| `work root must not contain or be contained by an immutable root` | `JSPARK_WORK_ROOT` may not sit inside (or contain) the model/recipe/Fly roots — the immutable roots are mounted read-only, the work root is writable. |
| `documentation placeholder remains in env` | You left a `.example.invalid` host, or a `198.51.100.*` / `192.0.2.*` documentation IP in `.env`. Replace every placeholder with your real values. |

## Stage 2 — SSH and remote execution

| Refusal | Cause and fix |
|---|---|
| `rank{N} remote command failed: ...` (often `Permission denied` or `Connection timed out`) | The controller SSHes to each rank with `BatchMode=yes` — **non-interactive key auth is required**. Install your controller key in each rank's `authorized_keys`, verify with `ssh -o BatchMode=yes rankN true`, and check that the user has Docker rights on the rank. |
| `rank{N} invalid preflight response` | The remote preflight printed something that is not clean JSON — most often a rank-side shell profile or site hook echoing text into stdout. Check `~/.bashrc`/`~/.profile` on that rank for `echo`s on non-interactive shells. |
| `rank{N} recipe-only verification schema drift` / `rank{N} preflight did not pass` | The rank's own checks failed; the rank printed a `REFUSE:` line on stderr — read it (Stage 3/4 below). |
| `rank{N} recipe changed after preflight` | The recipe copy on that rank differs from the preflight-verified one. Re-sync from the controller (`rsync -a --delete recipe/ rankN:/srv/jspark3-recipe/`) and rerun preflight — never edit in place on a rank. |

## Stage 3 — per-rank preflight (`remote_preflight.py`, runs on each Spark)

Each rank validates hardware, image, roots, fabric, and checkpoint bytes
before the fleet is allowed to start.

### Hardware and image

| Refusal | Cause and fix |
|---|---|
| `host architecture is not aarch64` | Only DGX Sparks (GB10, aarch64) are supported, and the preflight reads the actual system product. Running on any other machine will not pass. |
| GPU inventory refusal (from `exact_gpu_inventory`) | `nvidia-smi` must report exactly the GB10 / SM121 pair. No other GPU count or model passes — the recipe refuses anything but three DGX Sparks. |
| `OCI manifest/config identity drift` | The pulled image is not the pinned one. Pull exactly `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:9bb1557a...fd58` by digest. If the pull failed partway, `docker pull` again — digest-pinned pulls are all-or-nothing, never "close enough". |
| `deterministic release container name already exists` | A container named `jspark3-rank{N}` exists from an earlier attempt. Inspect it (`docker container inspect jspark3-rank{N}`), then remove it via `stop.sh --remove-confirm REMOVE-JSPARK3` from the controller, or `docker rm` it on the rank. |
| `model, work, or Fly root is not an exact canonical path` | The rank-side roots must be real directories, **not symlinks**, and the paths must match `.env` exactly. If `/srv/models` is a symlink to somewhere else, mount or move the real directory to the declared path. |
| `less than 8 GiB free at model/work path` | Free space on the model root's filesystem and the work root's parent, each ≥ 8 GiB. The model tree itself is ~180 GB per rank — plan the filesystem accordingly. |
| `less than 72 GiB host memory available` | `/proc/meminfo` `MemAvailable` is below 72 GiB at preflight time. Stop other memory-hungry services/models on that Spark and rerun. This gate runs on the host, before containers exist. |

### Fabric (the most common field failures)

| Refusal | Cause and fix |
|---|---|
| `two distinct fabric interfaces/HCAs are required` | Each rank needs exactly two different RoCE interfaces and two different HCAs, comma-separated in `.env`. |
| `fabric interface missing or MTU is not 9000` | The named interface does not exist or its MTU is not 9000: `cat /sys/class/net/<iface>/mtu`. Set MTU 9000 on both legs of all three Sparks (and confirm the switchless triangle is cabled pairwise). |
| `declared fabric address is not assigned` | The CIDR in `JSPARK_FABRIC_ADDRS_n` is not configured on that interface. `ip -j -4 address show dev <iface>` must list it. Assign it, or fix `.env`. |
| `pinned HCA port is not ACTIVE` | The RoCE port is down: check `cat /sys/class/infiniband/<hca>/ports/1/state`. Fix cabling/link, or the driver isn't loaded (`rdma-core` must be installed, `/dev/infiniband` present). |
| `GID is not the declared active RoCE-v2 interface` / `RoCE-v2 GID does not encode the declared IPv4 address` | The GID index you pinned is not a RoCEv2/IPv4 GID bound to that interface on this HCA. Find the common index: for each of the six HCAs check `gid_attrs/types/<idx>` (must contain `v2`) and `gid_attrs/ndevs/<idx>` (must be the fabric interface), and `gids/<idx>` must be the IPv4-mapped leg address. The **same index must work on all six HCAs** — set it as `JSPARK_IB_GID_INDEX`. |
| `management/socket interface missing` / `declared management address is not assigned to the socket interface` | `JSPARK_SOCKET_IFNAME_n` names the management NIC, and `JSPARK_RANKn_ADDR` must be an IPv4 assigned to it. |
| `peer management route leaves the wrong interface` | `ip route get <peer>` from this rank must exit through the management interface. Fix routes/tables — peers must be reachable over management, not a fabric leg. |
| `pinned Fly source mismatch: NAME` | The FlyCockpit checkout at `JSPARK_FLY_ROOT` is not at the pinned commit or was modified. Re-clone and `git checkout 9093765c757bd1976372196e44af84a67cf86bad`; do not touch the sources. |
| `checkpoint serving-byte gate failed` | Read the `validate_checkpoint.py` REFUSE line in the same output (Stage 4). |

### Checkpoint bytes (`validate_checkpoint.py`)

The validator hashes the checkpoint against the pinned ledger. It refuses
symlinked serving files by design — downloads must be regular files, not
links into an HF cache.

| Refusal | Cause and fix |
|---|---|
| `target SHA256SUMS must be a regular file` / `target SHA256SUMS hash drift` | The checkpoint's own ledger is missing or altered — the download is incomplete or from a different revision. Re-download the pinned revision (target `25a44fdb...`, mirror `e7c34dba...`). |
| `malformed ledger row N` / `duplicate ledger path ...` / `unsafe ledger row N` | Corrupted or hand-edited `SHA256SUMS` in the checkpoint. Re-download; never hand-patch the ledger. |
| `target native serving files must not be symlinks` | You downloaded into an HF cache and symlinked. Download with `huggingface-cli download --local-dir` to a real directory so files are regular. |
| `target publication-only omission inventory drift` / `target shard inventory drift` | Files are missing or unexpected files are present. The publication layout intentionally omits some rows (runtime/, materialization stubs); anything else missing means an incomplete download. |
| `target checksum mismatch inventory drift` | Present files hash differently than the ledger. **Note:** differences in exactly `LICENSE` and `README.md` are allowed and expected (upstream re-license); anything else is a real mismatch — re-download. |
| `target serving-critical drift: NAME` | One of `config.json`, `model.safetensors.index.json`, `tokenizer.json`, `tokenizer_config.json` differs from the pinned hash. Re-download the pinned revision. |
| `target indexed tensor-byte total drift` / `target index mapping drift` / `target physical safetensors-byte total drift` | The safetensors index and the 120 shards disagree (120 shards, exact byte totals are pinned). Truncated shard download — re-fetch. |
| `runtime config drift` / `runtime-view inventory drift` / `runtime-view link drift: NAME` / `runtime view must be a real sibling directory` | The TP3 runtime view was not produced by `prepare_runtime_views.py`, was hand-edited, or is not a sibling symlink-view of the native root. Rerun `prepare_runtime_views.py` with the documented `--target-view`/`--draft-view` paths, then `validate_checkpoint.py`. |
| `draft serving-critical drift: NAME` / `draft model byte-size drift` / draft GQA/tap/causality drift | The DFlash2 draft checkpoint (pinned revision `dc77ff1c...`, single `model.safetensors`) differs. It is fetched separately and **not mirrored** — re-download the pinned revision. |
| `serving checkpoint validation failed: ...` / `REFUSE: workers must be in [1,16]` | Wrapper-level failure; the inner refusal above has the real cause. |

## Stage 4 — start (`fleetctl.py start`)

| Refusal | Cause and fix |
|---|---|
| `start confirmation must be START-JSPARK3` | Pass `--confirm START-JSPARK3`. The explicit token is deliberate. |
| `preflight file SHA-256 mismatch` | The `--preflight` file changed after you computed `--preflight-sha256`. Rerun preflight and pass the fresh hash: `preflight_sha=$(sha256sum preflight.json | cut -d' ' -f1)`. |
| `preflight receipt does not bind this exact configuration` | The preflight was made with a different `.env`, recipe, or image than the start is using. Any change to `.env` or the recipe invalidates it — rerun `clean-room-setup.sh`, then start from the new receipt. |
| `invalid JSON receipt` / `receipt payload hash mismatch` / `receipt must be an object` / `receipt missing or symlinked` | The preflight receipt file is corrupt, hand-edited, or a symlink. Regenerate it; receipts are checksummed and will not validate after manual edits. |
| `local recipe manifest missing or unsafe` | The controller's own `recipe/SHA256SUMS` is missing, altered, or a symlink. Restore it from the release (`git checkout v1.1.0 -- recipe/SHA256SUMS` or re-clone) — do not regenerate it by hand. |
| `release manifest already exists` | A previous start wrote `jspark3-release-manifest.json`. Recover first: `rollback.sh` (stop + preserve), then `stop.sh --remove --remove-confirm REMOVE-JSPARK3` to delete the old containers and manifest. A fresh start needs a clean slate. |
| `rank{N} docker create returned an unsafe identity` | `docker create` failed or returned something unexpected — check Docker health on that rank (`docker info`), disk space, and that the pinned image is present (`docker image inspect` with the digest from INSTALL.md step 3). |
| `rank{N} container identity/argv contract drift` | The created container does not match the pinned contract (name, workdir, entrypoint, exact vLLM argv). This is checked post-create and catches images/recipes that drifted. Confirm the rank uses the pinned image and a hash-identical recipe copy. |
| `rank{N} required environment drift` / `rank{N} forbidden environment present` | The container's environment does not exactly match what the controller injected, or a forbidden NCCL override is present. Don't modify `rank_env` in `fleetctl.py` or set NCCL variables in the rank's Docker environment. |
| `rank{N} namespace/resource contract drift` / `... memlock contract drift` / `... capability/security contract drift` / `... InfiniBand device contract drift` / `... GPU request contract drift` | The container's runtime options differ from the pinned set (host networking, IPC, shm 32 GiB, memory 64 GiB, memlock unlimited, `IPC_LOCK`, `/dev/infiniband`, all GPUs, cgroup namespace private). These are set by the controller itself — if you see this, the container was not created by this controller version. |
| `rank{N} mount contract drift` / `duplicate mount destination` | The bind mounts differ from the pinned set (`/recipe`, `/sources/fly`, `/models` read-only; per-rank `/evidence` and cache dirs writable). Root causes are usually a non-canonical root path on the rank (symlinked roots) or leftover containers from an older recipe. |
| `rank{N} created container started unexpectedly` / `rank{N} container running before ordered start` | A container the controller had only created (not yet started) was found running. Something on the rank auto-started it (e.g. a `--restart` policy from a hand-made container with the same name). Remove the foreign container and rerun. |
| `rank{N} container admission state drift` | The container record shows `OOMKilled` or restarts at create time — the rank was under memory pressure during create. Free memory and recreate. |
| `manifest does not bind this environment/image` (verify/status) | The release manifest on disk was made with a different `.env`, recipe, or image than the current controller run — or the file was hand-edited (manifests carry a payload hash). Regenerate the manifest with a fresh start; never hand-edit. |
| `manifest rank set drift` / `manifest container binding drift` | The manifest's container rows are incomplete, out of order, or carry malformed IDs — normally the result of manual editing or a truncated write. Recreate via rollback/stop/remove + start. |
| `rank{N} manifest-bound container missing` (during verify/status) | The container ID recorded in the release manifest no longer exists on the rank — it was removed outside the lifecycle. Rebuild via rollback/stop/remove + start. |
| `rank{N} running container has no host PID` (status) | The container is running but the controller cannot resolve its host PID from the cgroup — typical after host Docker/cgroup driver changes. Restart the fleet from the controller. |
| `rank{N} cadence module install drift` / `rank{N} cadence capture evidence missing or drifted` (verify) | The Cadence (B4/B5) runtime modules or their capture receipts under `JSPARK_WORK_ROOT` don't match the pinned contract. The caches are per-rank under `work/rankN/cache/` — clear that rank's cache, recreate the fleet, and rerun verify. |
| `START REFUSED; at least one created container state is unconfirmed; inspect exact manifest IDs` | Start failed partway and the controller could not confirm every created container is stopped. Inspect the container IDs in the failure manifest on their ranks (`docker container inspect <id>`), stop leftovers, then remove and restart. |

## Stage 5 — in-container startup (`container_entry.sh`)

These fire inside each container during `docker start`. `docker logs
jspark3-rank{N}` (on the rank) shows them. Exit status 9, container stops.

| Refusal | Cause and fix |
|---|---|
| `REFUSE: cgroup requires memory.max=68719476736 and memory.swap.max=0` | The container's cgroup limits were rewritten by the host (or the container wasn't created by the controller). The pinned contract is exactly 64 GiB limit, no swap. Check for host-level cgroup policies/daemons rewriting limits, and rerun start from the controller. |
| `REFUSE: entry cgroup already has swap or OOM events` | This cgroup already recorded swap use or OOM kills before the entrypoint ran. Usually means a previous process ran in the same container, or the host forced swapping. Recreate the fleet (rollback/stop/remove + start). |
| `REFUSE: forbidden fabric override NCCL_*` | `NCCL_PROTO`, `NCCL_ALGO`, or `NCCL_IB_ADDR_RANGE` leaked into the container env. Unset them on the controller (and in any rank-side Docker/systemd env files) — they are deliberately not configurable. |
| `REFUSE: combined KDA environment drift` / `REFUSE: JSpark3 W8A16 environment drift` / `REFUSE: Cadence B4+B5 environment drift` | A pinned serving-environment variable was altered (KDA blocks=8, FG batched=1, W8A16 group=64, B4/B5 constants). These come from the controller's `rank_env`; if you hand-launched, copy the full environment from step 5 of INSTALL.md exactly. |
| `REFUSE: host-minted image receipt is missing or unsafe` | `/evidence/image-receipt.json` is absent or a symlink. It is installed by the controller during start; if you hand-launched, no receipt was minted — hand launches are outside the verified lifecycle. |
| `REFUSE: host receipt binding environment is incomplete` | `NODE_RANK`, `JSPARK_PREFLIGHT_SHA256`, or `JSPARK_RECIPE_MANIFEST_SHA256` is missing/malformed in the container env — again a hand-launch symptom. |
| `REFUSE: JSpark3 overlay hash drift` / `REFUSE: JSpark3 loader patcher hash drift` | The recipe copy mounted at `/recipe` differs from the verified one on that rank. Re-sync the recipe from the controller. |
| `REFUSE: installed JSpark3 overlay hash drift` | The overlay file inside the vLLM tree was changed after install (or a stale container image state). Recreate the fleet; do not modify installed files. |
| `REFUSE: target runtime path drift` / `REFUSE: draft runtime path drift` | `MODEL_PATH`/`DRAFT_PATH` don't match the pinned runtime-view paths — the model root on that rank doesn't contain the expected sibling runtime views. Rerun `prepare_runtime_views.py` on that rank. |
| `pinned Fly source mismatch: NAME` (during apply) | FlyCockpit checkout wrong/modified — see Stage 3. |

A healthy startup ends with:
`JSPARK3_STARTUP_PATCH_PASS rank=N overlay_sha256=5aeff0cf... group704=64 runtime_modules=169 logical_tensors=225`
followed by vLLM engine startup. Weight loading and graph capture take several
minutes; `health.sh` reports `STARTING` until the API answers.

## Stage 6 — runtime errors (not refusals)

| Symptom | Cause and fix |
|---|---|
| `persistent_topk would oversubscribe and the FilteredTopK fallback requires >=128KB smem per block` in the rank 0 log, server aborts | **v1.0.0 only**: a single decoding stream passed 32,768 tokens of context. Use v1.1.0, which carries the kernel disable. On v1.0.0 the unsupported workarounds are: keep every request ≤ 32,768 total tokens, hand-launch with `--max-model-len 406656` or lower, or apply the upstream one-line disable (the v1.0.0 patch contract correctly refuses it). Do not edit recipe scripts — they are hash-pinned. |
| NCCL init stalls/hangs on a correctly cabled fleet; rank 0 reaches rank 2 via rank 1's leg | Hand launch without `NCCL_IB_SUBNET_AWARE_ROUTING=1` (new in NCCL 2.30.7, defaults off; stock NCCL pairs NIC index to NIC index, which mis-routes on a switchless triangle). Set the full fabric environment from INSTALL.md step 5, or just use `start.sh`, which injects it. Preloading/substituting NCCL builds is outside the verified recipe. |
| Anecdotal "stray stub libcuda" crash reported with a host-built NCCL under `LD_PRELOAD` | Plausible but **unconfirmed**; if you hit it, capture full logs (`NCCL_DEBUG=INFO` is already set by the controller) and report rather than assuming. |
| Endpoint answers `/health` but requests hang at high context | The configured 1,000,000-token context is not a capacity certification. Large-context capacity remains unverified in v1.1.0. |
| `health endpoint is not HTTP 200` / `served-model identity drift` / `arithmetic gate failed` (verify) | The API is up but not serving the pinned identity (`glm-5.3-flash`) or failing the fixed arithmetic probe. Usually a partially started fleet: check rank 0 logs for engine errors, and re-verify with `health.sh` before running `verify.sh`. If the engine never became ready within the 3600 s ready timeout, look for OOM/fabric errors in `docker logs jspark3-rank0`. |
| `verify.sh` witness failures after a successful start | Rerun with the fleet idle; the witness has fixed pacing and fabric-counter checks. For repeatable evidence, results land in verify.json. |

## Quick reference

- Every refusal exits **status 9**; the fleet is unchanged or explicitly
  rolled back. `--dry-run` renders any command without host contact.
- The recipe, image, checkpoints, ports, and container contract are pinned by
  hash. When a refusal says "drift", something was modified — restore the
  pinned bytes rather than editing the checker.
- Healthy startup line: `JSPARK3_STARTUP_PATCH_PASS rank=N overlay_sha256=...`.
- Recovery sequence: `rollback.sh` → `stop.sh --confirm STOP-JSPARK3 --remove
  --remove-confirm REMOVE-JSPARK3` → fix the input → `clean-room-setup.sh` →
  `start.sh --confirm START-JSPARK3`.
