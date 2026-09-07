# Installation

This guide takes three NVIDIA DGX Sparks from bare Docker hosts to a verified
JSpark3 v1 endpoint. Every step is fail-closed: the recipe refuses to continue
when an input differs from the pinned one, so read the refusal text rather
than forcing past it.

Status: [v1.0.0](https://github.com/jakejharris/jspark3/releases/tag/v1.0.0)
was released 2026-09-02. The [v1.1.0 (Cadence) release](https://github.com/jakejharris/jspark3/releases/tag/v1.1.0),
dated 2026-09-07, carries the kernel fix in its transform contract plus the
two measured Cadence serving features. The command below selects that exact
tag; it does not fall back to an older release or the default branch.
The attributed target mirror is public at
<https://huggingface.co/jakejharris/jspark3> and remotely verified at the
weights-mirror revision
`e7c34dba923916754cfcb0bdf6c2c75a9b7ff1fc`. No JSpark3 GHCR image is
published for v1.0.0. The recipe uses the exact upstream serving image by
digest shown below. Obtain the recipe directory from the public GitHub tree or
release asset; the commands below do not change.

> **Read this before serving long contexts or launching containers by hand.**
> The v1.0.0 recipe as published aborts on any single-stream request whose
> context passes 32,768 tokens, and a hand launch needs one NCCL variable
> that v1.0.0 never documented. Both are fixed in v1.0.1. If you are setting
> up v1.0.0 anyway, read
> [Known issue: single-stream requests past 32,768 tokens](#known-issue-in-v100-single-stream-requests-past-32768-tokens)
> before your first request, and set the fabric environment from
> [step 5](#5-fabric-checks) if anything other than `start.sh` creates your
> containers.

## 1. Requirements

Hardware and network

- Three NVIDIA DGX Sparks (GB10, aarch64). The recipe refuses any other count.
- Two RoCE-v2 capable interfaces per Spark, cabled as a triangle: each pair of
  Sparks shares one direct leg. Each leg is its own IPv4 network, MTU 9000.
  A switchless triangle also requires `NCCL_IB_SUBNET_AWARE_ROUTING=1` (new
  in NCCL 2.30.7, off by default); step 5 shows the full fabric environment.
- A management network reachable from your controller host for SSH, the API,
  and the Gloo and TP sockets.
- One common RoCE-v2 IPv4 GID index on all six HCAs. The measured fleet saw
  index 3; check yours with `show_gids` or the sysfs paths in step 5.

Software on each Spark

- Docker with the NVIDIA container runtime, cgroup v2, `rdma-core`
  (`/dev/infiniband` present), and non-interactive SSH access for a user with
  Docker rights.
- At least 72 GiB of available host memory at preflight and at least 8 GiB
  free on the model and work filesystems. The model tree itself needs about
  180 GB per rank because every rank holds the full target checkpoint.

Controller host

- Any Linux or macOS machine with Python 3.9 or newer, `ssh`, and
  `sha256sum`. Rank 0 itself works as the controller.

## 2. Get the recipe onto every rank

Unpack the recipe to the same absolute path on all three Sparks. That path
becomes `JSPARK_RECIPE_ROOT` in `.env`, and the preflight refuses if the three
copies do not hash to the same recipe manifest.

```bash
# on the controller
tar -xzf jspark3-recipe-1.0.0.tar.gz          # produces ./recipe
(cd recipe && sha256sum -c SHA256SUMS)             # every line must say OK
for host in rank0 rank1 rank2; do
  rsync -a --delete recipe/ "$host":/srv/jspark3-recipe/
done
```

The recipe directory is mounted read-only inside the containers. Never edit it
in place on a rank; change it on the controller, re-verify, and re-sync.

## 3. Fetch the pinned inputs on every rank

Image, pinned by digest (about the size of a full vLLM CUDA image):

```bash
docker pull ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58
```

Checkpoints, pinned by revision, into directory names the preflight expects
under `JSPARK_MODEL_ROOT`:

```bash
export JSPARK_MODEL_ROOT=/srv/models
pip install -U "huggingface_hub[cli]"
huggingface-cli download Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw \
  --revision 25a44fdbf16862a46b7cc9921142c6c81350af2f \
  --local-dir "$JSPARK_MODEL_ROOT/Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw-25a44fdb"
huggingface-cli download incoai/GLM-5.3-Flash-DFlash2 \
  --revision dc77ff1c99eeb2df044ee3d4f0094eb033fee410 \
  --local-dir "$JSPARK_MODEL_ROOT/incoai--GLM-5.3-Flash-DFlash2-dc77ff1c-native"
```

You may fetch the target checkpoint from the JSpark3 mirror instead. It is an
exact, hash-verifiable copy of the same repository at the same revision, so
either source produces byte-identical serving files:

```bash
mirror=https://huggingface.co/jakejharris/jspark3
huggingface-cli download "${mirror#https://huggingface.co/}" \
  --revision e7c34dba923916754cfcb0bdf6c2c75a9b7ff1fc \
  --local-dir "$JSPARK_MODEL_ROOT/Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw-25a44fdb"
```

**The preflight validates bytes, not URLs.** It checks the checkpoint's hashes
against the pinned contract regardless of where you downloaded it from, so a
mirror that differs by one byte fails exactly as an altered upstream would. The
mirror is described file by file in `huggingface/jspark3/WEIGHTS-MANIFEST.json`
and can be checked directly with `python3 tools/mirror_weights.py verify <dir>`.
The immutable main revision above contains the remotely verified target mirror;
the pinned upstream repository remains an equivalent source.

The downloaded files must be regular files, not symlinks into a cache; the
checkpoint validator refuses symlinked serving files. The target repository
ships its own `SHA256SUMS`, which the validator uses as the ledger.

FlyCockpit sources at the pinned commit, read-only:

```bash
git clone https://github.com/FlyCockpit/GLM-5.3-Flash-EXL3-3x-DGX-Sparks \
  /srv/sources/FlyCockpit-GLM-5.3-Flash-EXL3-3x-DGX-Sparks
git -C /srv/sources/FlyCockpit-GLM-5.3-Flash-EXL3-3x-DGX-Sparks \
  checkout 9093765c757bd1976372196e44af84a67cf86bad
```

Read the licenses before serving. The target checkpoint is
attribution-required and the draft is non-commercial research and evaluation
use; see `docs/LICENSING.md`.

## 4. Build and validate the runtime views

The TP3 layout needs padded head counts in the model configs. The recipe never
edits the downloaded trees; it creates sibling "runtime view" directories that
symlink every file and carry a derived `config.json` whose hash is pinned.

```bash
cd /srv/jspark3-recipe
python3 scripts/prepare_runtime_views.py \
  --target-source "$JSPARK_MODEL_ROOT/Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw-25a44fdb" \
  --target-view   "$JSPARK_MODEL_ROOT/Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw-25a44fdb-tp3-runtime" \
  --draft-source  "$JSPARK_MODEL_ROOT/incoai--GLM-5.3-Flash-DFlash2-dc77ff1c-native" \
  --draft-view    "$JSPARK_MODEL_ROOT/incoai--GLM-5.3-Flash-DFlash2-dc77ff1c-native-tp3-runtime"
python3 scripts/validate_checkpoint.py \
  --target-root    "$JSPARK_MODEL_ROOT/Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw-25a44fdb" \
  --target-runtime "$JSPARK_MODEL_ROOT/Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw-25a44fdb-tp3-runtime" \
  --draft-root     "$JSPARK_MODEL_ROOT/incoai--GLM-5.3-Flash-DFlash2-dc77ff1c-native" \
  --draft-runtime  "$JSPARK_MODEL_ROOT/incoai--GLM-5.3-Flash-DFlash2-dc77ff1c-native-tp3-runtime"
```

The validator hashes all 120 target shards, so it takes a while. It prints one
JSON line with `serving_checkpoint_pass: true` on success. Run both commands on
every rank.

## 5. Fabric checks

On each Spark, for each of its two fabric interfaces:

```bash
cat /sys/class/net/<iface>/mtu                                  # must be 9000
cat /sys/class/infiniband/<hca>/ports/1/gid_attrs/types/<idx>   # must contain v2
cat /sys/class/infiniband/<hca>/ports/1/gid_attrs/ndevs/<idx>   # must be <iface>
cat /sys/class/infiniband/<hca>/ports/1/gids/<idx>              # IPv4-mapped address of the leg
```

The same `<idx>` must satisfy all six HCAs; it becomes `JSPARK_IB_GID_INDEX`.
The preflight repeats these checks and refuses on any mismatch.

### Fabric environment (required for hand launches)

The lifecycle controller injects a fixed fabric environment into every
container (the `rank_env` table in `recipe/scripts/fleetctl.py`). Containers
created by `start.sh` get it automatically. If you launch containers with
your own tooling, set the same environment yourself. The one variable that
is new in NCCL 2.30.7 and defaults to off is `NCCL_IB_SUBNET_AWARE_ROUTING`:

```bash
export NCCL_NET=IB NCCL_NET_PLUGIN=none NCCL_IB_DISABLE=0
export NCCL_IB_HCA="<both HCA names of this rank, comma-separated>"
export NCCL_IB_GID_INDEX="<JSPARK_IB_GID_INDEX from above>"
export NCCL_IB_ROCE_VERSION_NUM=2 NCCL_IB_ADDR_FAMILY=AF_INET
export NCCL_IB_SUBNET_AWARE_ROUTING=1   # new in NCCL 2.30.7, defaults to 0
export NCCL_CROSS_NIC=0 NCCL_IB_MERGE_NICS=0 NCCL_NVLS_ENABLE=0
export NCCL_CUMEM_ENABLE=0 NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_SOCKET_IFNAME=<management interface>
export GLOO_SOCKET_IFNAME=<management interface>
export TP_SOCKET_IFNAME=<management interface>
export MN_IF_NAME=<management interface>
export VLLM_HOST_IP=<this rank's management IPv4 address>
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
```

With `NCCL_IB_SUBNET_AWARE_ROUTING=0` (the stock default), NCCL pairs NIC
index to NIC index. On a switchless triangle that means rank 0 reaches
rank 2 over the connection to rank 1, and NCCL init stalls or hangs on a
fleet that is otherwise correctly cabled. With it set to 1, both sides pick
the local NIC whose subnet matches the peer, which is what one distinct
IPv4 network per leg is for.

Preloading or substituting a different NCCL build is outside the verified
recipe. The pinned image ships stock NCCL `2.30.7+cuda13.3` and every step
and measurement here assumes it. A host-built NCCL brought in under
`LD_PRELOAD` is one plausible route for a stray stub `libcuda` in the launch
path; that specific mechanism was reported secondhand and remains unconfirmed
without the affected user's logs.

## 6. Configure `.env` on the controller

```bash
cd recipe
cp .env.example .env
```

Fill every key. The parser refuses missing, unknown, duplicate, or
shell-expanded keys, placeholder values, non-canonical paths, non-IPv4
management addresses, a master address that is not rank 0's management
address, and any inherited `NCCL_PROTO`, `NCCL_ALGO`, or
`NCCL_IB_ADDR_RANGE` in the controller's environment.

| Key | Meaning |
|---|---|
| `JSPARK_RANK{0,1,2}_HOST` | SSH target for each rank, safe non-shell form. |
| `JSPARK_RANK{0,1,2}_ADDR`, `JSPARK_MASTER_ADDR` | Management IPv4 addresses; master equals rank 0. |
| `JSPARK_FABRIC_IFACES_n`, `JSPARK_FABRIC_ADDRS_n`, `JSPARK_HCAS_n` | The two fabric legs of rank n: interface names, CIDR addresses, HCA names. |
| `JSPARK_SOCKET_IFNAME_n` | Management interface used for Gloo and TP sockets on rank n. |
| `JSPARK_IB_GID_INDEX` | The common RoCE-v2 IPv4 GID index from step 5. |
| `JSPARK_MODEL_ROOT`, `JSPARK_RECIPE_ROOT`, `JSPARK_FLY_ROOT` | Read-only roots, identical on every rank. |
| `JSPARK_WORK_ROOT` | Writable per-rank root for evidence and caches. |
| `JSPARK_MASTER_PORT`, `JSPARK_API_BIND`, `JSPARK_API_PORT` | Torch master port and the API bind address and port on rank 0. |

Keep `.env` out of version control; the recipe's `.gitignore` already lists it.

## 7. Preflight, start, verify

```bash
./scripts/clean-room-setup.sh --env-file .env --output preflight.json
preflight_sha=$(sha256sum preflight.json | cut -d' ' -f1)
./scripts/start.sh --env-file .env --preflight preflight.json \
  --preflight-sha256 "$preflight_sha" --confirm START-JSPARK3
./scripts/health.sh --env-file .env --manifest jspark3-release-manifest.json
./scripts/verify.sh --env-file .env --manifest jspark3-release-manifest.json \
  --output verify.json --log-output verify-rank0.log
```

What happens:

1. `clean-room-setup.sh` checks `SHA256SUMS`, then runs the read-only
   preflight on all three ranks over SSH: architecture, GPU, image digests,
   fabric legs, GID entries, management routes, checkpoint identity, recipe
   manifest, free memory and disk, and the absence of a container with the
   release name.
2. `start.sh` binds to that preflight's SHA-256, mints one image receipt per
   rank, creates the three containers with deterministic names, and starts
   them in the order rank 2, rank 1, rank 0. It writes
   `jspark3-release-manifest.json`, which every later command requires.
3. Inside each container the entrypoint applies the five transforms and the
   W8A16 overlay, prints `JSPARK3_STARTUP_PATCH_PASS`, and starts vLLM. Rank 0
   serves the API on `JSPARK_API_BIND:JSPARK_API_PORT`.
4. `verify.sh` checks shard identity and CUDA graph capture in the rank 0 log,
   then runs the fixed one-warmup, three-score focused witness against the
   endpoint with fabric counter checks, and writes `verify.json`.

Weight loading and graph capture take several minutes; `health.sh` reports
`STARTING` until the API answers. Add `--dry-run` to any command to print the
exact remote commands without contacting a host.

### Known issue in v1.0.0: single-stream requests past 32,768 tokens

The v1.0.0 recipe as published aborts on the first decode step of any
single-stream request whose context passes 32,768 tokens. The server log
ends with:

```
persistent_topk would oversubscribe and the FilteredTopK fallback
requires >=128KB smem per block
```

What happens: the sparse-attention indexer's decode top-k runs vLLM's
`persistent_topk` kernel, which sizes itself from the configured
`max_model_len`. At 1,000,000 tokens and a single decoding stream under
DFlash2 it requests 85 cooperative CTAs against the 48 a GB10 can hold
residently, and the 128 KiB fallback needs more shared memory than a GB10
exposes. Two or more concurrently decoding sequences do not hit it; a lone
stream always does once its context passes 32,768 tokens. The v1.0.0
transform contract pinned that kernel enabled, while every measured run and
every published benchmark executed with it disabled, so the crash was never
exercised before release. `verify.sh` in v1.0.0 uses short prompts only and
cannot catch it.

Use the current recipe, which carries the disable in the transform itself
(v1.0.1 introduced this; v1.1.0 carries it forward); weights, benchmarks, and
the serving envelope are unchanged. Integrated verification has now passed
one pinned single-stream request with 48,957 prompt tokens and 51 completion
tokens on the assembled candidate. [Receipt and provenance](../results/evidence/candidate/cadence-v11/README.md#integrated-live-verification)
record the candidate and host verifier separately; the final archive has not
been cold-boot tested, and maximum-context capacity remains unverified.
If you cannot move off
v1.0.0, your options are all unsupported: keep every request's
total context at or below 32,768 tokens; hand-launch `vllm serve` with
`--max-model-len 406656` or lower (a ceiling derived from the kernel's
CTA arithmetic, documented in
[docs/LIMITATIONS.md](LIMITATIONS.md#kernel-disable-provenance-and-the-32768-token-single-stream-boundary-v100));
or apply the upstream one-line disable yourself, which the v1.0.0 patch
contract correctly refuses because it changes the pinned file hash. Do not
edit the recipe scripts to change `--max-model-len`: they are hash-pinned,
and the fleet refuses a modified recipe.

## 8. First request

```bash
curl -s http://<rank0-management-address>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "glm-5.3-flash",
        "messages": [{"role": "user", "content": "Write a haiku about three small computers."}],
        "max_tokens": 200,
        "chat_template_kwargs": {"enable_thinking": false}
      }'
```

Thinking is disabled by default in the chat template arguments; pass
`"enable_thinking": true` per request to enable it. Tool calling uses the
`glm47` parser and reasoning the `glm45` parser. `scripts/api_smoke.py
--base-url http://<address>:8000` runs a small end-to-end check.

If you are serving the v1.0.0 recipe, keep this first request short: a
single stream whose context passes 32,768 tokens aborts the server, per the
[known issue above](#known-issue-in-v100-single-stream-requests-past-32768-tokens).

## 9. Stop and remove

```bash
./scripts/rollback.sh --env-file .env --manifest jspark3-release-manifest.json
./scripts/stop.sh --env-file .env --manifest jspark3-release-manifest.json \
  --confirm STOP-JSPARK3 --remove --remove-confirm REMOVE-JSPARK3
```

`rollback.sh` stops the ranks and preserves the exact containers for
inspection. Removal needs its own confirm token.

## Troubleshooting refusals

| Refusal | Cause |
|---|---|
| `REFUSE: cgroup requires memory.max=68719476736 and memory.swap.max=0` | The container was not created by the controller, or the host rewrote cgroup limits. |
| `REFUSE: forbidden fabric override` | `NCCL_PROTO`, `NCCL_ALGO`, or `NCCL_IB_ADDR_RANGE` is set. Unset it. |
| `REFUSE: JSpark3 overlay hash drift` | The recipe copy on that rank differs from the verified one. Re-sync from the controller. |
| `pinned Fly source mismatch` | The FlyCockpit checkout is not at the pinned commit or was modified. |
| `fabric interface missing or MTU is not 9000` | Fix the interface configuration and rerun the preflight. |
| `target SHA256SUMS hash drift` or `target checksum mismatch inventory drift` | The checkpoint download is incomplete or from a different revision. |

Runtime errors (not refusals):

| Runtime error | Cause |
|---|---|
| `persistent_topk would oversubscribe and the FilteredTopK fallback requires >=128KB smem per block` | v1.0.0 known issue: a single decoding stream passed 32,768 tokens of context. Use the v1.0.1 recipe; see the [known issue](#known-issue-in-v100-single-stream-requests-past-32768-tokens). |
| NCCL init stalls; rank 0 to rank 2 connects over rank 1's leg | Hand launch without `NCCL_IB_SUBNET_AWARE_ROUTING=1`. Set the full fabric environment from [step 5](#5-fabric-checks). |

Every refusal exits with status 9 and leaves the fleet unchanged.
