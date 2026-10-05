# Troubleshooting

The installation and restart issues documented for v2.0.1 also apply to v2.0.2. The separate kernel-rebuild hotfix was validated with v2.0.1 on 2026-10-04; it is not a new v2.0.2 validation. The command examples below retain their documented v2.0.1 checkout and validation scope.

This page is for JSpark3 v2.0.1 (GLM-5.3 Flash). Find the message you see, then read what it means and what to do.
It is maintained on main alongside the [current install guide](../INSTALL.md); run the commands from your
`v2.0.1` tagged checkout.

- [v2.0.1 known issues and hotfixes](#v201-known-issues-and-hotfixes)
- [How script messages look](#how-script-messages-look)
- [Preflight](#preflight)
- [Settings and options](#settings-and-options): [settings profile](#settings-profile)
- [Downloads](#downloads): [the Hugging Face CLI](#the-hugging-face-cli), [the gated ablit source](#the-gated-ablit-source), [an interrupted download](#an-interrupted-download), [a file that does not match](#a-file-that-does-not-match), [disk space](#not-enough-disk-space)
- [Image, engine wheel and dependency wheels](#image-engine-wheel-and-dependency-wheels)
- [Splitting and converting](#splitting-and-converting)
- [Starting](#starting)
- [A rank never becomes ready](#a-rank-never-becomes-ready)
- [Possible stale Torch extension lock](#possible-stale-torch-extension-lock)
- [Status, stop and clearing the session store](#status-stop-and-clearing-the-session-store)
- [Checking answers: smoke and exactness](#checking-answers-smoke-and-exactness)
- [Checking the prompt cache and prompt reading](#checking-the-prompt-cache-and-prompt-reading)
- [The first reply is slow](#the-first-reply-is-slow)
- [API behaviour clients notice](#api-behaviour-clients-notice): [images in requests](#images-in-requests)
- [Reaching the server safely](#reaching-the-server-safely)
- [Asking for help](#asking-for-help)

## v2.0.1 known issues and hotfixes

These four install/startup entries supplement the release's API limitations; they do not renumber its known
issues. Source links below point to the immutable **v2.0.1** tree. Commands run in that tagged checkout unless
they explicitly invoke the checker from the separate main/docs checkout shown in [INSTALL](../INSTALL.md#quick-start-base-weights-the-default).

### Kernels rebuild on every start

**Validated on 3x DGX Spark (2026-10-04): retained boot reused the kernel cache; undo restored the original files.**

**Symptom:** later starts compile Torch extensions again despite keeping `$DATA/kernel-cache`.
**Cause:** a fresh container runs pip on every start; new compiler-source timestamps invalidate Ninja's cached
outputs (not separately measured in the validation run). The startup command is [scripts/serve.sh:125–126](https://github.com/jakejharris/jspark3/blob/v2.0.1/scripts/serve.sh#L125-L126),
the persistent mount is [scripts/serve.sh:114–116](https://github.com/jakejharris/jspark3/blob/v2.0.1/scripts/serve.sh#L114-L116),
and TensorFold calls Torch's extension loader at
[engine/src/tensorfold/cuda/build.py:23–29](https://github.com/jakejharris/jspark3/blob/v2.0.1/engine/src/tensorfold/cuda/build.py#L23-L29).

**Fix:** follow the standalone [kernel rebuild hotfix](hotfixes/v2.0.1-kernel-rebuild.md), which contains the exact
diff, before/after checksums and shutdown prerequisites. It adds the reviewed F1 timestamp helper and one
startup call; this documentation does not patch the shipped engine. **Verify:** its cold and retained-cache
boots must both pass readiness and smoke, and compiled artifact paths, timestamps and hashes must be unchanged
on the second boot on all three hosts. **Undo:** use that file's guarded reverse patch and cache restoration.
Hardware validation passed on 2026-10-04 with exactly that scope: three DGX Spark systems, retained-cache reuse and undo.

**If skipped:** starts remain slower because of compilation, with repeated exposure to the **hypothetical**
interrupted-compile lock below. No specific startup speedup is promised. Applying the hotfix does not remove
an existing lock.

### Possible stale Torch extension lock

**Symptom:** after stopping during compilation, a later start produces no further compiler output and never
becomes ready. **Cause — HYPOTHESIS, not observed in the boot walk:** an interrupted Torch build may leave its
file lock behind and a new process may wait on it. A lock's presence or age alone is not proof; active builds
also use locks. The recipe delegates locking/building to the installed Torch dependency
([engine/src/tensorfold/cuda/build.py:26–29](https://github.com/jakejharris/jspark3/blob/v2.0.1/engine/src/tensorfold/cuda/build.py#L26-L29));
Torch's implementation is not vendored in this tree, so this is a recovery hypothesis, not a demonstrated bug.

**Detect:** the host path to inspect is **`$DATA/kernel-cache/torch_extensions/<extension-name>/lock`**, not
`$DATA/kernel-cache/tensorfold`. `TORCH_EXTENSIONS_DIR=/kernel-cache/torch_extensions` is set in
[config/serve.env:61](https://github.com/jakejharris/jspark3/blob/v2.0.1/config/serve.env#L61)
and the host mapping is in [scripts/serve.sh:116](https://github.com/jakejharris/jspark3/blob/v2.0.1/scripts/serve.sh#L116).
For example, the expert extension name is `tensorfold_experts_v8`
([engine/src/tensorfold/cuda/experts.py:23](https://github.com/jakejharris/jspark3/blob/v2.0.1/engine/src/tensorfold/cuda/experts.py#L23));
use the actual path printed below. On the affected box:

```bash
(
set -euo pipefail
source scripts/lib.sh
load_cluster
docker logs --tail 100 "${CONTAINER_PREFIX}-rank${RANK}"
sudo find "$DATA/kernel-cache/torch_extensions" -type f -name lock -print
)
```

If the directory is absent or no lock is listed, this recovery does not apply. Preserve the container logs
before stopping: [scripts/stop.sh:46–48](https://github.com/jakejharris/jspark3/blob/v2.0.1/scripts/stop.sh#L46-L48)
removes the container. Run `scripts/stop.sh` on **all three boxes**, using `--container-prefix` for any CLI-only
prefix used at launch. Prevent automatic/concurrent restarts. Stop any other container and any host Python,
serving or build job using this cache; an empty compiler-process list alone is not enough. Include other
Docker daemons when checking. If you cannot establish that no process uses it, leave the locks alone.

**Fix:** only with **no container running** and no host cache user, on the affected box run the following for
each suspected leftover. It prompts for the exact listed path, confines it to this cache, and removes it from
the active `lock` name by renaming it to a backup. It does not delete compiled files or the cache.

```bash
(
set -euo pipefail
source scripts/lib.sh
load_cluster
running=$(docker ps -q)
test -z "$running" || { echo 'Stop all containers on this box first.' >&2; exit 1; }
cache=$(realpath -e -- "$DATA/kernel-cache/torch_extensions")
sudo find "$cache" -type f -name lock -print
read -r -p 'Exact lock path from the list above: ' lock_file
test ! -L "$lock_file"
lock_file=$(realpath -e -- "$lock_file")
[[ "$lock_file" == "$cache"/*/lock && -f "$lock_file" ]]
[[ ! -e "$lock_file.v201-backup" && ! -L "$lock_file.v201-backup" ]]
sudo mv -T -- "$lock_file" "$lock_file.v201-backup"
test ! -e "$lock_file"
printf 'Quarantined lock: %s\n' "$lock_file.v201-backup"
)
```

**Verify:** start ranks 2, 1, then 0 with the same settings. Require advancing logs, rank 0 readiness and
`scripts/smoke.sh` exit 0 / `SMOKE PASS`
([scripts/serve.sh:2–11](https://github.com/jakejharris/jspark3/blob/v2.0.1/scripts/serve.sh#L2-L11),
[scripts/wait-ready.sh:71–98](https://github.com/jakejharris/jspark3/blob/v2.0.1/scripts/wait-ready.sh#L71-L98),
[scripts/smoke.sh:2–7](https://github.com/jakejharris/jspark3/blob/v2.0.1/scripts/smoke.sh#L2-L7)).
Readiness timeout does not run a stop or lock cleanup
([scripts/wait-ready.sh:84–96](https://github.com/jakejharris/jspark3/blob/v2.0.1/scripts/wait-ready.sh#L84-L96));
waiting longer alone cannot remove a leftover. If the stall returns, keep the logs and investigate the first
error instead of repeatedly clearing locks.

**Undo:** restoring a stale lock may reproduce the stall, so normally retain the backup only for diagnosis.
To restore the exact pre-recovery state, stop all three ranks and all other cache users as above, then run on
the affected box. It refuses to overwrite a new lock:

```bash
(
set -euo pipefail
source scripts/lib.sh
load_cluster
running=$(docker ps -q)
test -z "$running" || { echo 'Stop all containers first.' >&2; exit 1; }
cache=$(realpath -e -- "$DATA/kernel-cache/torch_extensions")
read -r -p 'Exact quarantined lock path (ending in lock.v201-backup): ' backup
test ! -L "$backup"
backup=$(realpath -e -- "$backup")
[[ "$backup" == "$cache"/*/lock.v201-backup && -f "$backup" ]]
lock_file=${backup%.v201-backup}
[[ ! -e "$lock_file" && ! -L "$lock_file" ]]
sudo mv -T -- "$backup" "$lock_file"
test -f "$lock_file"
)
```

### Dead or unavailable upstream weight source

**Symptom:** a pinned download fails, or the source checker prints `FAIL`. **Cause:** a repository/revision may
be removed, made private, renamed or gated; a network failure is another possibility and does not prove deletion.
v2.0.1 pins distinct base, drafter and gated ablit sources at
[pins.env:26–46](https://github.com/jakejharris/jspark3/blob/v2.0.1/pins.env#L26-L46),
and downloads exact revisions at
[scripts/fetch-weights.sh:102–118](https://github.com/jakejharris/jspark3/blob/v2.0.1/scripts/fetch-weights.sh#L102-L118).
The unavailable Mia-AiLab EXL3 source concerns **v1.8.4 rollback**, not v2.0.1's MLX base weights; use the
[pinned rollback mirror instructions](../UPGRADING.md#going-back-to-v184). v2.0.1 requires a separate installation
([UPGRADING.md:34–36](https://github.com/jakejharris/jspark3/blob/v2.0.1/UPGRADING.md#L34-L36)).

**Detect / fix:** from the v2.0.1 checkout, with the sibling docs checkout from INSTALL:

```bash
../jspark3-install-docs/scripts/check-sources.sh "$PWD/pins.env"
```

The checker was added after the tag; its behavior is defined by
[scripts/check-sources.sh:87–163](../scripts/check-sources.sh#L87-L163), not by a v2.0.1 script.
Expect `check-sources: 4 passed, 0 failed` and exit 0. It makes anonymous metadata/HEAD checks; it does not
verify every shard. The gated ablit **401 / GatedRepo** weight response is expected only when its pinned
revision's metadata also resolves. If it fails, retain the output and investigate network access or the named
source before downloading. For ablit account access, follow the tag's token/terms diagnostics at
[scripts/fetch-weights.sh:144–157](https://github.com/jakejharris/jspark3/blob/v2.0.1/scripts/fetch-weights.sh#L144-L157).

If a source is gone, preserve any local files and their manifests. Recheck existing downloads with
`scripts/fetch-weights.sh --verify-only` using your configured weights/drafter. If the full source snapshot was
already removed after splitting, recheck the retained third with `scripts/split.sh --verify-only` instead;
when using DFlash2, also run `scripts/fetch-weights.sh --drafter-only --verify-only`. These paths rehash local
files and refresh their verification markers without fetching
([scripts/fetch-weights.sh:39–58](https://github.com/jakejharris/jspark3/blob/v2.0.1/scripts/fetch-weights.sh#L39-L58),
[scripts/fetch-weights.sh:153–192](https://github.com/jakejharris/jspark3/blob/v2.0.1/scripts/fetch-weights.sh#L153-L192),
[scripts/split.sh:79–88](https://github.com/jakejharris/jspark3/blob/v2.0.1/scripts/split.sh#L79-L88)).
If you need replacement bytes, use a maintainer-confirmed mirror at an immutable revision, stage its files
separately, and require agreement with the **original tag's manifests** before replacing anything. Keep the
old files for rollback. No replacement v2.0.1 mirror is prescribed here: without matching bytes, stop and
report the failing repository and pinned revision. Do not substitute `main`, change hashes or bypass verification.
If only DFlash2 is unavailable, the [no-drafter path](../INSTALL.md#running-without-the-draft-model) avoids its
download ([scripts/fetch-weights.sh:186–192](https://github.com/jakejharris/jspark3/blob/v2.0.1/scripts/fetch-weights.sh#L186-L192)).

**Verify:** the applicable local verifier must report every file matches / the third is verified; then use
preflight, readiness and smoke before serving clients. The checker may still fail upstream even when your
local files verify. **Undo:** the checker changes nothing; local verification only refreshes `.verified`
markers as cited above. No pin/config edit is required. If you replaced files from a mirror, stop all ranks,
restore your saved originals and rerun the same verifier; never restore a success marker without checking bytes.

### No-drafter preflight flag

**Symptom:** preflight reports `the drafter is downloaded and verified` as a failure although you intend to
serve without DFlash2. **Cause:** a `--drafter none` option on another command does not persist into preflight;
the latter reads its own options/settings
([scripts/preflight.py:225–238](https://github.com/jakejharris/jspark3/blob/v2.0.1/scripts/preflight.py#L225-L238),
[scripts/lib.sh:128–131](https://github.com/jakejharris/jspark3/blob/v2.0.1/scripts/lib.sh#L128-L131)).
It checks drafter files only for `dflash2`
([scripts/preflight.py:123–125](https://github.com/jakejharris/jspark3/blob/v2.0.1/scripts/preflight.py#L123-L125)).

**Fix:** before starting, on **each box** run the corrected command:

```bash
python3 scripts/preflight.py --for serve --drafter none
```

**Verify:** require exit 0 / `0 failed`, with no drafter-file check; address unrelated failures normally
([scripts/preflight.py:239–241](https://github.com/jakejharris/jspark3/blob/v2.0.1/scripts/preflight.py#L239-L241)).
Then start ranks 2, 1, 0 with `scripts/serve.sh --drafter none` on every box. This selects the model's own
prediction head rather than DFlash2
([scripts/serve.sh:10–11](https://github.com/jakejharris/jspark3/blob/v2.0.1/scripts/serve.sh#L10-L11),
[scripts/serve.sh:85–90](https://github.com/jakejharris/jspark3/blob/v2.0.1/scripts/serve.sh#L85-L90)).
**Undo:** this CLI flag changes no saved setting. To return to DFlash2, stop all ranks, download/verify it with
`scripts/fetch-weights.sh --drafter dflash2 --drafter-only`, run
`python3 scripts/preflight.py --for serve --drafter dflash2`, then start every rank with
`scripts/serve.sh --drafter dflash2`. Explicit flags override `cluster.env`
([scripts/lib.sh:129–131](https://github.com/jakejharris/jspark3/blob/v2.0.1/scripts/lib.sh#L129-L131));
drafter download and verification are at
[scripts/fetch-weights.sh:167–192](https://github.com/jakejharris/jspark3/blob/v2.0.1/scripts/fetch-weights.sh#L167-L192).

## How script messages look

- Every message starts with the script's name, for example `serve.sh: ...`. Warnings say `warning:` after the name.
- The recipe's shell scripts exit with code 2 on an error. If one had opened its log, the next line is `details: logs/<script>.log`. The separate source checker exits 1 when a source check fails, or 2 for invalid arguments or pins.
- On screen, messages name settings and show paths relative to `$DATA`. They never show the values of your settings, addresses or user names. The log in `logs/` has the full detail and is readable only by you.
- In the tables below, `<...>` stands for a part that changes: a rank, a size, a file name.
- Container names appear as `jspark3-rank<R>`. If you set `CONTAINER_PREFIX` in `cluster.env`, your prefix replaces `jspark3`.
- Every script takes `--help`. Most recipe scripts take `--dry-run`, which prints what would happen and changes nothing. The source checker only reads upstream metadata and takes no `--dry-run` option.

If you don't know where to start, run these on every box and read the first FAIL or error:

```sh
python3 scripts/preflight.py --for serve
scripts/status.sh
```

## Preflight

`python3 scripts/preflight.py` checks one box before a step. Run it on every box.

```sh
python3 scripts/preflight.py --for fetch    # before downloading
python3 scripts/preflight.py --for split    # before splitting
python3 scripts/preflight.py --for serve    # before starting (the default); checks everything
```

It takes the same `--weights`, `--drafter` and `--session-tier` options as the other scripts, so it checks the configuration you are about to use.

Each line reads `PASS  <check>`, or `WARN  <check>: <what to do>`, or `FAIL  <check>: <what to do>`. The last line counts them: `preflight --for <step>: <n> passed, <n> warnings, <n> failed`.

- **Exit 0:** nothing failed. Warnings don't stop anything.
- **Exit 1:** at least one FAIL. Fix every FAIL before the step.
- **Exit 2:** preflight could not read your settings. The message above it comes from the settings check; see [Settings and options](#settings-and-options).

The output never contains your settings' values, so you can share it as it is.

**Every step:**

| Check | Means | What to do |
|---|---|---|
| `python3 is 3.10 or newer` | FAIL reads `this python3 is <version>; ...`. The scripts and the pinned Hugging Face CLI need Python 3.10 or newer. | Install Python 3.10 or newer (INSTALL, "What you need"), then run preflight with it. |
| `docker runs for this user` | `docker info` failed for your user. | Install Docker and add your user to the `docker` group (INSTALL, "What you need"). Log out and in again after adding the group. |
| `DATA exists and is writable` | The `DATA` folder in `cluster.env` is missing, or your user can't write to it. | Create it on the box's local NVMe disk and make your user its owner. |

**Fetch:**

| Check | Means | What to do |
|---|---|---|
| `the Hugging Face CLI ('hf') is the pinned version 2.1.1` | FAIL ends `not installed: install it in a virtual environment: ...` when `hf` is missing, or `it is <version>: install it in a virtual environment: ...` when its installed version differs or cannot be read; `unknown` means the `hf` launcher, its interpreter or the package metadata could not be read, so reinstall the pinned CLI in the documented virtual environment. This release pins the CLI (`HF_HUB_VERSION` in `pins.env`), and the download script refuses any other version. | See [the Hugging Face CLI](#the-hugging-face-cli). |
| `HF_TOKEN is set for the gated ablit source` | Only with `--weights ablit`. The source is gated and no token is set. | See [the gated ablit source](#the-gated-ablit-source). |
| `HF_HUB_DISABLE_XET is not 1` (WARN) | Your environment turns Hugging Face's Xet transfer on. The scripts download over plain HTTPS by default. | Unset it unless you need Xet. |

**Split and serve:**

| Check | Means | What to do |
|---|---|---|
| `the pinned image is here, ID as pinned` | The container image is missing, or its ID is not the one in `pins.env`. | `scripts/pull-image.sh` |
| `the engine wheel is the pinned build` | `wheels/` has no engine wheel, or its content is not the pinned build. | `scripts/build-wheel.sh` |
| `the dependency wheels are in wheels/` | A wheel listed in `wheels.lock` is missing. | `scripts/fetch-wheels.sh` |

**Serve only:**

| Check | Means | What to do |
|---|---|---|
| `the <weights> rank <R> third is split and verified` | This box's part of the weights is missing, was never verified, or failed or did not finish its last check. | `scripts/split.sh` (INSTALL). For a third that is already here, `scripts/split.sh --verify-only` checks it again; add `--weights ablit` for ablit. |
| `the drafter is downloaded and verified` | The draft model is missing or unverified, or its last check failed or did not finish, and this start uses it. | `scripts/fetch-weights.sh`. For a copy that is already here, `scripts/fetch-weights.sh --drafter-only --verify-only` checks it again. Or start with `--drafter none` on every box. |
| `the chat template matches pins.env` | `template/chat-template.jinja` was changed. | `git checkout template/` |
| `the GPU is visible (nvidia-smi)` | `nvidia-smi -L` did not list the GPU. | Install or repair the NVIDIA driver. Reboot after a driver update. |
| `a container sees the GPU (docker run --gpus all)` | A short test container from the pinned image could not list the GPU the way `serve.sh` asks for it. | Pull the image first (`scripts/pull-image.sh`). If the image is already here, install or repair the NVIDIA Container Toolkit and restart Docker. |
| `RDMA devices exist (/dev/infiniband)` | No RDMA devices. The ranks can't talk over the cables. | Install the ConnectX-7 (mlx5) drivers and RDMA core. |
| `at least two RDMA ports are ACTIVE` | Fewer than two ports have a live link. Each box needs both: one to the previous rank and one to the next. | Cable both ConnectX-7 ports in the ring (rank 0 to 1, 1 to 2, 2 to 0) and bring the links up. |
| `<LAN_IFACE, PREV_IFACE or NEXT_IFACE> names an interface on this box` | That name in `cluster.env` is not an interface here. Names often change after an OS or firmware update. | Fix `cluster.env`. `ip -br link` lists the interfaces. |
| `<interface setting> is up` | The interface exists but its link is down. | Check the cable and bring the link up. |
| `<PREV_IFACE or NEXT_IFACE> has MTU 9000` (WARN) | The ring link uses another MTU. It may work, but it is not what was measured. | Set MTU 9000 on both ends of each ring cable. |
| `<PREV_PEER or NEXT_PEER> is not set, so the cable to it is not tested` (WARN) | No peer address to test the cable with. | Set it in `cluster.env` to get the cable check. |
| `<PREV_PEER or NEXT_PEER> answers a 9000-byte ping over <interface>` | A full-size ping without fragmenting did not get through that cable. | Check the cable, both addresses and the MTU on both ends. A ping that works at small sizes but fails here means an MTU mismatch. |
| `containers get unlimited locked memory` | A test container could not get unlimited locked memory, which RDMA needs. | Allow Docker's `--ulimit memlock=-1`. A stock Docker daemon allows it; look for a daemon or systemd limit that removed it. |
| `swap is off or unused` (WARN) | Swap is on and in use. | Optional: `scripts/host-prep.sh` turns swap off. |
| `API_PORT is free on this box` (rank 0) | Something already listens on the API port. | Stop it. Usually it is an earlier server: `scripts/stop.sh`, or a v1.8.x server (see OPERATIONS, upgrading). |
| `MASTER_PORT is free on this box` (rank 0) | Something already listens on the start-up port. | As above. |
| `the API listens on loopback only` (WARN, rank 0) | `API_HOST` is not a loopback address. The API has no authentication. | Set it back to `127.0.0.1` and reach the API as described in [Reaching the server safely](#reaching-the-server-safely). |
| `MASTER_ADDR is reachable over the LAN` (WARN, ranks 1 and 2) | This box could not ping rank 0's LAN address. | Rank 0 must be reachable at `MASTER_ADDR` from every box. Check the address and the LAN. Some networks block ping, so a WARN alone is not proof. |
| `the container name jspark3-rank<R> is free` | A container already has this rank's name. The FAIL detail says whose. | `a container from an earlier start has it`: run `scripts/stop.sh`. `a container this recipe did not start has that name`: the scripts leave it alone; set `CONTAINER_PREFIX` in `cluster.env` to another name, the same on all three boxes. |
| `no other container is running on this box` (WARN) | Other containers are running here; the WARN gives how many. | Stop any that use the GPU, its memory or the ports (another release, a benchmark). |
| `no other process is using the GPU` (WARN) | Another program is using the GPU, or `nvidia-smi` could not list GPU programs. | Stop it before serving: the server sizes itself from the memory that is free when it starts. |
| `vm.compaction_proactiveness is 0` (WARN) | The setting is not 0, or could not be read. Background memory compaction can stall serving after a build or split on the same box. | Run: `sudo sysctl -w vm.compaction_proactiveness=0` (or `scripts/host-prep.sh`); a reboot resets it. |
| `DATA has room for the session tier (<n> GiB, with 150 GiB left free)` (WARN) | With the session store on, `DATA` has less free space than the store's budget plus the 150 GiB it always leaves free. | Free space on `DATA`, or start with `--session-tier off`. Otherwise long conversations may not be kept on disk; answers are not affected. |

## Settings and options

These come from the settings check that every script runs first.

| Message | Means | What to do |
|---|---|---|
| `no cluster.env: cp cluster.env.example cluster.env and fill it in (INSTALL.md)` | This box has no `cluster.env`. | Do what it says, on every box. |
| `WEIGHTS must be base or ablit` | `WEIGHTS` in `cluster.env` has another value. | Fix it. |
| `DRAFTER must be dflash2 or none` | Same for `DRAFTER`. | Fix it. |
| `SESSION_TIER must be disk or off` | Same for `SESSION_TIER`. | Fix it. |
| `RANK must be 0, 1 or 2` | `RANK` is missing or wrong. | Each box has its own rank: 0, 1 or 2. |
| `DATA must be an absolute path` | `DATA` is empty or relative. | Use a full path starting with `/`. |
| `<setting> in cluster.env is empty or has characters the scripts don't accept` | A required setting is empty, or has spaces, quotes or other characters. | Use letters, digits and `_ . / : -` only. |
| `<setting> in cluster.env has characters the scripts don't accept` | Same for an optional setting. | As above, or leave it empty. |
| `--weights needs base or ablit`, `--drafter needs dflash2 or none`, `--session-tier needs disk or off` | The option has no value. | Add the value. |
| `--container-prefix needs a name` | The option has no value. | Add the name. |
| `CONTAINER_PREFIX must be lower-case letters, digits and _.- (at most 41)` | `CONTAINER_PREFIX` in `cluster.env`, or `--container-prefix`, has another value. | Use lower-case letters, digits, `_`, `.` and `-`, starting with a letter or digit. |
| `--weights ablit is not available in this tree (no manifests/ablit); use base` | The normal availability check can't find `manifests/ablit/rank0.sha256`. Normal operations and dry runs refuse this incomplete copy. A real `split.sh --verify-only` checks the selected rank's manifest instead; if that manifest is missing, it reports the missing-manifest message under [Splitting and converting](#splitting-and-converting). | Use base, or a complete release copy. |
| `the ablit download and conversion are not pinned in this tree yet. A verified ablit third you already have can be checked with scripts/split.sh --weights ablit --verify-only` | From `fetch-weights.sh`, `convert-ablit.sh` or `split.sh` with `--weights ablit`: this copy can't download or build ablit. | Use a complete release copy. A third you already have can still be checked, as the message says. |
| `unknown option <option> (--help)` | A misspelled option. | Run the script with `--help`. |

### Settings profile

`serve.sh` reads the [settings profile](https://github.com/jakejharris/jspark3/blob/v2.0.1/docs/OPERATIONS.md#settings-profiles) for your weights, `config/profiles/<weights>.env`, before any container starts. These messages mean the profile is missing, or the profile or `config/serve.env` was edited.

| Message | Means | What to do |
|---|---|---|
| `config/profiles/<weights>.env is missing from this tree; restore it: git checkout config/profiles/` | The profile for the weights you chose is not in this copy of the recipe. | Do what it says, then start again. |
| `config/profiles/<weights>.env does not set <key>; restore it: git checkout config/profiles/` | One of the profile's two drafting settings was deleted. | Do what it says, then start again. |
| `config/profiles/<weights>.env line <n> is not KEY=VALUE ...`, `... sets <key> twice`, `... sets <key>: a profile holds ...`, `config/profiles/<weights>.env: <key> must be ...`, `... the <key> confidence must be from 0.1 to 0.9` | The profile was edited into a form `serve.sh` does not accept. Nothing was started. | `git checkout config/profiles/`, then start again. |
| `config/serve.env: TF_GLM_REPLY_PREFILL=1 needs TF_GLM_FAIR_SCHED=1 (the engine refuses to start otherwise)` | `config/serve.env` was edited to turn on a setting without the one it needs. Nothing was started. | `git checkout config/serve.env`, then start again. |
| `config/profiles/<weights>.env differs from the file this release shipped (SHA256SUMS): this is not a measured configuration` | A warning. The start goes on with your edited profile, and the published numbers do not apply to it. | If you did not mean to change it: stop all three, `git checkout config/profiles/` on every box, and start again. |

## Downloads

Before a fresh install, run the [source check](../INSTALL.md#quick-start-base-weights-the-default). It checks the
pinned upstream revisions without downloading weights or using your credentials. A failure may mean a source
disappeared, became private, changed its access conditions, or could not be reached. A 401 is expected only for
the gated ablit weight-file HEAD, alongside accessible metadata for its pinned revision. Do not change pins to
make a failed check pass; investigate the source or network first.

### The Hugging Face CLI

The download script uses the Hugging Face CLI (`hf`) at the version this release pins, `HF_HUB_VERSION` in `pins.env` (2.1.1). Install it in a virtual environment, which leaves the system Python alone; DGX OS refuses a plain `pip install`. It needs Python 3.10 or newer on the box.

| Message | Means | What to do |
|---|---|---|
| `no 'hf' command: install the pinned Hugging Face CLI in a virtual environment: python3 -m venv ~/hf-cli && ~/hf-cli/bin/pip install "huggingface_hub==2.1.1", then export PATH="$HOME/hf-cli/bin:$PATH" (INSTALL.md, 'What you need')` | `hf` is not on your path. | Run the commands it shows, in the shell you install from. If `python3 -m venv` says ensurepip is not available, install your distribution's venv package first (`python3-venv` on Ubuntu). If you installed it earlier from another shell, only the `export PATH=...` part is missing. Then check with `hf version`. |
| `the Hugging Face CLI is version <version>; this release pins HF_HUB_VERSION=2.1.1. Install it in a virtual environment: python3 -m venv ~/hf-cli && ~/hf-cli/bin/pip install "huggingface_hub==2.1.1", then export PATH="$HOME/hf-cli/bin:$PATH" (INSTALL.md, 'What you need')` | The first `hf` on your path has a different installed version, or its launcher, interpreter or package metadata could not be read (`unknown`). The script refuses to download with it. | Reinstall the pinned CLI in the documented virtual environment as above. If `hf version` still shows another version, an older `hf` comes first on your `PATH`: run `which -a hf` and remove or reorder it. |

Only the gated ablit source uses `HF_TOKEN`. The base weights and the draft model download without a token, even if one is set. With ablit selected, the script checks the token and your access before it downloads anything, the base weights and the draft model included (also with `--drafter-only`).

To download through a Hugging Face mirror, set `HF_ENDPOINT` to its address. The download script and `hf` both use it. The scripts send your token only to Hugging Face: to the endpoint you use (`https://huggingface.co` by default, or the mirror you set in `HF_ENDPOINT`), and to Hugging Face's own hosts when the official huggingface_hub client follows its redirects. Set `HF_ENDPOINT` only to a mirror you trust with your token. The script's own requests (the size lookup and the access check) never send the token on a redirect to another site, and they refuse a redirect from HTTPS to plain HTTP. An inherited Hugging Face staging setting (`HUGGINGFACE_CO_STAGING`) can't redirect the download: the scripts leave it out of every download and use `https://huggingface.co`, or the mirror you set in `HF_ENDPOINT`.

### The gated ablit source

The ablit weights are built from a gated Hugging Face repository. Before `scripts/fetch-weights.sh --weights ablit` can download it:

1. Sign in to Hugging Face with your own account.
2. Open the source's page and accept its terms yourself.
3. Create a read token in your Hugging Face settings.
4. Give the token to the script without leaving it in your shell history:

   ```sh
   read -rs HF_TOKEN && export HF_TOKEN    # paste the token; nothing is shown
   scripts/fetch-weights.sh --weights ablit
   unset HF_TOKEN
   ```

   Don't put the token on a command line, in `cluster.env` or in a file you share. The scripts never print it.

The script prints the ablit notice, then checks the token and your access before any download. A missing, rejected or unaccepted token stops it before anything is downloaded or changed in `DATA`. If the access check gets no answer, the script warns and tries the download anyway. `--verify-only` needs no token.

| Message | Means | What to do |
|---|---|---|
| `the refusal-removed source is gated: accept its terms on its Hugging Face page with your account, then export HF_TOKEN=<your token> and run this again` | No token is set. | Steps 1 to 4 above. |
| `Hugging Face did not accept HF_TOKEN: create a read token in your Hugging Face settings, export HF_TOKEN=<your token>, and run this again` | Hugging Face answered 401: the token is wrong, revoked or expired. | Create a new read token and set it again. |
| `your Hugging Face account has not accepted the source's terms yet: open https://huggingface.co/<source>, accept them, and run this again` | Hugging Face answered 403. | Accept the terms on that page while signed in as the token's owner. If you already did, your token may be a fine-grained token without access to gated repositories: use a read token, or give the fine-grained token read access to public gated repositories. |
| `Hugging Face answered HTTP <code> for the gated source; check https://huggingface.co/<source> and run this again` | Any other answer. | Open the page in a browser to see whether the repository or your account has a problem, then run again. |
| `warning: could not reach Hugging Face to check access; trying the download anyway` | The access check got no answer. The download is tried anyway. | If the download then fails, check this box's internet access. |
| `your Hugging Face account has access to the gated source` | The access check passed. | Nothing. |

The ablit terms and the responsibility line are in [OPERATIONS.md](https://github.com/jakejharris/jspark3/blob/v2.0.1/docs/OPERATIONS.md#weights-base-or-ablit) and INSTALL.

### An interrupted download

| Message | Means | What to do |
|---|---|---|
| `<what>: download failed; run this again (it resumes)` | The download stopped: a dropped connection, a full disk, or Ctrl-C. | Run the same command again. Files already complete are kept and partial files resume. |
| `<what>: download failed. If the log shows 401 or 403, accept the source's terms on its Hugging Face page and export HF_TOKEN; otherwise run this again (it resumes)` | Same, for the gated source. | Check `logs/fetch-weights.log` for 401 or 403 and see [the gated ablit source](#the-gated-ablit-source). Otherwise run again. |

If a box restarts during a download, run the command again after the restart. After every download the script checks every file, so an interrupted download can't pass as a complete one.

### A file that does not match

Every downloaded file is checked against a pinned SHA-256 list, and every split part against a shipped manifest. A mismatch means the file on your disk is not the file this release was tested with.

A check first removes the folder's `.verified` marker and writes it again only when every file matches. A check that fails or is stopped leaves the folder unmarked, and the steps that need it refuse it until a check passes. If a check was stopped before it finished, run the same command with `--verify-only` to check again.

| Message | Means | What to do |
|---|---|---|
| `<what> does not match <list>. Files that differ or are missing: <files>. Delete them and run this again to re-download.` | Those files are damaged, incomplete or from another revision. The folder is no longer marked verified. | Delete the files it names and run the same command again. |
| `<what>: <folder> does not exist (run without --verify-only to download it)` | `--verify-only` found nothing to check. | Run without `--verify-only`. |
| `<what>: every file matches` | All files are as pinned, and the folder is marked verified. | Nothing. |

If the same file fails again after a fresh download, check the disk: `sudo dmesg | grep -i -E 'nvme|error'`. Don't edit the SHA-256 lists or the manifests to make a check pass. The tested configuration is what they describe.

### Not enough disk space

| Message | Means | What to do |
|---|---|---|
| `<what> needs <n> GB more but the DATA disk has <n> GB free` | The download would not fit. | Free space on `DATA`. INSTALL's "Disk, time and memory per step" lists what each step needs. |
| `warning: <what>: could not read sizes from Hugging Face; skipping the space check` | The download goes ahead without a space check. | Make sure `DATA` has room. A full disk shows up as a failed download. |

## Image, engine wheel and dependency wheels

| Message | Means | What to do |
|---|---|---|
| `docker could not pull the image; check this box's internet access and docker login state` | `docker pull` failed. | Check internet access to NVIDIA's registry, and any `docker login` this box needs. |
| `the pulled image's ID is not IMAGE_ID in pins.env; do not use it (docker image rm it and pull again)` | The image you got is not the pinned one. | Remove it with `docker image rm` and run `scripts/pull-image.sh` again. |
| `warning: the pinned image is not pulled (scripts/pull-image.sh)` or `warning: the local image ID is not the pinned IMAGE_ID (scripts/pull-image.sh)` | A later step found no pinned image. | `scripts/pull-image.sh` |
| `pull the pinned image first: scripts/pull-image.sh` | Same, as an error. | As above. |
| `the wheels in wheels/ do not match wheels.lock (names above); delete those files and run this again` | Some dependency wheels are damaged or are not the pinned versions. | Delete the named files from `wheels/` and run `scripts/fetch-wheels.sh` again. |
| `this tree has no engine/ source; use a complete release archive` | The `engine/` folder is missing. | Use a complete copy of the release. Check it with `python3 tools/payload.py verify`. |
| `the wheel build failed` or `the build did not produce wheels/<wheel>` | Building the engine wheel failed. | Read `logs/build-wheel.log`. Check the tree with `python3 tools/payload.py verify`, then run again. |
| `the built wheel's content is not WHEEL_CONTENT_SHA256 in pins.env: do not serve it. Check engine/ against SHA256SUMS (python3 tools/payload.py verify)` | The engine you built is not the tested engine. | Don't serve it. Run `python3 tools/payload.py verify`, restore any file it names, and build again. |
| `pins.env WHEEL_CONTENT_SHA256 is PENDING: this tree has no pinned engine build to compare with` | This copy has no pinned engine build. | Use a release archive. |
| `warning: no engine wheel in wheels/ (scripts/build-wheel.sh)` or `warning: wheels/<wheel> is not the pinned engine build (scripts/build-wheel.sh)` | A later step found no pinned engine wheel. | `scripts/build-wheel.sh` |
| `build the engine wheel first: scripts/build-wheel.sh` | Same, as an error. | As above, and `scripts/fetch-wheels.sh` if the dependency wheels are missing. |

## Splitting and converting

**`scripts/split.sh`** writes this box's third of the weights and checks every file. With `--verify-only` it reads every file of the third that is already here, even one that passed before.

| Message | Means | What to do |
|---|---|---|
| `the <weights> snapshot is not downloaded and verified yet (<command>)` | The full weights are not ready. | Run the command it names: `scripts/fetch-weights.sh`, and for ablit also `scripts/convert-ablit.sh`. |
| `the <what> needs about <n> GB but the DATA disk has <n> GB free` | Not enough room for the split. Nothing was deleted; any earlier output is as it was. | Free space on `DATA`. |
| `<what>: the split failed` | The split stopped with an error. | Read `logs/split.log`, then run again. |
| `<what> does not match its manifest (files listed above). It is kept at <folder>.partial for inspection; nothing was marked verified` | The new split differs from the shipped manifest. | Delete the `.partial` folder and run `scripts/split.sh` again. If it fails the same way, check the full weights with `scripts/fetch-weights.sh --verify-only`. |
| `<what> does not match its manifest (files listed above). Split it again: scripts/split.sh --rank <R>` | From `--verify-only`: the third was changed or damaged after it was split or copied. It is no longer marked verified, so `serve.sh` refuses it. | Do what it says on a box that has the full weights (add `--weights ablit` for ablit). For a copied third, copy it again and run the same `--verify-only` command. |
| `this tree has no manifest for the <what> (manifests/<weights>/rank<R>.sha256), so it cannot be verified` | The manifest is missing from this copy. With `--verify-only`, the third's files are kept but it is no longer marked verified. | Use a complete release copy, then run the same `--verify-only` command again. Restoring the manifest does not mark the third verified by itself. |
| `<what>: nothing to verify at <folder>` | Nothing to check yet. | Run `scripts/split.sh` without `--verify-only`. |
| `--rank needs 0, 1 or 2` | `--rank` has no valid value. | Fix the option. |

`--all` splits all three parts on one box, for copying to the others. Each part is checked against its manifest in the same way.

The file check lists each problem on its own line, then a count:

- `MISSING <file>`: the file should be there and isn't.
- `EXTRA <file> (<n> bytes)`: a file that should not be there.
- `DIFFERENT <file> (<n> bytes): sha256 <got>, expected <want>`: the file's content is wrong.

**`scripts/convert-ablit.sh`** builds the ablit weights and all three thirds from the gated source and the base weights, inside the pinned image with no network, on this box's CPU and disk.

| Message | Means | What to do |
|---|---|---|
| `the ablit download and conversion are not pinned in this tree yet. ...` | This copy can't build ablit. | See [Settings and options](#settings-and-options). |
| `scripts/ablit/convert-weights.py does not match ABLIT_CONVERTER_SHA256 in pins.env; restore it: git checkout scripts/ablit/` | The converter was changed. | Do what it says. |
| `the base snapshot is not downloaded and verified: scripts/fetch-weights.sh --weights ablit` | The conversion needs the base weights too. | Do what it says. |
| `the refusal-removed source is not downloaded and verified: scripts/fetch-weights.sh --weights ablit` | The gated source is missing. | Do what it says. See [the gated ablit source](#the-gated-ablit-source). |
| `pull the pinned image first: scripts/pull-image.sh` | The conversion runs inside the pinned image, which is not on this box. | Do what it says. |
| `build the engine wheel first: scripts/build-wheel.sh` | The conversion uses the engine build, which is not on this box. | Do what it says. |
| `the conversion needs about <n> GB but the DATA disk has <n> GB free` | The converted weights and the three thirds take a little under twice the source's size. Nothing was deleted: earlier outputs, their verified marks and `$DATA/ablit/work` are as they were. | Free space on `DATA`. [OPERATIONS.md](https://github.com/jakejharris/jspark3/blob/v2.0.1/docs/OPERATIONS.md#weights-base-or-ablit) has the measured cost. |
| `ablit weights and all three thirds already built and verified` | Nothing to do. | Copy rank 1 and rank 2 to their boxes, if you have not. |
| `the conversion failed (logs/convert-ablit.log); nothing was marked verified` | The converter stopped with an error. | Read `logs/convert-ablit.log`. Check free disk and memory, then run again. |
| `the converted weights do not match manifests/inputs/ablit-weights.sha256 (files listed above); kept in $DATA/ablit/work, nothing marked verified` | The output is not the tested ablit weights. | Check the inputs with `scripts/fetch-weights.sh --weights ablit --verify-only`, then run again. The next run clears `$DATA/ablit/work` once its space check passes. |
| `the conversion id is <id>, not ABLIT_CONVERSION_ID in pins.env; kept in $DATA/ablit/work, nothing marked verified` | The weights match, but the conversion's identity record does not (`missing` if it wrote none). | Check that `pins.env` and `scripts/ablit/` are unchanged (`git status`), then run again. If it repeats, keep `$DATA/ablit/work` for your report ([asking for help](#asking-for-help)). |
| `the rank <R> third does not match manifests/ablit/rank<R>.sha256 (files listed above); kept in $DATA/ablit/work, nothing marked verified` | A converted third is not the tested one. | As for the converted weights above. |
| `ablit weights and all three thirds built and verified (<size> in $DATA/ablit)`, then `next: copy rank 1 and rank 2 to their boxes (INSTALL.md), or run scripts/split.sh --weights ablit --verify-only on each` | Done. | Copy `$DATA/ablit/rank1` and `$DATA/ablit/rank2` to those boxes and check each there with `scripts/split.sh --weights ablit --verify-only`. |
| `the ablit weights do not match their manifest (files listed above); run scripts/convert-ablit.sh again` | From `--verify-only`: the converted weights were changed, damaged or removed later. They are no longer marked verified. | Do what it says. |

## Starting

**`scripts/serve.sh`** starts this box's rank. Start rank 2, then rank 1, then rank 0.

| Message | Means | What to do |
|---|---|---|
| `docker is not installed (INSTALL.md, 'What you need')` | No `docker` command. | Install Docker. |
| `jspark3-rank<R> is already running on this box; stop it first: scripts/stop.sh` | This rank is already up. | Stop it, or leave it running. |
| `jspark3-rank<R> from an earlier start still exists (<state>); remove it with scripts/stop.sh, then start again` | A stopped container from an earlier start still has the name, for example one kept with `stop.sh --keep`. `serve.sh` never reuses or removes it. | Run `scripts/stop.sh`, which removes it and its log (save the log first if you need it). Or start under another prefix, as [INSTALL.md](../INSTALL.md#stopping-restarting-upgrading) describes. |
| `a container named jspark3-rank<R> exists on this box and was not started by this recipe; it is left alone. Set CONTAINER_PREFIX in cluster.env to another name, the same on all three boxes` | Something else has a container with this rank's name. The scripts never stop or remove a container they did not start. | Do what it says. |
| `pull the pinned image first: scripts/pull-image.sh` | No pinned image. | Do what it says. |
| `build the engine wheel first: scripts/build-wheel.sh (and scripts/fetch-wheels.sh)` | No pinned engine wheel. | Do what it says. |
| `the <weights> rank <R> third is not split and verified on this box: scripts/split.sh` | This box's part of the weights is missing, was never verified, or failed or did not finish its last check. | Run the command it names, with the same `--weights` option you start with. For a third that is already here, add `--verify-only` to check it again. |
| `the drafter is not downloaded and verified on this box: scripts/fetch-weights.sh (or serve with --drafter none on every box)` | The draft model is missing or unverified, or its last check failed or did not finish. | Fetch it, or check a copy you have with `scripts/fetch-weights.sh --drafter-only --verify-only`, or start every box with `--drafter none`. |
| `template/chat-template.jinja does not match TEMPLATE_SHA256 in pins.env; restore it: git checkout template/` | The chat template was changed. | Do what it says. |
| `warning: the DATA disk has <n> GB free; the session tier stops writing below 150 GiB free and uses up to <n> GiB, so conversations may not be kept on disk` | The session store may run out of room. The server still starts and answers are not affected. | Free space on `DATA`, or start with `--session-tier off`. See [OPERATIONS.md](https://github.com/jakejharris/jspark3/blob/v2.0.1/docs/OPERATIONS.md#disk-full). |
| `an engine flag has characters serve.sh doesn't pass through: use letters, digits and _./:=,+-` | An extra engine flag after `--` has a character the script refuses. | Remove it. Flags other than the shipped ones are not a measured configuration. |
| `--timeout needs a number of seconds` | `--timeout` has no number. | Fix the option. |
| `docker could not start jspark3-rank<R>` | `docker run` failed. | Read Docker's error in `logs/serve.log`, and run `python3 scripts/preflight.py --for serve`. A container that already has the name is not the cause: `serve.sh` stops before `docker run` with its own message for that. |
| `rank <R> started. Start the other follower, then rank 0, which waits until all three answer` | Ranks 1 and 2 are up and waiting. | Continue with the next rank. |
| `rank 0 started; check readiness with scripts/wait-ready.sh` | Rank 0 is up and loading. | Run `scripts/wait-ready.sh` on rank 0. |

## A rank never becomes ready

`scripts/wait-ready.sh` on rank 0 waits until the API lists the model and answers a one-token request. It waits up to 40 minutes by default and prints `still loading, <n> min` while it waits. On ranks 1 and 2 it only checks that the box's own container is running.

| Message | Means | What to do |
|---|---|---|
| `ready after <n> min: the API on port <port> answered through all three ranks` | Ready. | Run `scripts/smoke.sh`. |
| `jspark3-rank<R> is not on this box; start it with scripts/serve.sh` | No container for this rank. | Start it, in order: rank 2, rank 1, rank 0. |
| `jspark3-rank<R> stopped (exit code <n>). Its last output is in the log. Stop all three ranks (scripts/stop.sh on each box), then start ranks 2 and 1, then 0` | The rank's engine exited. Its last 300 log lines are in `logs/wait-ready.log`. | Read the first error there, fix it, then do what the message says. |
| `the three ranks were started with different settings; rank(s) <R> differ from rank 0. WEIGHTS, DRAFTER and SESSION_TIER must be the same in cluster.env on every box (scripts/status.sh shows each box's). Stop all three, fix cluster.env, start again` | The engine compares every rank's settings at start and refuses a mix. | Do what it says. A one-off `--drafter` or `--session-tier` on one box only causes this too, and so does a `config/serve.env` that differs between boxes: make it the same on all three. |
| `not ready after <n> min. On each box, scripts/status.sh shows its container; read 'docker logs jspark3-rank<R>' there for the first error. Then stop all three and start again (followers first), or wait longer with scripts/wait-ready.sh --timeout` | Nothing failed outright, but the API never answered. | See below. |

**When the ranks never connect**, go through these on every box, in order:

1. `scripts/status.sh`. Are all three containers running, with the same release, weights, draft model and session tier?
2. `docker logs jspark3-rank<R> 2>&1 | grep -m1 '\[fabric\]'` shows the ring settings that rank used, such as its RDMA devices and network plugin. Include it in a report.
3. `docker logs jspark3-rank<R> 2>&1 | grep -i -m5 -E 'error|fail|timeout'` finds the first errors. The earliest error on any rank is usually the cause; later errors on the other ranks often follow from it.
   - Each start installs the engine from `wheels/` inside the container. An install error at the top of the log means a missing or damaged wheel: run `scripts/fetch-wheels.sh` and `scripts/build-wheel.sh`.
4. `python3 scripts/preflight.py --for serve`. It tests the interfaces, the links, MTU 9000 and a full-size ping over each ring cable.
5. Check the start order: rank 2, rank 1, then rank 0. The ranks find rank 0 at `MASTER_ADDR` over the LAN. If you started them in another order, stop all three and start again in order.
6. After a reboot, run `scripts/stop.sh` and then `scripts/host-prep.sh` on each box before starting (see [OPERATIONS.md](https://github.com/jakejharris/jspark3/blob/v2.0.1/docs/OPERATIONS.md#reboot)).
   **Erratum:** that tagged section's claim that warm starts are quick because kernels are cached is incorrect
   for unmodified v2.0.1. Every start installs the engine wheel in a fresh container, refreshing source timestamps
   and rebuilding kernels even with the cache retained. Allow compilation to finish; see
   [kernels rebuild on every start](#kernels-rebuild-on-every-start) for the pending hotfix and
   [possible stale Torch extension lock](#possible-stale-torch-extension-lock) if compiler output stops advancing.
7. If other programs on the boxes use GPU memory, loading can run out of memory. See [OPERATIONS.md](https://github.com/jakejharris/jspark3/blob/v2.0.1/docs/OPERATIONS.md#other-work-on-the-boxes).

**When it is just slow:** every v2.0.1 start installs the engine wheel in a fresh container. Pip refreshes the
compiler-source modification times, so the Torch extensions rebuild even with `$DATA/kernel-cache` retained.
The documented first and later start times are both about 5 minutes for all three hosts, including compilation.
If compiler output is still advancing, wait longer with `scripts/wait-ready.sh --timeout <seconds>`. A compile
error needs diagnosis from the log; a retained cache does not guarantee a successful or faster start.

See [v2.0.1 known issues and hotfixes](#v201-known-issues-and-hotfixes) for the pending rebuild hotfix and
[hypothetical stale-lock recovery](#possible-stale-torch-extension-lock), including verification and undo.

## Status, stop and clearing the session store

| Message | Means | What to do |
|---|---|---|
| `rank <R>: no container named jspark3-rank<R> (start it with scripts/serve.sh)` (exit 1) | This box's rank is not running. | Start it, in order. |
| `jspark3-rank<R> was not started by this recipe (CONTAINER_PREFIX in cluster.env names this recipe's containers)` | A container this recipe did not start has the rank's name. `status.sh` does not report on it. | Check `CONTAINER_PREFIX` in `cluster.env`. If it is right, that container belongs to something else: set another prefix on all three boxes. |
| `rank <R>: jspark3-rank<R> <state> since <time>; release <version>, <weights> weights, drafter <drafter>` | The container and how it was started. | Compare the three boxes. They must match. |
| `session tier: disk (...)`, `session tier: off (...)` | Whether this container keeps conversation state on disk. | Nothing. |
| `identity: <item> <value> (as pinned)` (from `status.sh --identity`) | That part of the running container matches this release. | Nothing. |
| `identity: <item> <value> DIFFERS from <pinned value>` (exit 1) | That part of the running container does not match this tree's pins. The item names which: the image, the engine wheel or version, the chat template, the fabric launcher, the rank data manifest, the settings profile or the recipe tree. | For `settings profile`: stop all three, `git checkout config/profiles/`, start again. For `recipe tree`: the tree changed after the start (`git status` shows what); restore it or restart from it. For anything else, redo that INSTALL step on this box, then stop and start all three. |
| `session tier: unknown (this container was not started by this recipe's serve.sh)` | The container has no session-tier label. | Stop it and start it with `scripts/serve.sh`. |
| `API on port <port> does not answer yet (still loading, or see scripts/wait-ready.sh)` | Rank 0 is not ready yet. | `scripts/wait-ready.sh` |
| `jspark3-rank<R> stopped and removed` | Done. Its Docker log went with it. | Nothing. |
| `jspark3-rank<R> stopped and kept (with its logs); start the next run under another prefix, or remove it with scripts/stop.sh` | `stop.sh --keep` stopped the container and kept it with its log. While it exists, `serve.sh` won't start this rank under the same name and `clear-sessions.sh` refuses. | Read or save its log, then remove it with a plain `scripts/stop.sh`, or start the next run under another prefix ([INSTALL.md](../INSTALL.md#stopping-restarting-upgrading)). |
| `jspark3-rank<R> is not on this box; nothing to stop` | Nothing to stop. | Nothing. |
| `jspark3-rank<R> was not started by this recipe; it is left alone (CONTAINER_PREFIX in cluster.env names this recipe's containers)` | A container with the rank's name exists, but this recipe did not start it. `stop.sh` did nothing. | Check `CONTAINER_PREFIX` in `cluster.env`. If it is right, the container belongs to something else; stop it the way it was started. |
| `docker could not stop jspark3-rank<R>` | `docker stop` failed (with `--keep`). | Read `logs/stop.log`. Check that Docker runs (`docker info`). |
| `docker could not remove jspark3-rank<R>` | `docker rm -f` failed. | Read `logs/stop.log`. Check that Docker runs (`docker info`). |
| `jspark3-rank<R> exists on this box: stop it first (scripts/stop.sh), on all three boxes` | `clear-sessions.sh` refuses while this rank's container exists, running or kept with `--keep`. | Run a plain `scripts/stop.sh` on all three boxes, then clear on each box. |
| `nothing to clear at <folder>` | No saved sessions for that weights and draft-model pair. | Nothing. Use `--all` to clear every pair. |
| `could not delete the session files` | Deletion failed. | Read `logs/clear-sessions.log`. Check that your user owns `$DATA/sessions`. |

## Checking answers: smoke and exactness

Run both on rank 0 after any change, while no other client uses the server.

**`scripts/smoke.sh`** prints one line per check, then `SMOKE PASS <n>/<n>` or `SMOKE FAIL <n>/<n>`, and exits 1 on any failure.

| Line | Checks | If it fails |
|---|---|---|
| `models` | `glm53` is listed. | The API answers but serves something else. Check `scripts/status.sh` on rank 0. |
| `arithmetic` | 17*23 is answered as 391. | Wrong answers from a server that is up usually mean mismatched parts: check that all three boxes show the same weights and release, and that the splits are verified (`scripts/split.sh --verify-only`). |
| `json` | An exact JSON reply. | As above. |
| `image` | A solid red test image is named as red. | As above. If only this fails, include the line in your report. |
| `count` | Counting 1 to 200 with nothing missing or wrong. | Corrupted long output. Run `python3 scripts/exactness.py` and check the ring cables with preflight. |
| `streaming` | A streamed reply arrives as well-formed server-sent events ending in `[DONE]`, with visible text, final token counts and the server's timings, and its first token arrives within a generous limit. | A failure here shows as the `request` line below. |
| `request` | A request or the streamed reply failed. The line `SMOKE FAIL request/streaming: <reason>`, printed before the results, says why without showing the server's reply or address. | Check `scripts/wait-ready.sh`. `server unreachable or timed out` means the check could not connect, or timed out before the server answered; it does not prove that nothing answered. Check the address in `cluster.env` or the URL you gave. For a timeout or a 5xx, read the first error in `docker logs jspark3-rank0`; a timeout alone does not show that the server is unwell. For `stream first-token time exceeded its generous bound`, check that no other client is using the server, then run it again. |

**`python3 scripts/exactness.py`** sends each prompt twice, once drafted and once one token at a time, and checks that the replies match token for token.

| Result | Means | What to do |
|---|---|---|
| `EXACTNESS PASS` (exit 0) | Every pair matched, and drafting was proven to happen. | Nothing. |
| `EXACTNESS REJECT` (exit 20) | A pair differed. Its line says `first differing token at <n>`. | This should not happen on a release build. Stop using the server for real work, and report it with `exactness.json`. It holds the replies to the public test prompts, nothing from your users. |
| `EXACTNESS INVALID` (exit 21) | The run proved nothing either way. Each prompt line or an `INVALID:` line says why. | See below. |

`INVALID` reasons:

- `request failed: <error>`: a request did not complete. Check the server is ready and quiet, then run again.
- `cannot verify one serving start`: `/metrics` did not report the server's start time. The check needs it to prove both replies came from one serving start.
- `serving start changed`: the server restarted during the run. Find out why (`docker logs jspark3-rank0`), then run again.
- `no token evidence (ids, token_sha or stats missing/invalid)`: the server did not return the token evidence the check needs. Check that you serve this release, not another server on the same port.
- `drafting or serial reference not proven`: the drafted reply showed no drafting, or the reference reply did. Check how the server was started (`scripts/status.sh`).

## Checking the prompt cache and prompt reading

Run both on rank 0 after `scripts/wait-ready.sh`, while no other client uses the server. Each prints its progress and ends with a PASS line (exit 0) or a FAIL line that names the step and the reason. A FAIL exits 1, or 2 when the check could not connect or timed out before the server answered. Reaching a time limit set in the script is a FAIL. Exit 2 also means the arguments were wrong. Neither check shows the server's reply or address.

**`python3 scripts/cache-check.py`** runs three steps (`cold`, `next-turn`, `negative-control`), then prints `SKIP  cache_salt: config/serve.env does not set TF_GLM_CACHE_SALT=1; salt checks were not run` and `CACHE PASS 3/3`. The SKIP is expected on this release: `config/serve.env` does not set `TF_GLM_CACHE_SALT=1`, because this release does not isolate cached prompts per request (see the `cache_salt` known issue [below](#api-behaviour-clients-notice)). It means isolation was not tested, not that it passed.

- `cold prompt reused cached tokens` or `different-nonce control reused cached tokens`: a prompt with a fresh random first line reported cached tokens. Check that you serve this release (`scripts/status.sh --identity`), not another server on the same port.
- `next turn did not reuse the entire earlier prompt`: the server reported fewer cached tokens than the earlier prompt's length, or the next turn was not longer than the earlier prompt. It does not necessarily mean the whole prompt was read again. Run the check again with no other client. If it fails again, report it with the output and `scripts/status.sh --identity`.
- `next-turn arithmetic answer is incorrect` or `cold control and warm arithmetic answers do not agree`: wrong answers from a server that is up. Treat it like a smoke `arithmetic` failure.

**`python3 scripts/prefill-check.py`** sends one prompt of each size to warm up, then measures one new prompt of about 8,000 and one of about 32,000 tokens and prints a line for each with its token count, reading rate and floor. `PREFILL PASS 2/2` means both rates meet the floors set in the script and neither reply took far longer than its floor allows.

- `cold prefill rate is below its floor`: prompt reading was slower than the floor set in the script. Run it again with no other client. If it fails again, check that all three boxes show the same weights and release, and report it with the output.
- `post-warmup reply is a compile-sized latency outlier`: a measured reply took far longer than its floor allows, which can mean GPU code compiled during the run. Run the check again.
- A failure during `warmup-8000` or `warmup-32000`: after a fresh start, warm-up can compile GPU code. Wait for `scripts/wait-ready.sh`, then run the check once more.
- `fresh prompt reused cached tokens; cold prefill was not measured` or `usage prompt length disagrees with /tokenize`: the measurement is not valid. Check that you serve this release, not another server on the same port.
- `--floor-scale <x>` multiplies both floors. Each result line prints the scale used, so report it with any result you share.

## The first reply is slow

Most slow first replies have one of these causes:

- **The first start after install.** GPU code compiles before the server answers at all. `wait-ready.sh` covers that wait. It is not part of any request.
- **A long new prompt.** The server reads the whole prompt before the first token. Time to first token grows with prompt length. [BENCHMARKS.md](https://github.com/jakejharris/jspark3/blob/v2.0.1/docs/BENCHMARKS.md#time-to-first-token-cold-cold_ttft_s) has the measured cold figures. Returning to a conversation is fast while its state is in memory or in the session store.
- **Reasoning comes first.** The model always writes its reasoning before the answer. A request that sets no reasoning effort runs at High effort, so the answer text starts later than at Low.
  - Stream the reply and show `reasoning_content` as it arrives, or send `"reasoning_effort": "low"`.
  - A client that shows only `content` looks idle while the model reasons.
- **More than 8 requests at once.** The 9th and later wait in a queue for a free slot. Nothing returns 429; the wait just shows up as a slow first token.
- **Several requests at the same moment.** With the draft model on, short text prompts that arrive together while no reply is streaming are read together in one pass, and a short text prompt that arrives while no reply is streaming may wait up to 25 ms for a second one; long prompts never wait. Without the draft model, prompts that arrive together are not read together in one batch, and first tokens arrive later. A prompt that arrives while replies are streaming, unless it is very short, is read in small steps between reply steps, so its first token comes later than on an idle server. See the request scheduling limitations [below](#api-behaviour-clients-notice).
- **The same prompt again, an earlier turn again, or a new conversation with the same system prompt.** Without a shorter saved state to resume from, each is read again in full; an exact repeat with the draft model on is skipped while the prompt is still in memory. With the draft model on, regenerating or resending an earlier turn after later turns have been sent resumes only from a shorter state the disk session store has finished saving, and in testing it read the whole prompt again. See the prompt reuse limitations [below](#api-behaviour-clients-notice).
- **Non-streaming requests.** They send nothing until the whole reply is done, reasoning included. Use `"stream": true`.
- **After clearing the session store, or with it off.** A long conversation that left memory is read again from scratch on its next turn.
- **Without the draft model, on a long conversation that includes images.** See [below](#a-long-conversations-next-reply-takes-as-long-as-the-first).
- **Other work on the boxes.** Anything else using the GPUs, memory or the ring cables slows every request.

### A long conversation's next reply takes as long as the first

- **When:** serving without the draft model (`DRAFTER=none`), returning to a long conversation that includes images after it has left the memory cache, for example after other long conversations ran in between.
- **Cause:** Without the draft model, a long conversation that includes images may not be saved to the disk session cache, and each saved state takes more memory, so fewer long conversations stay cached. Returning to such a conversation after it has left the memory cache can take as long as its first prompt. Text-only conversations of about 40,000 tokens are saved; longer text-only conversations were not tested.
- **What to do for now:**
  - Send fewer images in very long conversations while serving without the draft model.
  - Or serve with the draft model, if its non-commercial license fits your use.
  - Or keep fewer very long conversations in rotation at once.
- No setting in this release fixes it. See the long-conversation cache limitation [below](#api-behaviour-clients-notice).

## API behaviour clients notice

The API is OpenAI-compatible chat completions on rank 0, at `http://127.0.0.1:8002/v1`, with model id `glm53`.

**Known issues**, most noticeable first:

1. `stop` is ignored. A reply ends at the model's end of turn or at `max_tokens`.
2. `response_format` is ignored. JSON mode and `json_schema` are not enforced, so the reply is free text.
3. Identical prompts without a `seed` return identical outputs, even above temperature 0: a chat app's regenerate returns the same reply, and two users who send the same prompt get the same answer. Send a different `seed` with each request when you want a different sample.
4. `n`, `logprobs`, presence and frequency penalties and `logit_bias` are ignored.
5. A wrongly typed field, such as a string `temperature`, may return HTTP 500 instead of 400.
6. Non-streaming requests send nothing until the reply is complete. Behind a proxy with an idle timeout, use `stream: true`.
7. The `model` field is not validated; every request is served by GLM-5.3 Flash.
8. Without the draft model, a long conversation that includes images may not be saved to the disk session cache, and each saved state takes more memory, so fewer long conversations stay cached. Returning to such a conversation after it has left the memory cache can take as long as its first prompt. Text-only conversations of about 40,000 tokens are saved; longer text-only conversations were not tested.
9. Reasoning is always on, and no setting turns it fully off; v1.8.4 had it off by default. A request that sets no reasoning effort runs at High, and the lowest effort is low; a top-level `reasoning_effort: "none"` and `chat_template_kwargs: {"enable_thinking": false}` are both treated as low, so a reply can still begin with a short reasoning passage. Even at low effort, a small `max_tokens` can be used up by reasoning and return no visible text; allow a few hundred tokens or more.
10. With the draft model on, a short text request that arrives while no reply is streaming may wait up to 25 ms for a second request before its prompt is read. Without the draft model, prompts that arrive together are not read together in one batch, and first tokens arrive later: with the base weights and 8 requests at once, the median first token arrives after 1.9 s, with visible text 0.41 s later (median over the replies that showed text), against 0.54 s with the draft model, where the median first visible text arrives at 1.5 s; in both sets, 3 of 24 short replies spent their 96-token limit on reasoning and showed no text. Reading prompts that arrive together in one batch without the draft model is planned for v2.0.2. With 16 requests at once, twice the server's 8 reply slots, total output with the draft model on is about 10 to 15% lower than with 8 (about 15% with the base weights, about 10% with the refusal-removed (ablit) weights), because the second eight prompts are read in small steps while the first eight replies stream. A streaming reply can occasionally pause between updates, and the pauses are longest while a long new prompt is being read: the longest measured pause was about 0.75 seconds, with a 36,180-token prompt. While a prompt of that length is being read, one step can pause every streaming reply at once for up to about 0.53 seconds.
11. JSpark3 v2.0.1 saves the state at the end of each prompt it reads, whichever client sent it, and reuses it when a later prompt starts with that entire earlier prompt, such as the next turn of a conversation; it then reads only the rest. Sharing only a system prompt is not enough: no state is saved where a system prompt ends, so a prompt with the same system prompt but a different first message is read in full. With the draft model on, it also skips reading a prompt that exactly repeats the latest prompt of a conversation, such as regenerating the latest reply, while that state is still in memory. With the draft model on, only the latest state of each conversation stays in memory, so regenerating or resending an earlier turn after later turns have been sent does not get this shortcut. Such a request resumes only from a shorter state that the disk session store has finished saving. The store saves in the background while the server is idle and may not yet hold a given turn, or may have skipped it; in testing, these regenerations read the whole prompt again. Without the draft model, an exact repeat is never skipped, but earlier turns' states can stay in memory until evicted, so regenerating a later turn can resume from the previous turn's state. Regenerating the first reply of a conversation after later turns, or resending it without the draft model, reads the whole prompt again, because saved state is reused only when it is shorter than the new prompt.
12. Conversations that share only a system prompt do not share cached work. Saved state is matched by prompt content, not by conversation: a prompt reuses an earlier prompt's state only when it starts with that entire earlier prompt, whichever conversation sent it. No state is saved at the end of a system prompt, so a new conversation that starts with the same system prompt as an earlier one, but has a different first message, reads its whole prompt again. A fix is planned for v2.0.2.
13. A client that disconnects while its connection's socket number is 1024 or higher is not detected, so its generation runs to completion and holds its slot. Normal connection counts do not reach this; very many idle keep-alive clients could. A fix is planned for v2.0.2.
14. If `max_tokens` cuts off a tool call, `finish_reason` is `length` (or `tool_calls` if an earlier call in the same reply was complete), the cut-off call is left out of the final `tool_calls`, and its raw text is returned in `content`. When streaming, its name and partial `arguments` (incomplete JSON) have already been sent. Raise `max_tokens` for tool use.
15. The `usage` block in replies does not include `prompt_tokens_details.cached_tokens`. The number of prompt tokens the server reused from saved state is reported in the reply's `tensorfold.cached` field instead (in the final chunk when streaming). For a request that forces a tool call, this count can be too high, even above the prompt's length. v1.8.4 returned this field, so a client that reads it must switch to `tensorfold.cached` when upgrading. A fix for both is planned for v2.0.2.
16. v2.0.1 does not support per-request cache isolation. It ignores the `cache_salt` request field, and all clients of one server share its saved prompt state. A request whose prompt starts with another client's entire earlier prompt reuses that state, which shows in the reply's cached-token count and in a faster first token. v1.8.4's engine honored `cache_salt`, so a deployment that relied on it to keep clients apart is no longer isolated after upgrading. If clients must not learn about each other's prompts, give each one its own server with its own session folder. Per-request isolation is planned for v2.0.2.

**Thinking: always on; the lowest reasoning effort is low.** This release has no thinking-off mode.

- The reasoning comes in `reasoning_content` and the answer in `content`. The reasoning is never folded into `content`.
- `usage.completion_tokens` counts every output token, reasoning included. There is no separate count of reasoning tokens.
- Requests that set no reasoning effort run at high effort; set `"reasoning_effort": "low"` for shorter replies.
- `"reasoning_effort": "low"`, `"minimal"` or `"none"`, or `"chat_template_kwargs": {"enable_thinking": false}`, gives Low. The reply still has a reasoning section.
- `"medium"` or `"high"` gives High; `"xhigh"` or `"max"` gives Max.
- **Empty `content` with `finish_reason: "length"`** means `max_tokens` ran out while the model was still reasoning. Raise `max_tokens`, or ask for Low effort. Even at low effort, a small `max_tokens` can be used up by reasoning and return no visible text; allow a few hundred tokens or more.
- Clients built for v1.8.4, which had reasoning off by default, should send `"reasoning_effort": "low"` and read only `content`.

**Cached prompt tokens are in `tensorfold.cached`, not in `usage`.** It is known issue 15.

- A client or dashboard that reads `usage.prompt_tokens_details.cached_tokens` finds no such field.
- Read `tensorfold.cached` instead: the number of prompt tokens the server reused from saved state. In a non-streaming reply it sits next to `usage`. When streaming, it is in the final chunk, also when the request does not ask for usage.

**HTTP errors:**

| Status | Usual cause | What to do |
|---|---|---|
| 400 `context_length_exceeded` | The prompt is longer than the context window. | Shorten the conversation. |
| 408 | The request body arrived too slowly. | Check the client's network path to the server. |
| 413 | The request body is over 192 MiB. | Send smaller images or fewer of them. |
| 500 | A wrongly typed field (see [API limitations](#api-behaviour-clients-notice)). | Check field types. |
| 501 on `OPTIONS` | The server has no CORS support. | Browsers can't call it directly. See [Reaching the server safely](#reaching-the-server-safely). |
| 503 | Too much request data in flight at once, or too many requests with images ([below](#images-in-requests)). | Retry with backoff. Limit concurrent large requests at your proxy. |
| connection refused | You are not on rank 0, or the server is not ready. | See [Reaching the server safely](#reaching-the-server-safely) and `scripts/wait-ready.sh`. |

[OPERATIONS.md](https://github.com/jakejharris/jspark3/blob/v2.0.1/docs/OPERATIONS.md#capacity) explains slots, queueing and limits.

### Images in requests

Images go in a chat message as inline base64 `data:` URLs ([OPERATIONS.md](https://github.com/jakejharris/jspark3/blob/v2.0.1/docs/OPERATIONS.md#images-in-requests)). The error's `message` field says what was refused.

| Message | Means | What to do |
|---|---|---|
| `remote image URLs are disabled; send a data: URI ...` (HTTP 400) | The image part points to an `http://` or `https://` address. This release does not fetch images. | Download the image yourself and send it as a base64 `data:` URL, for example `data:image/png;base64,<the file in base64>`. |
| `image URLs must be data: URIs ...` (HTTP 400) | The image part's `url` is neither a `data:` URL nor a web address. | Send it as a base64 `data:` URL, as above. |
| `the image data URL does not decode` or `the image could not be decoded` (HTTP 400) | The base64 text is broken, or the file is not a JPEG, PNG or WebP image. | Re-encode the file as JPEG, PNG or WebP, then base64 it again without line breaks. |
| `an image is larger than 32 MB`, `image data URL exceeds the 32 MB encoding limit` or `an image exceeds 32000000 decoded pixels` (HTTP 400) | One image is over a per-image limit. | Resize or recompress it. |
| `a request may contain at most 16 images` (HTTP 400) | Too many images in one request, counting the whole conversation. | Send fewer images, or start a new conversation. |
| `image requests are busy (4 active or waiting); retry later` (HTTP 503) | Four requests with images are already being prepared, waiting or answered. A fifth is refused, not queued. | Retry with backoff. Limit requests with images to 4 at a time at your proxy. |
| `image preparation is busy (2 slots); retry later` or `prepared image budget is busy (1024 MiB); retry with fewer or smaller images` (HTTP 503) | The server is already preparing as many images as it can hold. | Retry with backoff, with fewer or smaller images. |

## Reaching the server safely

The server listens on loopback (127.0.0.1) only, with no authentication and no CORS. Anyone who can reach the port can use the model. So:

- **"Connection refused" from another machine is expected.** Use an SSH tunnel from your laptop to rank 0:

  ```sh
  ssh -N -L 8002:127.0.0.1:8002 <you>@<rank 0 address>
  ```

  Then use `http://127.0.0.1:8002/v1` on the laptop.
- **For other users or services,** put a reverse proxy with authentication and TLS on rank 0 in front of `127.0.0.1:8002`. [OPERATIONS.md](https://github.com/jakejharris/jspark3/blob/v2.0.1/docs/OPERATIONS.md#reaching-the-api-safely) lists what it needs.
- **Don't change `API_HOST` to `0.0.0.0` or a LAN address.** That puts an unauthenticated model on your network. Preflight warns about it.
- **Browser apps** can't call the API directly: `OPTIONS` returns 501 and there are no CORS headers. Call it from a server, or add CORS at your proxy.
- A port that is "already in use" on your laptop means something there already uses 8002. Pick another local port: `-L 18002:127.0.0.1:8002`, then use `http://127.0.0.1:18002/v1`.

## Asking for help

Open an issue at github.com/jakejharris/jspark3 with:

- the release (`scripts/status.sh` shows it) and what you ran;
- the exact message, and the output of `python3 scripts/preflight.py --for serve` from every box (safe to share as it is);
- the log the message names (`logs/<script>.log`), after you read it. It is readable only by you and keeps your own paths and user name, so remove them first;
- `scripts/status.sh` from every box;
- the first error from `docker logs jspark3-rank<R>`, with a few lines around it;
- for wrong answers, `SMOKE` output and `exactness.json`.

Before posting any log, remove host names, addresses, user names, home paths, tokens and anything from your users' prompts. Never share `cluster.env` as it is, and never share files from `$DATA/sessions`. [OPERATIONS.md](https://github.com/jakejharris/jspark3/blob/v2.0.1/docs/OPERATIONS.md#logs-and-what-to-share) has the full list.
