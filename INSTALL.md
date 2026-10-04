# Installing JSpark3 v2.0.1

This is the maintained v2.0.1 guide. The `main` branch carries this guide and the source checker; the runnable
recipe is at the `v2.0.1` tag. Keep the two checkouts separate as step 1 shows. The older `docs/INSTALL.md` on main
belongs to v1.1.0. These corrections do not change the v2.0.1 engine or its launch scripts.

Before troubleshooting an install or restart, see [v2.0.1 known issues and hotfixes](docs/TROUBLESHOOTING.md#v201-known-issues-and-hotfixes).
The separate kernel rebuild hotfix instructions are **pending hardware validation (target 2026-10-04)**.

This guide takes three NVIDIA DGX Sparks from nothing to a running GLM-5.3 Flash server, using only public downloads.
Every download is pinned and checked by sha256, and the engine is built from the source in the tagged recipe.

> **Security: the server has no authentication and no CORS policy.** By default rank 0 listens on loopback only
> (`127.0.0.1:8002`). Reach it through an SSH tunnel, for example `ssh -N -L 8002:127.0.0.1:8002 you@rank0-host`,
> then use `http://127.0.0.1:8002/v1` on your own machine. If others need it, put an authenticating reverse proxy in
> front of it. Never expose the port to a network you don't fully control.

> **The session tier stores conversation state on disk.** By default each box keeps up to 64 GiB of conversation
> state on its own disk, including the draft model's cache state, so a long conversation you come back to resumes
> quickly. Nothing is sent anywhere. See [Session tier](#session-tier) for where it lives, how to clear it, and how to
> turn it off (`SESSION_TIER=off`).

## Contents

- [v2.0.1 known issues and hotfixes](docs/TROUBLESHOOTING.md#v201-known-issues-and-hotfixes)
- [What you need](#what-you-need)
- [Choose: weights, draft model, session tier](#choose-weights-draft-model-session-tier)
  - [Tuned settings per weights](#tuned-settings-per-weights)
- [Quick start (base weights, the default)](#quick-start-base-weights-the-default)
- [Disk, time and memory per step](#disk-time-and-memory-per-step)
- [Ablit weights (opt-in)](#ablit-weights-opt-in)
- [Running without the draft model](#running-without-the-draft-model)
- [Other containers on the box](#other-containers-on-the-box)
- [Session tier](#session-tier)
- [Where things live](#where-things-live)
- [Checking the result](#checking-the-result)
- [Stopping, restarting, upgrading](#stopping-restarting-upgrading)

## What you need

- **Three NVIDIA DGX Spark** systems running DGX OS, each with Docker and the NVIDIA Container Toolkit (both ship
  with DGX OS), and your user in the `docker` group.
- **Network:** Cable the boxes' ConnectX-7 ports in a ring (rank 0 to rank 1, rank 1 to rank 2, rank 2 to rank 0),
  with each port up, RDMA working and MTU 9000; tensors travel over these cables. Every box also needs a shared LAN
  on which it can reach rank 0. The engine uses that LAN only to coordinate startup; tensor traffic stays on the
  cables.
- **Disk:** a local NVMe directory on each box (`DATA`). See [Disk, time and memory per step](#disk-time-and-memory-per-step).
- **Internet access** on each box during install, for Hugging Face, PyPI and NVIDIA's container registry. No account
  is needed for the default install; the ablit weights need a Hugging Face account.
- **Tools:** `git`, `python3` (3.10 or newer) and the Hugging Face CLI at the version this release was installed
  with (`HF_HUB_VERSION` in `pins.env`). Install it in a virtual environment, which leaves the system Python alone
  (DGX OS refuses a plain `pip install`):
  `python3 -m venv ~/hf-cli && ~/hf-cli/bin/pip install "huggingface_hub==2.1.1"`, then
  `export PATH="$HOME/hf-cli/bin:$PATH"` in the shell you install from; this provides `hf`. If `python3 -m venv` says
  ensurepip is not available, install your distribution's venv package first (`python3-venv` on Ubuntu).
  `scripts/fetch-weights.sh` refuses another version when it downloads; checking files you already have
  (`--verify-only`) does not need the CLI. This CLI runs on the host; the container has its own Python packages
  (`wheels.lock`), a separate environment, so the two versions differ.

On **each box**, add your login user to Docker's group if it is not already a member:

```bash
sudo usermod -aG docker "$(id -un)"
```

Log out completely and log in again (disconnect and reconnect SSH too), then check in the new login shell:

```bash
id -nG                           # must include docker
docker info --format '{{.ServerVersion}}'   # must succeed without sudo
```

## Choose: weights, draft model, session tier

Three settings, each the same on all three boxes. Set them in `cluster.env` or pass them to the scripts.

| Setting | Values | Default |
|---|---|---|
| `WEIGHTS` / `--weights` | `base`: public GLM-5.3 Flash weights (MIT). `ablit`: refusal-removed (abliterated) weights, an opt-in install for ablit development, red-teaming and refusal research ([details](#ablit-weights-opt-in)). Each variant runs with its own tuned settings from `config/profiles/` ([details](#tuned-settings-per-weights)) | `base` |
| `DRAFTER` / `--drafter` | `dflash2`: the DFlash2 draft model (CC BY-NC-ND 4.0, non-commercial). `none`: no draft model; the weights' own multi-token prediction head drafts instead ([details](#running-without-the-draft-model)) | `dflash2` |
| `SESSION_TIER` / `--session-tier` | `disk`: conversation state kept on each box's disk. `off`: none kept on disk ([details](#session-tier)) | `disk` |

The defaults are one line each in `config/defaults.env`.

### Tuned settings per weights

Each weight variant runs with its own tuned settings: `config/profiles/base.env` and `config/profiles/ablit.env`.
The weights setting selects the file, so there is nothing else to set. A profile holds the draft policy used with
the draft model (`SERVE_DRAFT_POLICY`) and the one used without it (`NO_DRAFTER_POLICY`); every other engine setting
is shared, in `config/serve.env`, and the session tier's settings follow `SESSION_TIER` only. Each value in a profile
carries a comment saying where it comes from. `serve.sh` checks the file before anything starts and refuses a value a
profile may not hold. Each container records the sha256 of the profile it started with, and
`scripts/status.sh --identity` reports a profile that differs from the one shipped. The profiles are generated by the
release tooling; editing one gives you a configuration that was not measured.

The two weight variants side by side:

| | base (default) | ablit (opt-in) |
|---|---|---|
| What | GLM-5.3 Flash, 4-bit MLX-format weights with the multi-token prediction head | Refusal-removed (abliterated) weights from `orcarouter/GLM-5.3-Flash-Uncensored-MLX` |
| License | MIT | MIT, plus the source model card's use conditions ([quoted below](#ablit-weights-opt-in)) |
| Access | Anonymous download | Gated: Hugging Face account, accept the source's terms on its page, your own `HF_TOKEN` |
| Tuned settings | `config/profiles/base.env` | `config/profiles/ablit.env` |
| Install steps | fetch, split, serve | fetch (base and source), convert, split, serve |
| Download and verify | `scripts/fetch-weights.sh` | `scripts/fetch-weights.sh --weights ablit` |
| Convert | not needed | `scripts/convert-ablit.sh` |
| Split and verify | `scripts/split.sh` | `scripts/split.sh --weights ablit` |
| Serve | `scripts/serve.sh` | `scripts/serve.sh --weights ablit` |
| Status of the tagged recipe | ready | ready: download, conversion and the three thirds (`scripts/convert-ablit.sh`), or checking a third you already have (`manifests/ablit/`) |

## Quick start (base weights, the default)

Do steps 1 to 6 on **every box** unless a step says otherwise.

**1. Get the recipe and check upstream sources first.**

```bash
git clone --depth 1 --branch main https://github.com/jakejharris/jspark3.git jspark3-install-docs
git clone --branch v2.0.1 https://github.com/jakejharris/jspark3.git && cd jspark3
../jspark3-install-docs/scripts/check-sources.sh "$PWD/pins.env"
cp cluster.env.example cluster.env
```

Run the source check before installing the CLI, pulling the image or downloading weights. It needs only Python
3.10+ and makes anonymous, read-only API/HEAD requests: the base and drafter revisions must be public, the ablit
revision's metadata must resolve and its weight-file HEAD must return **401 / GatedRepo**, and the NGC manifest
digest must resolve with the pinned arm64 image ID. No login, personal token, model download or Docker daemon is used.
It ignores Hugging Face login state and `HF_ENDPOINT` so it checks the upstream sources themselves. A 401 alone
does not prove the gated revision exists; the metadata check is required too.

Expect `check-sources: 4 passed, 0 failed` and exit 0. Any changed response, missing pin, timeout or network error
exits nonzero: stop and investigate before the expensive steps. This checks source availability, not every weight
file or image layer; the install scripts still verify downloaded bytes. Maintainers can run the same command
periodically (for example daily), using an absolute path to the tagged checkout's `pins.env`, and alert on nonzero.
The source checker is maintained on main and is not present in the immutable v2.0.1 tag. Continue using this guide
while running all remaining commands from the `jspark3` tagged checkout.

Edit `cluster.env`: `DATA` (a local NVMe folder you own), `RANK` (0, 1 or 2; different on each box), `MASTER_ADDR`
(rank 0's LAN address), and the interface names: `LAN_IFACE`, plus `PREV_IFACE` and `NEXT_IFACE` (the two cable ports;
`ip -br link` lists them). Optionally set `PREV_PEER` and `NEXT_PEER` to the neighbours' cable addresses, so preflight
can test each cable at MTU 9000.

**2. Create `DATA` and check the box.**

```bash
mkdir -p "$(. ./cluster.env && echo "$DATA")"    # the DATA folder you set in cluster.env
```

If you will download the weights or the draft model in step 4, check the box for the download:

```bash
python3 scripts/preflight.py --for fetch
```

It checks `python3`, Docker, that `DATA` is writable, that the Hugging Face CLI is the pinned version and, for the
ablit weights, that `HF_TOKEN` is set. Each line is PASS, WARN or FAIL, with what to do. Output names settings, never
your addresses or paths, so you can share it.

If everything step 4 would download is already on the box (see **Already have the files?** under step 5), skip this
check: it fails without the Hugging Face CLI, and checking files you already have does not need it. The preflight in
step 6 checks the box before serving.

**3. Pull the image, fetch the dependency wheels, build the engine.**

```bash
scripts/pull-image.sh       # the pinned NVIDIA PyTorch image, by digest, anonymously; checks its image ID
scripts/fetch-wheels.sh     # the engine's PyPI dependencies, each checked against wheels.lock
scripts/build-wheel.sh      # builds engine/ inside the image without network; prints and checks its content digest
```

The build, split and conversion steps run in temporary containers that docker removes when each finishes, and
`build-wheel.sh` replaces the engine wheel it built before (in `wheels/`) when it builds again. Apart from what "What
the scripts remove" lists ([Where things live](#where-things-live); each step replaces its own earlier output, and the
split and the conversion do so only after their space check passes), none of them removes anything else.

**4. Download the weights and the draft model.**

```bash
scripts/fetch-weights.sh
```

This downloads the base weights and the DFlash2 draft model at pinned revisions and checks every file against
`manifests/inputs/`. It resumes if interrupted. Rather than downloading on every box, you can download and split on
one box and copy each box its third (step 5); a box that receives its third needs only the draft model:
`scripts/fetch-weights.sh --drafter-only`.

**5. Split the weights into this box's third.**

```bash
scripts/split.sh
```

The split runs on the CPU inside the image without network, then every file of the result is checked against the
shipped `manifests/base/rank<R>.sha256`. A mismatch names the files and nothing is marked verified. To prepare all
three thirds on one box instead, run `scripts/split.sh --all`, copy `$DATA/base/rank1` to rank 1's `$DATA/base/` (and
rank 2 likewise), then run `scripts/split.sh --verify-only` on each receiving box; it checks every file of the copy,
even if the folder is already marked verified.

Do steps 3 and 5 before you start serving on a box, not while it serves: building and splitting churn through
memory, and on a box that is serving, the kernel's background memory compaction can then stall the server. The
scripts warn if a server from this recipe is running.

**Already have the files?** Any input can come from a copy you already have on the box, as long as it checks out.
Put it where the scripts expect it (move or copy it, or make a symbolic link to a folder on the same box), then run
the check instead of the download:

| You have | Put it at | Then run |
|---|---|---|
| the dependency wheels listed in `wheels.lock` | `wheels/` in the recipe folder | `scripts/fetch-wheels.sh --verify-only` |
| the base weights at the pinned revision | `$DATA/base/weights` | `scripts/fetch-weights.sh --verify-only` |
| the DFlash2 draft model at the pinned revision | `$DATA/drafter` | `scripts/fetch-weights.sh --drafter-only --verify-only` |
| the refusal-removed source at the pinned revision | `$DATA/ablit/source` | `scripts/fetch-weights.sh --weights ablit --drafter none --verify-only` (checks the base weights too) |
| this box's third of the base weights | `$DATA/base/rank<R>` | `scripts/split.sh --verify-only` |
| this box's third of the ablit weights | `$DATA/ablit/rank<R>` | `scripts/split.sh --weights ablit --verify-only` |

Each check reads every file and compares it with the shipped lists; only a complete match is marked verified, and
`serve.sh` refuses anything unverified. Checking leaves the files themselves unchanged. A check of the weights, the
draft model or a third reads every file even if the folder was marked verified before: it removes the folder's mark
first and writes it again only on a complete match. If a check fails or is stopped, the folder stays unverified until
a check of it passes; after a stopped check, run the same command again.

A link is fine for anything that is only checked and served. Weights you will split (`$DATA/<weights>/weights`) or
convert from (`$DATA/base/weights`, `$DATA/ablit/source`) are read inside the container through `$DATA`, where a link
to a folder outside `$DATA` leads nowhere: move them there, or copy them with `cp -al` (hard links: no extra disk,
same filesystem only).

Link only the folders this table names (for example `$DATA/base/rank<R>`), never a parent such as `$DATA` or
`$DATA/base`. Each check removes the `.verified` marker beside the folder it checks and writes it again only when
every file matches, and a later split or conversion, once its space check passes, replaces its own output folders (see
"What the scripts remove" below); through a linked parent, both would happen inside your original copy.

**6. Preflight for serving.**

```bash
python3 scripts/preflight.py          # GPU, RDMA ports, interfaces and MTU, cable pings, locked memory, ports, disk
sudo sysctl -w vm.compaction_proactiveness=0   # if preflight warns about it (see below)
scripts/host-prep.sh                  # optional, needs sudo: turns swap off and drops the page cache (see below)
```

Preflight warns when `vm.compaction_proactiveness` is not 0. With background compaction on, a box that has just
built or split can stall the server while the kernel compacts memory; setting it to 0 avoids that. Preflight only
reads the setting (it cannot use sudo); `host-prep.sh` sets it along with the rest. Either lasts until the next
reboot.

Preflight also checks that the container name this box will use (`<CONTAINER_PREFIX>-rank<R>`, by default
`jspark3-rank<R>`) is free, and warns if other containers or processes are using the box. See
[Other containers on the box](#other-containers-on-the-box).

**7. Start: ranks 2 and 1 first, then rank 0.**

```bash
scripts/serve.sh            # on rank 2, then on rank 1, then on rank 0
```

On rank 0, `serve.sh` waits until the model loads on all three boxes and answers a request through them, printing
progress every minute; every v2.0.1 start rebuilds the Torch GPU extensions, including restarts with a kernel cache.
It gives up after 40 minutes with a message saying
what to check (`--timeout` changes that; `scripts/wait-ready.sh` waits again). It also prints the active session tier.

**8. Check it.**

```bash
scripts/smoke.sh                     # on rank 0, or on your machine through the tunnel
python3 scripts/exactness.py         # drafted replies must equal non-drafted ones, token for token
python3 scripts/cache-check.py       # the next turn reuses the earlier prompt; a new prompt reuses nothing
python3 scripts/prefill-check.py     # reading a new prompt of about 8,000 and 32,000 tokens meets the floors
```

If you changed `API_PORT`, give the exactness check the port: `python3 scripts/exactness.py --url http://127.0.0.1:<port>`.

Run the cache and prefill checks on rank 0 after `scripts/wait-ready.sh`, while no other client uses the server. Like
`smoke.sh`, they read `API_HOST` and `API_PORT` from `cluster.env`, or take the server's URL as their first argument.
The cache check prints a `SKIP  cache_salt` line, then `CACHE PASS 3/3`. The SKIP is expected on this release:
`config/serve.env` does not set `TF_GLM_CACHE_SALT=1`, because this release does not isolate cached prompts per
request (see the `cache_salt` known issue in [LIMITATIONS.md](https://github.com/jakejharris/jspark3/blob/v2.0.1/LIMITATIONS.md)). `PREFILL PASS 2/2` means both reading
rates meet the floors set in the script.

The API is OpenAI-compatible: `POST /v1/chat/completions`, model name `glm53`. Read
[LIMITATIONS.md](https://github.com/jakejharris/jspark3/blob/v2.0.1/LIMITATIONS.md) before pointing a client at it.

`host-prep.sh` turns swap off, stops proactive memory compaction, drops the page cache and compacts free memory. It
persists nothing (a reboot undoes it). Apart from the compaction setting above, it is not needed for correct or
stable serving: the engine sizes its memory from what the kernel reports as available, which counts reclaimable page
cache, and reads conversation state with direct IO.

### Other containers on the box

The scripts name their containers `<CONTAINER_PREFIX>-rank<R>` (by default `jspark3-rank<R>`) and label them. They
never stop, remove or reuse a container they did not start: if a container with that name exists, `serve.sh` and
preflight refuse and say so. To run alongside containers you keep (for example a previous release's, stopped), set
`CONTAINER_PREFIX` in `cluster.env` to a name nothing else uses, the same on all three boxes. Stopped containers
from other releases can stay; running ones hold GPU memory and ports, so stop them with their own tools before you
start this one ([UPGRADING.md](UPGRADING.md)).

## Disk, time and memory per step

Per box, base weights with the draft model. Download, build, conversion and disk figures were measured on one DGX Spark over our connection while another job shared its network and disk. Start times and the kernel cache were measured on the three-Spark cluster. Pull and download times depend on your connection.

| Step | Disk | Time | Memory |
|---|---|---|---|
| Image (`pull-image.sh`) | about 25 GB | about 35 minutes | n/a |
| Wheels and engine build | under 50 MB | about 7 seconds | n/a |
| Base weights download (`fetch-weights.sh`) | 181,741,759,037 bytes (181.7 GB; 54 files) for the weights; with the 2,342,460,697-byte draft model the download is 184,084,219,734 bytes (184.1 GB) | about 39 minutes, including the checksum check | n/a |
| Draft model download | included in the row above | about 31 seconds | n/a |
| Split, one third (`split.sh`) | 63.9 GB for the first third, 62.8 GB for each of the other two | about 3.5 minutes per third | about 4 GiB |
| Ablit conversion (`convert-ablit.sh`) | below the table | below the table | below the table |
| First start, kernels compiled (`serve.sh`) | about 45 MB per host | about 5 minutes for all three hosts, including compiling the kernels | the server uses the box's unified memory |
| Later starts (`serve.sh`) | existing kernel cache rewritten | about 5 minutes, including rebuilding the Torch extensions | as above |
| Session tier | up to 64 GiB, written only while 150 GiB stays free | n/a | n/a |

**Ablit conversion** (opt-in, one box). Measured on one DGX Spark: the 200.1 GB source download took about an hour on our connection. Converting, splitting and checking then took about 26 minutes of processing, used no GPU and under 4 GiB of process memory, and needed about 371 GB of free disk beyond the downloaded source (about 571 GB in all), on top of the base weights you already installed. `scripts/convert-ablit.sh` checks for about 400 GB free before it starts.

**Total disk to budget (estimates from the rows above).** GB means decimal GB; GiB means 2^30 bytes. These totals
assume Docker's image storage, the recipe and `DATA` share a disk; if they do not, budget each filesystem separately.

| Install path | Working files, before the session tier | With the full session tier and its free-space reserve |
|---|---|---|
| Base, download and split one third on each box | about **273 GB** per box: 25 + 184.084 + 63.9 | about **503 GB** per box: 273 + (64 + 150) GiB, about 230 GB extra |
| Ablit conversion box, following the opt-in steps in place of base steps 4–7 | about **780 GB**: 25 + 184.084 + 200.1 + 371; includes all three ablit thirds, no base third | about **1,010 GB** |
| Ablit conversion after a completed base install, retaining its one base third | about **844 GB**: 780 + 63.9 | about **1,074 GB** |

Allow extra headroom for logs, temporary files and other software: these are rounded working-set estimates, not
minimum disk capacities. The conversion's **400 GB free-space check** is higher than the measured 371 GB of output;
before conversion, allow about **809 GB** total on the opt-in conversion box (25 + 184.084 + 200.1 + 400), or
**873 GB** if retaining a base third, even with the session tier off. With the full session reserve, the larger
totals in the last column cover this check. A base box that splits all three thirds instead of one needs another
125.6 GB (62.8 + 62.8). A box receiving only its third does not need the full download or conversion workspace.

**Restart timing in v2.0.1:** `serve.sh` creates a fresh container and installs the wheel on every start. Pip gives
the installed compiler sources fresh modification times, so Ninja rebuilds the Torch extensions even though
`$DATA/kernel-cache` persists. The roughly five-minute first and later start figures both include compilation;
retaining the cache does not promise a faster restart
([v2.0.1 scripts/serve.sh:125–126](https://github.com/jakejharris/jspark3/blob/v2.0.1/scripts/serve.sh#L125-L126),
[engine/src/tensorfold/cuda/build.py:23–29](https://github.com/jakejharris/jspark3/blob/v2.0.1/engine/src/tensorfold/cuda/build.py#L23-L29)).
See the [pending kernel rebuild hotfix](docs/hotfixes/v2.0.1-kernel-rebuild.md) and
[possible stale compile lock](docs/TROUBLESHOOTING.md#possible-stale-torch-extension-lock).

After `split.sh` has verified this box's third, the full download (`$DATA/base/weights`) is not needed to serve, so
you may delete it. Splitting again then means downloading it again.

## Ablit weights (opt-in)

Refusal-removed (abliterated) weights from `orcarouter/GLM-5.3-Flash-Uncensored-MLX`, an opt-in install for ablit
development, red-teaming and refusal research. The default install is base. The weights are converted on your own
box from the gated source and the base weights; this project hosts none of the v2.0.1 weights.

**License:** MIT (Copyright (c) 2026 Z.AI Co., Ltd), plus the use conditions on the source's model card, quoted
verbatim:

> - It is released **strictly for legitimate research** — interpretability, AI-safety and refusal-mechanism study, red-teaming, robustness evaluation, and controlled experiments.
> - **You assume full responsibility and liability** for how you use it and for everything it generates. Do not deploy it to end users or in production without adding your own safety, moderation, and abuse-prevention layers.
>
> By downloading or using this model you acknowledge and accept the above.

You are responsible for complying with the source model's terms and for how you use this model and what it generates. JSpark3 provides conversion tooling only; it hosts none of these weights and does not endorse any use of them.

**Access:**

1. Create a Hugging Face account (or sign in).
2. Open the source's page, `https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-MLX`, and accept its terms
   there yourself.
3. Create a read token in your Hugging Face settings and export it in the shell you install from:
   `export HF_TOKEN=<your token>`. The scripts read it from the environment and never print or store it. The base
   weights and the draft model are always fetched without it. The scripts send your token only to Hugging Face: to the
   endpoint you use (`https://huggingface.co` by default, or the mirror you set in `HF_ENDPOINT`), and to Hugging
   Face's own hosts when the official huggingface_hub client follows its redirects. Set `HF_ENDPOINT` only to a mirror
   you trust with your token. The script's own requests (the size lookup and the access check) never send the token on
   a redirect to another site, and they refuse a redirect from HTTPS to plain HTTP. An inherited Hugging Face staging
   setting (`HUGGINGFACE_CO_STAGING`) can't redirect the download: the scripts leave it out of every download and use
   `https://huggingface.co`, or the mirror you set in `HF_ENDPOINT`. Before it downloads anything, the base weights
   and the draft model included, the script reads one byte of the source's first weight file to confirm your access,
   so a missing or refused token stops it before any download. It then downloads only the files listed in
   `manifests/inputs/ablit-source.sha256`. `--verify-only` needs no token.

**Install** (in place of steps 4 to 7 above, whose notes still apply; or set `WEIGHTS=ablit` in `cluster.env`). The
conversion writes all three thirds, so it needs to run on one box only:

```bash
scripts/fetch-weights.sh --weights ablit     # checks your access first, then base weights, the gated source and the draft model (unless DRAFTER=none), each checked by sha256
scripts/convert-ablit.sh                     # converts without network; checks the weights and all three thirds
```

Copy `$DATA/ablit/rank1` to rank 1's `$DATA/ablit/` (and rank 2 likewise). On each receiving box:

```bash
scripts/fetch-weights.sh --drafter-only      # the draft model only
scripts/split.sh --weights ablit --verify-only   # reads every file of the copied third and checks it against manifests/ablit/rank<R>.sha256
```

If that box's `cluster.env` sets `WEIGHTS=ablit`, `--drafter-only` first checks your ablit access, so it needs
`HF_TOKEN` there too; it still downloads only the draft model, without your token. To fetch the draft model without a
token, run `scripts/fetch-weights.sh --weights base --drafter-only` instead.

Then, in place of step 6, run preflight for the ablit weights on all three boxes, after the copy and the check, and
start them as in step 7:

```bash
python3 scripts/preflight.py --weights ablit # on all three boxes
scripts/serve.sh --weights ablit             # ranks 2 and 1, then 0
```

Without the draft model, add `--drafter none` to `fetch-weights.sh`, `preflight.py` and `serve.sh` (or set
`DRAFTER=none` in `cluster.env`), and skip `fetch-weights.sh --drafter-only` on the receiving boxes.

You can instead run `fetch-weights.sh --weights ablit` and `convert-ablit.sh` on every box; each box then converts
on its own. The conversion runs `scripts/ablit/reproduce-ablit.py` inside the pinned image, keeps its receipts, input
notices and identity record in `$DATA/ablit/work`, and puts `WEIGHTS-LICENSE.txt` and `TEMPLATE-ADDITIONS-LICENSE.txt`
beside the converted weights: keep them with anything you derive from these weights.

If the source refuses the download, `fetch-weights.sh` says whether your token is missing or the terms are not yet
accepted. Downloads use plain HTTPS (`HF_HUB_DISABLE_XET=1`) unless you export `HF_HUB_DISABLE_XET=0`; the Xet
transfer path has failed on some hosts, and plain HTTPS resumes reliably.

Refusal-removed (abliterated) weights are built locally from `orcarouter/GLM-5.3-Flash-Uncensored-MLX@c02a5f6fa06f0aa444877b44d19fd5c96390329f`, an abliterated GLM-5.3 Flash checkpoint. The conversion uses `TensorFold/GLM-5.3-Flash-MLX-4bit-MTP@76add2a341a1cd90ad0e86bb69839ea9c35827c6` for the four-bit tensor layout and restores its native prediction layer. The source carries an MIT LICENSE plus model-card purpose and deployment conditions. The chat template is the base checkpoint's MIT template plus six lines added by this recipe, also under MIT; its reconstructed bytes are hash-checked.

## Running without the draft model

The DFlash2 draft model is licensed CC BY-NC-ND 4.0: non-commercial use only. It is downloaded unmodified from its
publisher and never redistributed. For commercial use, run without it:

```bash
scripts/fetch-weights.sh --drafter none     # skips the draft model
python3 scripts/preflight.py --drafter none # serving preflight, after splitting as in step 5
scripts/serve.sh --drafter none             # on all three boxes (or DRAFTER=none in cluster.env)
```

Follow the other quick-start steps too. If using fetch-mode preflight, pass `--drafter none` there as well.
Alternatively set `DRAFTER=none` in `cluster.env` on every box so fetch, preflight and serve all select the same mode.

`serve.sh --drafter none` starts the engine without the draft model, with the drafting setting from the weights
profile's `NO_DRAFTER_POLICY`: the weights' own multi-token prediction head drafts up to seven tokens, kept while its
confidence stays at or above the profile's level. You pass nothing else. Its numbers are in the README's third
results set. It was checked on the GPU with the base weights: drafting with the prediction head gave the same tokens, text and finish reasons as plain one-token-at-a-time decoding on every check prompt.

## Session tier

With `SESSION_TIER=disk` (the default), each box keeps the conversation state it computed (the model's cache state
for each conversation, the draft model's cache state when the draft model is on, and the prompt's token ids) on its
own disk, so a long conversation that left memory, or one from before a restart, can resume without recomputing once
the store has saved it.

- **Where:** `$DATA/sessions/<weights>-<drafter>/<engine build>/` on each box (mounted at `/sessions` in the
  container). Files are readable only by root, as the server writes them.
- **How much:** up to 64 GiB per box. The engine skips a write that would leave less than 150 GiB free on that
  disk, so leave room for both; `serve.sh` warns when there isn't. Least recently used state is evicted first;
  nothing expires with time. State from other weights or another engine build is never reused.
- **Clear it:** stop all three boxes (`scripts/stop.sh`), then on each box run `scripts/clear-sessions.sh` (this
  weights and draft model combination) or `scripts/clear-sessions.sh --all` (everything under `$DATA/sessions`).
- **Turn it off:** set `SESSION_TIER=off` in `cluster.env` on all three boxes, or start every box with
  `scripts/serve.sh --session-tier off`. No conversation state is then kept on disk. `serve.sh`, `wait-ready.sh` and
  `status.sh` print which mode is active. Off is not the configuration the numbers were measured with.

Container logs (`docker logs`) are separate from the session tier; `scripts/stop.sh` removes them with the container.

## Where things live

Under `DATA` on each box:

| Path | What |
|---|---|
| `base/weights/`, `ablit/source/`, `ablit/weights/` | downloads and conversions, each with a `.verified` marker beside it once a check passes |
| `base/rank<R>/`, `ablit/rank<R>/` | this box's third, with a `.verified` marker once a check passes |
| `ablit/work/` | the ablit conversion's receipts, input notices and identity record |
| `drafter/` | the DFlash2 draft model |
| `kernel-cache/` | persistent GPU kernel cache; v2.0.1 still rebuilds the Torch extensions on every start |
| `sessions/` | the session tier |

In the recipe folder: `wheels/` (the built engine and its dependencies) and `logs/` (one private log per script,
mode 0600, with the detail behind any failure message; read a log before you share it).

The containers write `kernel-cache/` and `sessions/` as root. To stop serving without removing anything, use
`scripts/stop.sh --keep` on each box (it keeps the container and its logs). `scripts/clear-sessions.sh` removes
session state without sudo; to remove `DATA` entirely, stop all three boxes and use `sudo rm -rf` on it.

**What the scripts remove** on their own; nothing else is removed:
- the temporary containers of the build, the split, the ablit conversion and preflight's locked-memory probe (docker
  removes each when it finishes);
- a folder's `.verified` marker, removed when a check of the folder starts and written again only when the check
  passes;
- the engine wheel `build-wheel.sh` built before, when it builds again;
- an unverified `$DATA/<weights>/rank<R>` that `split.sh` is about to split again, and, before a fresh conversion,
  `convert-ablit.sh`'s own `$DATA/ablit/work`, `$DATA/ablit/weights` and the three `$DATA/ablit/rank<R>` (when they
  are not all verified already), each only after that step's space check passes, so a refusal for lack of space leaves
  them as they were;
- at each start, `serve.sh` clears `kernel-cache/tensorfold/prefix-snapshots`, so every start is cold;
- `scripts/stop.sh` without `--keep` removes this recipe's containers and their logs.

## Checking the result

- `python3 tools/payload.py verify` checks every file of this recipe against `SHA256SUMS`.
- `python3 tests/check-template.py`, `python3 tests/check-serve-render.py` and `bash tests/check-leaks.sh` check the
  chat template, the rendered serve command against the measured one, and that script output carries no tokens,
  addresses or local paths; `tests/run.sh` runs them all.

## Stopping, restarting, upgrading

- **Stop:** `scripts/stop.sh` on each box (order doesn't matter). It removes this recipe's container; weights, kernel
  caches and the session tier stay. `scripts/stop.sh --keep` stops it but keeps the container and its logs; since
  `serve.sh` never reuses an existing container, start the next run under another prefix
  (`scripts/serve.sh --container-prefix <name>` on all three boxes, or `CONTAINER_PREFIX` in `cluster.env`), or
  remove the kept one first with `scripts/stop.sh`. Session state is kept per weights and draft model combination,
  not per prefix, so a run under a new prefix resumes the same conversations.
- **Status:** `scripts/status.sh` on any box. `scripts/status.sh --identity` also reads from inside the running
  container what it runs (image, engine wheel and its content digest, chat template, launcher, data manifest, the
  weights profile and the recipe tree it was started from) and compares each with this tree's pins.
- **If a rank fails:** stop all three, then start ranks 2 and 1, then 0. Containers have no restart policy, so
  after a reboot nothing runs until you start it.
- **Upgrading from v1.8.x, and going back to v1.8.4:** see [UPGRADING.md](UPGRADING.md).
- Operations and troubleshooting: [docs/OPERATIONS.md](https://github.com/jakejharris/jspark3/blob/v2.0.1/docs/OPERATIONS.md),
  [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).
