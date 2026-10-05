# Operations

The v2.0.2 candidate passed the live retry and is ready for review; publication awaits approval. See [the release gate](../RELEASE-GATE.md).
Any performance comparisons below describe the v2.0.1 measurements.

This guide covers running JSpark3 v2.0.2 (GLM-5.3 Flash) after [INSTALL.md](../INSTALL.md) is done. It explains how to:

- start, check and stop the server;
- look after the on-disk session store;
- switch weights or draft model;
- upgrade from v1.8.x, or roll back to v1.8.4;
- handle a reboot, an address change or a full disk.

For error messages, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md). For what the published numbers measure, see [BENCHMARKS.md](BENCHMARKS.md).

Run every command from the top of the release checkout on the box named. A "box" is one DGX Spark.

## Three things to know first

1. **Three boxes, one engine.** Each box runs one part of the model, called a rank: 0, 1 or 2. Rank 0 also serves the API. If any rank stops, the whole engine stops. Start and stop all three together.
2. **The API has no password.** It listens on `127.0.0.1:8002` on rank 0 only. It has no authentication, no rate limit and no CORS. Reach it through an SSH tunnel or an authenticating reverse proxy (see [Reaching the API safely](#reaching-the-api-safely)). Never open port 8002 to a network, and never change `API_HOST` to `0.0.0.0`.
3. **All three boxes must agree.** Each box has its own `cluster.env`. `WEIGHTS` and `DRAFTER` must be the same on all three. `MASTER_ADDR` must be rank 0's LAN address on all three.

## Command summary

| Task | Where | Command |
|---|---|---|
| Prepare memory, once per boot (optional) | each box | `scripts/host-prep.sh` (asks for sudo) |
| Check the box before starting | each box | `python3 scripts/preflight.py --for serve` |
| Start | rank 2, then rank 1, then rank 0 | `scripts/serve.sh` |
| Wait until ready | rank 0 | `scripts/wait-ready.sh` |
| Quick health | rank 0 | `curl -s http://127.0.0.1:8002/health` |
| State of this box | any box | `scripts/status.sh` |
| Is it running exactly what this release ships? | any box | `scripts/status.sh --identity` |
| Check answers after a change | rank 0 | `scripts/smoke.sh`, then `python3 scripts/exactness.py` |
| Check prompt-cache reuse | rank 0 | `python3 scripts/cache-check.py` |
| Check how fast a new prompt is read | rank 0 | `python3 scripts/prefill-check.py` |
| Stop | every box | `scripts/stop.sh` |
| Clear the session store | every box, all three stopped | `scripts/clear-sessions.sh` |
| Turn the session store off | every box | `SESSION_TIER=off` in `cluster.env` |
| Engine log | any box | `docker logs jspark3-rank<R>` (R is that box's rank) |

Every script takes `--help`. Most take `--dry-run`, which prints what the script would do and changes nothing.

## Start

1. **Once per boot, prepare memory on each box (optional):** `scripts/host-prep.sh`. It runs its steps with sudo, so it may ask for your password. It:
   - turns swap off;
   - stops the kernel from compacting memory in the background;
   - flushes and drops the page cache, then compacts free memory.

   Why: the GPU and CPU share each box's memory. The engine sizes itself from available memory when it starts. A box with fragmented memory or a large cache gives it less room. The measured runs used this host state.

   - It is recommended, not needed for correct answers.
   - Run it before the first start after each boot. Running it again before a later start is harmless.
   - It saves nothing to the system's settings, so a reboot undoes it.

2. **Check each box:** `python3 scripts/preflight.py --for serve`. It prints one PASS, WARN or FAIL line per check. It exits 1 on any FAIL. Fix every FAIL before you start (see [TROUBLESHOOTING.md](TROUBLESHOOTING.md#preflight)). Read each WARN; a WARN does not stop you.

3. **Start the ranks in order:** run `scripts/serve.sh` on rank 2, then on rank 1. Wait a few seconds, then run it on rank 0. The ranks find each other through rank 0.

4. **Rank 0 waits until it is ready.** On rank 0, `serve.sh` waits until the model answers a one-token request, which passes through all three ranks, then returns.
   - It gives up after 40 minutes; change that with `--timeout SECONDS`.
   - With `--no-wait` it returns right away; then run `scripts/wait-ready.sh`, which takes the same `--timeout`.

How long a start takes:

- The first start after install compiles GPU code. Measured first start: about 5 minutes for all three hosts, including compiling the kernels.
- The compiled code is kept in `$DATA/kernel-cache` (about 45 MB per host), so later starts skip that step. Measured later start: about 5 minutes.
- Don't stop a first start because it looks stuck. `docker logs -f jspark3-rank0` shows progress.

What a start changes on disk: each start empties `$DATA/kernel-cache/tensorfold/prefix-snapshots`, as in the measured runs. It does not touch the session store, weights or anything else.

**Extra engine flags.** `scripts/serve.sh -- <flags>` passes extra flags to the engine. Use it only for experiments. Any change makes the server differ from the configuration that was measured, so the figures in [BENCHMARKS.md](BENCHMARKS.md) no longer describe your server. The same applies to editing `config/serve.conf`, `config/serve.env` or a settings profile in `config/profiles/`: they are generated from the measured configuration, and you should not edit them. `scripts/status.sh --identity` reports a profile that differs from the one shipped.

## Is it ready? Is it right?

"Ready" and "right" are different checks.

**Ready** means rank 0's API answers. On rank 0:

```sh
scripts/wait-ready.sh                          # waits until the model is listed and answers a one-token request
curl -s http://127.0.0.1:8002/health           # {"ok":true}
curl -s http://127.0.0.1:8002/v1/models        # one model, id "glm53", with its context window
scripts/status.sh                              # container state on this box; on rank 0, API health too
scripts/status.sh --identity                   # also compares what the container runs with this release's pins
```

`--identity` reads from inside the running container: the image, the engine wheel and installed version, the chat template, the fabric launcher, the rank's data manifest, the settings profile and the recipe tree recorded at start. It prints one `identity:` line for each, ending `(as pinned)` or `DIFFERS from <pinned value>`, and exits 1 if anything differs. Run it on every box after an install, an upgrade or any change to the tree.

On ranks 1 and 2, `wait-ready.sh` and `status.sh` only check that the box's own container is running. Those ranks have no API.

**Right** means the answers are what this release produces. Run these checks on rank 0 after any change: a new install, an upgrade, a weights or draft-model switch, a driver or OS update, recabling.

```sh
scripts/smoke.sh                     # prints SMOKE PASS n/m or SMOKE FAIL n/m
python3 scripts/exactness.py         # exit 0 PASS, 20 REJECT, 21 INVALID
python3 scripts/cache-check.py       # exit 0 PASS, 1 FAIL, 2 no response (could not connect, or timed out before the server answered) or bad arguments
python3 scripts/prefill-check.py     # exit 0 PASS, 1 FAIL, 2 no response (could not connect, or timed out before the server answered) or bad arguments
```

If you changed `API_PORT`, pass `--url http://127.0.0.1:<port>` to `exactness.py`; it does not read `cluster.env`. `smoke.sh`, `cache-check.py` and `prefill-check.py` do, or take the URL as their first argument.

- `smoke.sh` runs quick checks that can fail: the model list, an arithmetic answer, a JSON answer, counting to 200, a small inline image and a streamed reply. The streamed reply must arrive as well-formed server-sent events ending in `[DONE]`, with visible text, final token counts and the server's timings, and its first token must arrive within a generous limit.
- `exactness.py` checks that drafting never changes an answer. Each prompt runs twice at temperature 0:
  - once drafted, the normal fast path;
  - once decoded one token at a time with drafting off.

  The two replies must match token for token. One prompt is about 30,000 tokens long, so the long-context path is covered too.
  - Exit 20 (REJECT): a pair differed. It prints the first token that differs.
  - Exit 21 (INVALID): the run could not prove anything, for example a request failed or no drafting took place.
  - The prompts are public. The gate run for this release used a separate, unpublished prompt set.
- `cache-check.py` checks prompt-cache reuse in three steps: a new prompt must report zero cached tokens; the next turn of that conversation must reuse the whole earlier prompt and answer an arithmetic question correctly; and the same conversation with a different first line must reuse nothing and still answer correctly. It then prints a `SKIP  cache_salt` line and `CACHE PASS 3/3`. The SKIP is expected on this release: `config/serve.env` does not set `TF_GLM_CACHE_SALT=1`, because this release does not isolate cached prompts per request (see the `cache_salt` known issue in [LIMITATIONS.md](../LIMITATIONS.md)). It does not test exact repeats or restoring a conversation from the session store.
- `prefill-check.py` checks how fast a new prompt is read. It builds prompts of about 8,000 and 32,000 tokens, sends one of each size to warm up, then one new prompt of each size to measure; every prompt must report zero cached tokens. `PREFILL PASS 2/2` means both reading rates meet the floors set in the script and neither reply took far longer than its floor allows. It is a timing check: a pass does not prove that no GPU code compiled during the run. After a fresh start, warm-up can take longer while GPU code compiles; if a warm-up step fails then, wait for readiness and run it once more.

These checks send real requests, so run them while no other client is using the server.

## Stop and restart

```sh
scripts/stop.sh        # on every box
```

`stop.sh` stops and removes the box's rank container. It never deletes data: the weights, the split, the kernel cache and the session store all stay. INSTALL's "What the scripts remove" lists everything any script removes on its own.

- **Container names.** The rank containers are `jspark3-rank0`, `jspark3-rank1` and `jspark3-rank2`. To use another name in place of `jspark3`, set `CONTAINER_PREFIX` in `cluster.env`, the same on all three boxes. The scripts stop or remove only containers this recipe started. A container that something else started under the same name is left alone.
- **Keeping a stopped container.** `scripts/stop.sh --keep` stops the container but does not remove it, so its log stays readable with `docker logs`. `serve.sh` never reuses or removes an existing container, so the next start under the same name refuses until a plain `scripts/stop.sh` removes it. To start again and keep the old container, start under another prefix, as [INSTALL.md](../INSTALL.md#stopping-restarting-upgrading) describes.

- **Drain first.** Stopping cuts off requests in flight; their clients see a dropped connection. Stop sending new traffic at your proxy, wait for running requests to finish, then stop.
- **Order** does not matter when stopping.
- **Restart** means stop all three, then [start](#start) in order: rank 2, rank 1, then rank 0. You cannot restart one rank on its own. If one rank dies, stop the other two and start all three again.
- **cluster.env edits** take effect at the next start.

## The session store: what it keeps and how to clear it

**What it is for.** The first request in a long conversation spends a long time reading the prompt. The engine saves the processed state at the end of each prompt it reads, so a later request that starts with the whole earlier prompt, such as the next turn of the conversation, can skip the part already read. Not every saved state stays in memory or reaches the disk store (known issue 11 in [LIMITATIONS.md](../LIMITATIONS.md)).

- It changes speed only: a reply from restored state is identical to one computed from scratch.
- It is on in the shipped configuration, and the published numbers were measured with it on.

**Exact repeats, and other clients.** With the draft model on, when a prompt is exactly the same as one the server read recently, images included, the server does not read it again. For each prompt it keeps two things in memory: the processed state, and the model's scores for the first token of the reply. A repeat restores that state, picks its first token afresh with its own request settings, such as temperature and `seed`, and writes the rest of its reply as usual. No part of an earlier reply is stored or sent back. These entries count toward the memory layer's limits in the table below, and nothing expires with time. None is written to disk, so after a restart, or once an entry is pushed out of memory, a repeat is handled like any other prompt. With the draft model on, only the latest state of each conversation stays in memory, so regenerating or resending an earlier turn after later turns have been sent does not get this shortcut. Such a request resumes only from a shorter state that the disk session store has finished saving. The store saves in the background while the server is idle and may not yet hold a given turn, or may have skipped it; in testing, these regenerations read the whole prompt again. Without the draft model, an exact repeat is never skipped, but earlier turns' states can stay in memory until evicted, so regenerating a later turn can resume from the previous turn's state. Regenerating the first reply of a conversation after later turns, or resending it without the draft model, reads the whole prompt again, because saved state is reused only when it is shorter than the new prompt. Like the conversation cache, which skips the part of a prompt the server has already read, the exact-repeat shortcut applies to every client of the server. Cached work is shared across clients. Response timing and the returned `tensorfold.cached` count can reveal that a matching prompt or prefix was already cached, and how many tokens were reused. The cache does not return another client's stored reply.

- v2.0.2 does not support per-request cache isolation (known issue 16 in [LIMITATIONS.md](../LIMITATIONS.md)). If clients must not learn about each other's prompts, give each one its own server with its own session folder.
- This release has no setting that turns the conversation cache off. It is shared by every client, with the same effect: when another client recently sent a prompt that this one begins with, the reply can start sooner and its `tensorfold.cached` count shows how much was reused. Turning the session store off does not change that, because the memory layer stays shared.

**What it keeps.** Each box keeps its own share of every saved conversation:

- the conversation's prompt as token ids, so the files hold encoded conversation content;
- the model's processed state for that prompt (its attention cache, for this box's part of the model);
- with the draft model on, the draft model's state for the prompt as well.

The store stays on each box's own disk; nothing in this release sends it anywhere. The engine creates its files readable by their owner only (files mode 0600, folders 0700).

**Where it lives.** On every box:

```
$DATA/sessions/<weights>-<drafter>/<namespace>/
```

- Inside the container it is mounted read-write at `/sessions`. Saved conversations go under `sessions-v2/rank<R>/` within it.
- Each weights-and-drafter combination gets its own folder, for example `base-dflash2` or `base-none`.
- The namespace comes from the engine version this release pins. A release with a different engine pin writes to a new namespace folder. Old folders stay on disk until you remove them.

**How big it gets.** The limits are per box, set in `config/serve.env`:

| Layer | Setting | Limit |
|---|---|---|
| Memory | `TF_GLM_CACHE_GIB` | 5 GiB, or less if the context window needs the room |
| Memory staging for disk writes | `TF_GLM_SESSION_STAGE_MIB` | 1 GiB |
| Disk | `TF_GLM_DISK_GIB` | 64 GiB for the folder in use |

- The 64 GiB cap applies to the combination you are serving. Each combination you have ever served keeps what it saved, up to its own 64 GiB, until you clear it.
- The engine also skips any disk save that would leave less than 150 GiB free on that disk. It keeps serving, and clients see nothing. New long conversations just stop getting faster on return.

**How long it keeps things.** Nothing expires with time. An entry stays until one of these happens:

- the folder reaches its cap: the least recently used conversations are removed first to make room;
- a start finds an entry saved by different weights or a different engine setup: that entry is deleted;
- you clear the store.

Unless the store fills up, a conversation from months ago can still be on disk.

**Treat it as user data.**

- Protect `$DATA/sessions` the way you protect conversation logs.
- Include it in your data-retention policy.
- Clear it when you must delete a user's data.
- Never copy it off the box to share it.

**How to clear it.**

1. Stop all three ranks with a plain `scripts/stop.sh`. `clear-sessions.sh` refuses while the box's rank container exists, including one kept with `--keep`.
2. Run the same command on **all three** boxes. Each box holds its own share of every conversation, so clear them together.

   ```sh
   scripts/clear-sessions.sh          # this release's folder for the current weights and drafter
   scripts/clear-sessions.sh --all    # everything under $DATA/sessions: every combination, old namespaces too
   ```

3. Start again.

**What clearing costs.** Clearing never changes an answer. It only makes returning conversations slower, once:

- The next request in each existing conversation reads its whole prompt again. So the first reply after a clear takes as long as a brand-new conversation of the same length; [BENCHMARKS.md](BENCHMARKS.md) has the cold time-to-first-token figures.
- That request saves the conversation again, so later returns are fast.

Measured on the base weights with the draft model: a conversation of about 100,000 tokens, pushed out of memory and then continued ([how it was measured](BENCHMARKS.md#returning-to-a-conversation-that-left-memory-session_tierevicted_return_s)), returned its first token in 1.5 s with its state restored from disk. Reading the same conversation from scratch took 49.9 s.

**Turning it off.** Set `SESSION_TIER=off` in `cluster.env` on all three boxes, then restart. To try it for one start, pass `scripts/serve.sh --session-tier off` on all three boxes instead.

- The switch turns off every disk save and removes the `/sessions` mount from the container. With the store off, the session store and the engine's conversation caches write nothing derived from conversations to disk.
- Two things sit outside that statement:
  - **Container logs.** Docker keeps each container's output, the text `docker logs` shows. The engine does not log requests, but an error message or traceback can include text from a request.
    - To clear them, run `scripts/stop.sh` on each box. It removes the container, and with Docker's default log driver (`json-file`) the container's log goes with it. The next start creates a fresh container with an empty log.
    - With `scripts/stop.sh --keep`, the container and its `json-file` log stay until a plain `scripts/stop.sh` or `docker rm` removes the container.
    - To cap how much a running container keeps, set log rotation in Docker's own settings, for example `"log-opts": {"max-size": "10m", "max-file": "3"}` in `/etc/docker/daemon.json`. Docker applies it to containers created after it restarts: stop the ranks, restart Docker, then start. Rotation limits the size; it does not erase what is still in the current file.
    - If your Docker sends logs elsewhere (for example `journald` or `syslog`), removing the container does not clear them. Clear them where they are kept.
    - A log you saved with `docker logs ... > file` is a copy. Delete it when you are done.
  - **Kernel cache.** `$DATA/kernel-cache` holds compiled GPU code. It is designed to be keyed by tensor shapes, not by what prompts say. That is how it is built, not something we audited file by file. Delete the folder if you want it gone; the next start recompiles.
- `wait-ready.sh` and `status.sh` report the result: `session tier: off (no conversation state is kept on disk)`.
- Use only this switch. Changing single settings in `config/serve.env` can leave a disk store on.
- To turn it back on, set `SESSION_TIER=disk`, which is the default.

- What it costs: a conversation that has left memory has no disk copy, so returning to it means reading the whole prompt again. The same applies to every conversation after a restart. Expect about the time of a fresh read of the same length, the from-scratch figure above. Returns with the store off were not measured separately.
- The memory layer still helps conversations that come back quickly, but it holds fewer of them: 8 instead of 64.
- Turning the store off is a change from the measured configuration.

**When to clear it.**

- The disk is filling up.
- After an upgrade: the old version's namespace folders are never read again.
- After you stop serving a weights or drafter combination: its folder stays until you clear it.
- When you must delete user data.

## Switching weights or draft model

You choose these at install, and you can change them later. The rules for every switch:

- Stop all three ranks first.
- Make the same change on all three boxes:
  - set `WEIGHTS=`, `DRAFTER=` or `SESSION_TIER=` in `cluster.env` to keep the change;
  - or pass `--weights`, `--drafter` or `--session-tier` to `serve.sh` on all three boxes to try it for one start.
- The new combination starts with an empty session-store folder. The old folder stays until you clear it.
- Switching weights also switches the [settings profile](#settings-profiles).
- After starting, run the [smoke and exactness checks](#is-it-ready-is-it-right).

### Settings profiles

Each weight variant ships its own measured settings profile. The files are `config/profiles/base.env` and `config/profiles/ablit.env`. A profile holds the variant's two drafting settings: `SERVE_DRAFT_POLICY`, used with the draft model, and `NO_DRAFTER_POLICY`, used with the built-in head when you run without it. Every other engine setting is shared by both variants, in `config/serve.env`.

- `serve.sh` loads the profile that matches `WEIGHTS` (or `--weights`), so switching weights switches settings too. You pass no extra option.
- `DRAFTER` picks which of the profile's two drafting settings is used ([below](#draft-model-dflash2-or-none)).
- `serve.sh` checks the profile before any container starts and stops on a value a profile may not hold. It records the profile's sha256 on the container.
- The published numbers were measured with these files as shipped. The release tooling writes them; do not edit them by hand. If a profile differs from the shipped file, `serve.sh` warns that this is not a measured configuration, and `scripts/status.sh --identity` reports the difference. Run the [smoke and exactness checks](#is-it-ready-is-it-right), and measure speed yourself.
- To get a shipped profile back: `git checkout config/profiles/`.

### Weights: base or ablit

**base** (the default) is the public `TensorFold/GLM-5.3-Flash-MLX-4bit-MTP` weights (formerly `Vontra/GLM-5.3-Flash-MLX-4bit-MTP`, which redirects; MIT) at a pinned revision. INSTALL fetches, verifies and splits them. This project hosts none of the v2.0.2 weights. Download size: 181,741,759,037 bytes (181.7 GB; 54 files) for the weights; with the 2,342,460,697-byte draft model the download is 184,084,219,734 bytes (184.1 GB).

Weights the scripts split or convert from (`$DATA/base/weights`, `$DATA/ablit/source`) are read inside the container through `DATA`. If you already have them in another folder, move them into `DATA`, or copy them with `cp -al` (hard links: no extra disk, same filesystem only). Never symlink them: inside the container the link leads nowhere ([INSTALL.md](../INSTALL.md), step 5). Where a link is fine, link only the folders INSTALL's "Already have the files?" table names (for example `$DATA/base/rank<R>`), never a parent such as `DATA` or `$DATA/base`. Each check removes the `.verified` marker beside the folder it checks and writes it again only when every file matches. Once its space check passes, a later split or conversion replaces its own output folders. Through a linked parent, both would happen inside your original copy.

**ablit** is an opt-in install for ablit development, red-teaming and refusal research. It is refusal-removed (abliterated) weights from `orcarouter/GLM-5.3-Flash-Uncensored-MLX` at a pinned revision. The installer converts them on your own machine; nobody hosts the converted weights. License: MIT (Copyright (c) 2026 Z.AI Co., Ltd), plus the use conditions on the source model card, quoted below:

> - It is released **strictly for legitimate research** — interpretability, AI-safety and refusal-mechanism study, red-teaming, robustness evaluation, and controlled experiments.
> - **You assume full responsibility and liability** for how you use it and for everything it generates. Do not deploy it to end users or in production without adding your own safety, moderation, and abuse-prevention layers.
>
> By downloading or using this model you acknowledge and accept the above.

You are responsible for complying with the source model's terms and for how you use this model and what it generates. JSpark3 provides conversion tooling only; it hosts none of these weights and does not endorse any use of them.

The source repository is gated. To get access:

1. Sign in to Hugging Face with your own account.
2. Open the source's page and accept its terms.
3. Create an access token with read permission.
4. Give the token to the scripts without leaving it in your shell history:

   ```sh
   read -rs HF_TOKEN && export HF_TOKEN    # paste the token, press Enter; nothing is echoed or saved
   ```

   Never put the token in `cluster.env`, a script or a log you share. The public downloads (base weights and the draft model) run without your token, even if `HF_TOKEN` is set. With ablit selected, `fetch-weights.sh` checks your token and your access to the gated source before it downloads anything, so a missing or refused token stops it before any download. `--verify-only` needs no token. Every download uses the Hugging Face CLI at the version this release pins (INSTALL, "What you need"); `fetch-weights.sh` refuses any other version ([TROUBLESHOOTING.md](TROUBLESHOOTING.md#the-hugging-face-cli)).

Then fetch and convert ablit as [INSTALL.md](../INSTALL.md#ablit-weights-opt-in) describes. The conversion writes all three thirds, so it needs to run on one box only. In short, on that box:

```sh
scripts/fetch-weights.sh --weights ablit     # checks your access first, then base weights and the gated source, each checked by sha256
scripts/convert-ablit.sh                     # converts without network; checks the weights and all three thirds
```

Copy `$DATA/ablit/rank1` to rank 1's `$DATA/ablit/`, and `$DATA/ablit/rank2` to rank 2's. On each of those two boxes:

```sh
scripts/fetch-weights.sh --drafter-only          # the draft model only
scripts/split.sh --weights ablit --verify-only   # reads every file of the copied third and checks it against its manifest
```

If `WEIGHTS=ablit` is already set in that box's `cluster.env`, `--drafter-only` first checks your ablit access, so it needs `HF_TOKEN` there too. It still downloads only the draft model. To fetch the draft model without a token, run `scripts/fetch-weights.sh --weights base --drafter-only` instead.

You can instead run `fetch-weights.sh --weights ablit` and `convert-ablit.sh` on every box; each box then converts on its own.

- **Conversion cost.** Measured on one DGX Spark: the 200.1 GB source download took about an hour on our connection. Converting, splitting and checking then took about 26 minutes of processing, used no GPU and under 4 GiB of process memory, and needed about 371 GB of free disk beyond the downloaded source (about 571 GB in all), on top of the base weights you already installed. `scripts/convert-ablit.sh` checks for about 400 GB free before it starts.
- The pinned base weights, TensorFold/GLM-5.3-Flash-MLX-4bit-MTP at revision 76add2a341a1cd90ad0e86bb69839ea9c35827c6 (MIT), supply the four-bit tensor layout and the native prediction layer that the conversion restores. The chat template is the base checkpoint's MIT template plus six lines added by this recipe, also under MIT; its reconstructed bytes are hash-checked. Keep both the base weights and the ablit source until `convert-ablit.sh` reports `ablit weights and all three thirds built and verified`.
- On your own machine, `scripts/convert-ablit.sh` runs the pinned conversion scripts in `scripts/ablit/` inside the release's pinned container image, with no network and no GPU, then writes and checks the three per-host parts. It keeps its receipts, input notices and identity record in `$DATA/ablit/work`, and puts `WEIGHTS-LICENSE.txt` and `TEMPLATE-ADDITIONS-LICENSE.txt` beside the converted weights. Keep those two files with anything you derive from these weights.
- To serve ablit, set `WEIGHTS=ablit` on all three boxes and start. `serve.sh` loads the ablit settings profile with it.
- Ablit has its own published numbers (the "ablit weights + draft model" set in [BENCHMARKS.md](BENCHMARKS.md)).
- To go back to base, set `WEIGHTS=base` on all three boxes. The base split stays on disk unless you deleted it.

### Draft model: dflash2 or none

The draft model makes decoding faster by proposing several tokens at a time, which the main model then checks. A draft only changes speed, never the answer (`exactness.py` checks this).

- **dflash2** (the default) uses `incoai/GLM-5.3-Flash-DFlash2`. Its license is CC BY-NC-ND 4.0, **non-commercial**. INSTALL downloads it unmodified; this project never redistributes it.
- **none** drafts with the weights' own built-in multi-token prediction head instead. No non-commercial draft model is involved, so this is the mode for commercial use. Check the other components' licenses in the README too. It was checked on the GPU with the base weights: drafting with the prediction head gave the same tokens, text and finish reasons as plain one-token-at-a-time decoding on every check prompt.

To switch to `none`, set `DRAFTER=none` in `cluster.env` on all three boxes and start. To try it for one start, pass `scripts/serve.sh --drafter none` on all three boxes instead. `serve.sh` then starts the engine with no draft model, drafting with the built-in head at the [settings profile](#settings-profiles)'s `NO_DRAFTER_POLICY`. You set nothing else. With `dflash2`, the profile's `SERVE_DRAFT_POLICY` is used.

Two lines in the engine's own files are out of date and remain outside the scope of v2.0.2. `engine/NOTICE` says to select `--drafter none --draft-policy c7:0.3`. Don't add `--draft-policy` yourself: `scripts/serve.sh --drafter none` (or `DRAFTER=none` in `cluster.env`) applies the value from the weights' own settings profile, and the default base weights use `c7:0.45`. `engine/README.md` points at a RELEASE-FACTS.md file that does not ship; the measured results are in the README's Results section, `docs/BENCHMARKS.md` and `release/MEASUREMENTS-v2.0.1.md`.

- Without the draft model, a long conversation that includes images may not be saved to the disk session cache, and each saved state takes more memory, so fewer long conversations stay cached. Returning to such a conversation after it has left the memory cache can take as long as its first prompt. Text-only conversations of about 40,000 tokens are saved; longer text-only conversations were not tested. [TROUBLESHOOTING.md](TROUBLESHOOTING.md#a-long-conversations-next-reply-takes-as-long-as-the-first) has what to do for now.

To switch back to `dflash2`, first fetch the draft model if you skipped it at install: `scripts/fetch-weights.sh --drafter dflash2`.

Speed without the draft model is published separately, as the "base weights, no draft model" set in [BENCHMARKS.md](BENCHMARKS.md).

## Upgrading from v1.8.x, and rolling back to v1.8.4

[UPGRADING.md](../UPGRADING.md) has the full steps. Read it first. In short:

- **v2.0.2 is a separate install.** The candidate uses its own checkout (branch `release/v2.0.2`) with its own `DATA` folder. Its weights are new public 4-bit weights, not the v1.8.x files, and nothing from v1.8.x is reused or converted.
- **Only one release runs at a time.** Each release uses nearly all of each box's memory.

v1.8.4 stays available; see rolling back. With base weights, two measured cases favour it (three DGX Sparks; v1.8.4 at its default, reasoning off, and v2.0.1 at reasoning effort low; an appliance comparison with different model IDs, not a same-weights claim). On prose replies, v1.8.4 shows the first visible text about 0.1 s sooner, because v2.0.1 writes a short reasoning passage first (known issue 9). With base weights and a single client on short code replies, the first visible text arrives in about the same time, with v1.8.4 about 2% faster. If you keep very many idle keep-alive clients connected, read known issue 13 first; that fix is outside this release; no target version is assigned.

Reasoning is now always on. v1.8.4 had it off by default and honoured requests to turn it off; v2.0.2 runs a request with no reasoning setting at High and treats a request to turn it off as low (known issue 9). Replies begin with a reasoning passage, returned as `reasoning_content`, before the visible text, and it uses part of `max_tokens`.

**Upgrade:**

1. Stop v1.8.x with its own stop command, from its own checkout (v1.8.4: OPERATIONS.md, "Stop, restart and upgrade").
2. Install v2.0.2 by following [INSTALL.md](../INSTALL.md).

**Clients need changes too:**

| | v1.8.4 | v2.0.2 |
|---|---|---|
| Base URL | port 8888, all interfaces | `http://127.0.0.1:8002/v1`, rank 0 loopback only |
| Model id | `glm-5.3-flash` | `glm53` (the field is not checked; any value reaches the one model) |
| Thinking | off unless the request turns it on | **always on**: High effort by default, Low on request |
| JSON output | `response_format` with `json_schema` | `response_format` is ignored; ask for JSON in the prompt and validate the reply |

To keep port 8888 for existing clients, set `API_PORT=8888` in rank 0's `cluster.env`. The server still listens on loopback only.

- **Thinking:** the model writes its reasoning first, in `reasoning_content`, and the answer goes in `content`. The reasoning is never folded into `content`. This release has no thinking-off mode.
  - Requests that set no reasoning effort run at high effort; set `"reasoning_effort": "low"` for shorter replies.
  - `"reasoning_effort": "low"` (or `"minimal"`, or `"none"`, or `"chat_template_kwargs": {"enable_thinking": false}`) thinks at Low effort. The reply still has a reasoning section.
  - `"medium"` and `"high"` give High. `"xhigh"` and `"max"` give Max.
  - `usage.completion_tokens` counts every output token, reasoning included. There is no separate count of reasoning tokens.
  - A small `max_tokens` can run out during the reasoning and return an empty `content` with `finish_reason: "length"`. Leave room in `max_tokens`, or ask for Low effort. Even at low effort, a small `max_tokens` can be used up by reasoning and return no visible text; allow a few hundred tokens or more.
  - Clients that expected v1.8.4's no-thinking replies should send `"reasoning_effort": "low"` and read only `content`.
- **Ignored fields:** v2.0.2 accepts `stop`, `n`, `logprobs`, presence and frequency penalties, and `logit_bias`, then ignores them without an error. A client that relies on `stop` must trim the reply itself.
- **`seed`:** identical prompts without a `seed` return identical outputs, even above temperature 0. Send a different `seed` with each request when you want a different sample.
- **Context window:** 262,144 tokens. Longer prompts get HTTP 400 `context_length_exceeded`.
- **Cached prompt tokens:** v1.8.4 returned `usage.prompt_tokens_details.cached_tokens`; v2.0.2 does not (known issue 15 in [LIMITATIONS.md](../LIMITATIONS.md)). Read the reply's `tensorfold.cached` field instead, in the final chunk when streaming.
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md#api-behaviour-clients-notice) lists the full set of API differences.

**Roll back to v1.8.4:**

1. Run `scripts/stop.sh` on all three boxes.
2. Get v1.8.4 at its release tag. In your v1.8.4 checkout, run `git fetch --tags && git checkout v1.8.4`. If you no longer have one, run `git clone --branch v1.8.4 https://github.com/jakejharris/jspark3.git jspark3-v1.8.4`.
3. Start v1.8.4 from that checkout, following its INSTALL and OPERATIONS.
4. Point clients back at port 8888 and model `glm-5.3-flash`.

- v1.8.4 listens on all network interfaces, not loopback. Put its firewall rules back in place before you start it.
- Rollback leaves v2.0.2's data untouched. Coming back later is a normal [start](#start).
- **Keep the v1.8.4 checkout and weights if you have the disk space.** Otherwise a rollback means downloading the v1.8.4 weights again. v1.8.4 needs roughly 164 GiB of weights, 2.34 GB of draft weights and a 21 GB image per Spark, plus build layers and caches (v1.8.4 INSTALL).

## Host changes

### Reboot

After a reboot:

- Docker keeps the stopped containers after a reboot, but it does not restart them. `serve.sh` refuses to start while a container from an earlier start still exists, so remove it first.
- The memory settings from `host-prep.sh` do not survive a reboot.

On each box:

1. If you need the previous container's log, save it first: `docker logs jspark3-rank<R> > rank<R>.log 2>&1`.
2. `scripts/stop.sh` removes the stopped container, and its log with it.
3. `scripts/host-prep.sh`
4. `python3 scripts/preflight.py --for serve`
5. [Start](#start) in order: rank 2, rank 1, then rank 0.

Warm starts are quick because the kernel cache is on disk.

A desktop session holds memory the engine could otherwise use, and neither INSTALL nor `host-prep.sh` changes it. If you don't use the desktop on these boxes, you can boot them to text mode with `sudo systemctl set-default multi-user.target`. Undo it with `sudo systemctl set-default graphical.target`.

If you set `PREV_PEER` and `NEXT_PEER` in `cluster.env`, preflight checks each ring cable end to end with large-packet pings. Set them: those pings catch a cable that came up wrong after a reboot.

### Address or interface change

Only these `cluster.env` lines hold addresses or interface names:

- `MASTER_ADDR`: rank 0's LAN address, on all three boxes.
- `PREV_PEER` and `NEXT_PEER`: optional cable addresses.
- `LAN_IFACE`, `PREV_IFACE` and `NEXT_IFACE`: interface names. An OS or firmware update can rename these.

To make the change:

1. Stop all three ranks.
2. Edit `cluster.env` on every box that changed.
3. Run preflight on each box.
4. Start in order.

Point your SSH tunnels and proxies at rank 0's new address.

### Recabling

The ring cables must run in one direction: each box's `NEXT_IFACE` port connects to the next rank's `PREV_IFACE` port (0 to 1, 1 to 2, 2 to 0).

- If you swap cables, either swap them back, or swap `PREV_IFACE` and `NEXT_IFACE` in the affected boxes' `cluster.env` so the names match the cables again.
- Then run preflight and the [checks](#is-it-ready-is-it-right).

### Disk full

Stop the engine before you delete anything under `$DATA`. Never delete the split you are serving.

What uses space, and what it costs to free it:

| What | Where | Grows? | Free it with | Cost of freeing |
|---|---|---|---|---|
| Session store | `$DATA/sessions/` | up to its cap per folder | `scripts/clear-sessions.sh --all` | returning conversations are slow once |
| A weights variant you no longer serve | `$DATA/<variant>/`, and for ablit also `$DATA/ablit/source` | no | delete the folder | fetch (and for ablit, convert), then split again to switch back |
| Full base download, after `split.sh` has verified the split | `$DATA/base/weights` (only on the box that downloaded it) | no | delete the folder | a new download to split again. Keep it while you still plan to convert ablit, which reads it |
| Kernel cache | `$DATA/kernel-cache` | slowly | delete the folder | the next start recompiles for several minutes |
| Script logs | `logs/` in the checkout | yes | delete old files | none |
| Docker container logs | Docker's data folder | yes, while the container runs (Docker's default settings, no rotation) | removed with the container by a plain `stop.sh` (`stop.sh --keep` keeps both); to cap them, set log rotation in Docker's own settings | none |
| Unused Docker images | Docker's data folder | no | `docker image rm <image>`; keep the pinned image | pulling it again |

Check space with `df -h "$DATA"` before starting. With the session store on (the default), preflight warns when `DATA` has less free space than the store's budget plus 150 GiB. That is only a warning, and it does not guarantee room for every write a start makes. While serving, a nearly full disk only stops new session-store saves (see [the session store](#the-session-store-what-it-keeps-and-how-to-clear-it)).

### Other work on the boxes

The engine takes its memory once, at start, from what is free at that moment. Then it keeps it.

- If another program holds a lot of memory at start, the engine gets less room or fails to start. That includes another model server, a desktop session or a large file copy.
- Starting other GPU work while the engine serves competes with it for the same memory.
- Keep the three boxes for this server while it runs. Run `host-prep.sh` before a start if anything else ran since boot.

### OS, driver, Docker or firmware updates

1. Update with the engine stopped.
2. Then run preflight, start, and the [checks](#is-it-ready-is-it-right).

The container image is pinned by digest, so a Docker update does not change the image. A driver or firmware update can change behaviour, which the checks catch.

## Reaching the API safely

The API on rank 0 accepts any request from any program that can reach `127.0.0.1:8002`. Keep it that way.

**From your laptop, an SSH tunnel:**

```sh
ssh -N -L 8002:127.0.0.1:8002 <you>@<rank 0 address>
# then use http://127.0.0.1:8002/v1 on the laptop
```

**For other users or services**, put an authenticating reverse proxy on rank 0, in front of `127.0.0.1:8002`. If those users must not share cached prompts, also read [exact repeats, and other clients](#the-session-store-what-it-keeps-and-how-to-clear-it). Give the proxy:

- authentication;
- TLS;
- a rate or concurrency limit (the engine has none);
- read and idle timeouts long enough for long replies;
- response buffering turned off for streaming.

**Web pages in a browser** cannot call the API directly. It sends no CORS headers and answers `OPTIONS` with 501. Call it from a server, or add CORS at your proxy.

## Images in requests

Both weight variants accept images in chat messages as inline `data:` URLs (base64). Remote image URLs are refused. Each request takes up to 16 images, at most 32 MB per image and 32 MB in total, and at most 32 megapixels per image. This release does not measure how well the model understands images.

- Send each image as a content part of a user message: `{"type": "image_url", "image_url": {"url": "data:image/png;base64,<the file in base64>"}}`. JPEG, PNG, WebP and GIF files are read. Animated inputs use frame zero as a still image; later frames are not interpreted.
- **At most 4 requests with images at a time,** counting those being prepared, waiting and answered. A fifth is refused with HTTP 503, not queued, so retry it. Requests without images don't count toward this limit.
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md#images-in-requests) lists the messages a refused image gets.

## Capacity

- **8 requests at a time.** A ninth waits in a queue until a slot frees. Nothing returns HTTP 429; overload shows up as waiting. Requests with images have their own limit and are refused, not queued, past it ([Images in requests](#images-in-requests)). Clients with short timeouts may give up while queued, so set your proxy's concurrency limit with that in mind.
- **Requests that arrive together.** With the draft model on, short text prompts that arrive together while no reply is streaming are read together in one pass, as many as fit in one combined read. Without the draft model, prompts that arrive together are not read together in one batch, and first tokens arrive later. A prompt that arrives while replies are streaming, unless it is very short, is read in small steps between reply steps, so those replies keep streaming while it is read, with occasional pauses. The measured effects are in [request scheduling limitations](TROUBLESHOOTING.md#api-behaviour-clients-notice).
  - To make such a pass, a short text prompt that arrives while no reply is streaming may wait up to 25 ms for a second prompt. It waits only when the draft model is on, a slot is free, every waiting prompt is a short text prompt, and the waiting prompts fit in one combined read. Long prompts never wait. The limit is `TF_GLM_EXPRESS_GATHER_MS` in `config/serve.env`, and the wait counts toward the request's time to first token.
- **Shared context memory.** All running requests share one pool of context memory. A request near the full context window can wait in the queue until other long requests finish.
- **Use streaming for long replies.** A non-streaming request sends nothing until the whole reply is done, and proxies with short idle timeouts cut those connections. With `"stream": true`, tokens arrive as they are made.
- **Request size limits.** Bodies over 192 MiB get 413. Too much request data in flight at once gets 503. A body that arrives too slowly gets 408.
- **Monitoring.** `/metrics` on rank 0 reports running and waiting requests (`vllm:num_requests_running`, `vllm:num_requests_waiting`). Request logging is quiet: the engine does not log each request, so collect request logs at your proxy.

## Logs, and what to share

- **Engine log** (each box): `docker logs jspark3-rank<R>`. If you set `CONTAINER_PREFIX`, use it in place of `jspark3`.
  - At start it prints a `[fabric]` line with the ring settings the rank used. Check it first when ranks fail to connect.
  - The log goes away when `stop.sh` removes the container; `stop.sh --keep` keeps the container and its log. Save it first if you need it: `docker logs jspark3-rank<R> > rank<R>.log 2>&1`.
- **Script logs**: `logs/<script>.log` in the checkout, readable only by you (mode 0600).
  - On screen, script messages name settings and paths relative to `$DATA`, never their values.
  - The log files hold the detail: values from `cluster.env`, full paths and command output.
  - `fetch-weights.sh` never writes your Hugging Face token to the screen or its log. The release's `tests/check-token-canary.py` checks this.

Before you post a log in an issue or a chat, remove:

- host names, IP and MAC addresses;
- user names and home paths;
- any token (an `hf_...` string);
- anything from your users' prompts.

Never share `cluster.env` as it is. Never share files from `$DATA/sessions`.

## Terms

- **Box:** one DGX Spark. **Rank:** the part of the model a box runs, 0, 1 or 2. Rank 0 also serves the API.
- **`DATA`:** the folder on each box that holds weights, kernel caches and the session store. Set in `cluster.env`.
- **Preflight:** `scripts/preflight.py`. It checks a box before a step and refuses with a reason. It changes nothing on the box, except that the serve check starts two short test containers from the pinned image (it never pulls the image), and Docker removes them when they finish.
- **Session store** (the "session tier" in script messages): conversation state kept on disk so a returning conversation skips most of its prompt work. It can be turned off.
- **Draft model:** a small model that guesses the next few tokens for the main model to check, so replies come faster. `DRAFTER=none` (or `--drafter none`) runs without it. `scripts/exactness.py` checks that it does not change greedy replies.
- **Settings profile:** the file of drafting settings (`config/profiles/<weights>.env`) that `serve.sh` loads for the weights you serve. Each weight variant ships its own measured settings profile.
- **Base and ablit weights:** base is the default. Ablit is the opt-in, refusal-removed variant ([INSTALL.md](../INSTALL.md#ablit-weights-opt-in)).
- **Reasoning effort:** Low, High or Max. The model always reasons first; there is no off setting.
- **Smoke and exactness:** the two answer checks to run after any change ([TROUBLESHOOTING.md](TROUBLESHOOTING.md#checking-answers-smoke-and-exactness)).
