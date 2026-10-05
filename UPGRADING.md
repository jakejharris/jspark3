# Upgrading to JSpark3 v2.0.2

JSpark3 v2.0.2 fixes image-history checkpoint reuse and adds GIF frame-zero support.
See [the release gate](RELEASE-GATE.md) and [image-fix evidence](release/v2.0.2/ROOTCAUSE.md).

## From v2.0.1

**All three ranks must run the same release.** Stop all three old ranks before starting any v2.0.2 rank; mixed-version rings are unsupported.

The engine changes only in the image checkpoint and GIF decoder fixes. Weights, draft policies, native CUDA
sources and general checkpoint settings stay at v2.0.1. Use a separate checkout and DATA directory, retain the
v2.0.1 containers and data for rollback, and verify any copied weights using the existing install checks.
Build the v2.0.2 wheel with `scripts/build-wheel.sh`; its content must match `pins.env`.

The session folder and arithmetic identity include the v2.0.2 wheel digest. Do not copy old session state into
that folder. The first request is cold; useful image-boundary checkpoints must be saved before later rotations
can reuse them. A cache fix revision gets a new digest and a new session folder.

During an approved serving window, stop the old ranks, start the v2.0.2 ranks 2, 1, 0, then run readiness,
identity and smoke checks plus the live GIF and rolling-image checks. To roll back, stop v2.0.2 and
restart the retained v2.0.1 ranks 2, 1, 0 with their original paths, then verify readiness and smoke again.
Do not delete either release's sessions as part of this upgrade.

## From v1.8.x, and going back to v1.8.4

v2.0.2 is a new installation, not an in-place upgrade. The engine changes from vLLM to a fork of TensorFold 0.3.6.2
(MIT), and the weights change from the v1.8.x EXL3 files to public 4-bit MLX-format weights split across the three
hosts. Stop v1.8.x before you start v2.0.2, and keep your v1.8.4 checkout and weights if you might roll back.

v1.8.4 stays available; see rolling back. With base weights, two measured cases favour it (three DGX Sparks; v1.8.4 at its default, reasoning off, and v2.0.1 at reasoning effort low; an appliance comparison with different model IDs, not a same-weights claim). On prose replies, v1.8.4 shows the first visible text about 0.1 s sooner, because v2.0.1 writes a short reasoning passage first (known issue 9). With base weights and a single client on short code replies, the first visible text arrives in about the same time, with v1.8.4 about 2% faster. If you keep very many idle keep-alive clients connected, read known issue 13 first; that fix is outside this release; no target version is assigned.

Reasoning is now always on. v1.8.4 had it off by default and honoured requests to turn it off; v2.0.2 runs a request with no reasoning setting at High and treats a request to turn it off as low (known issue 9). Replies begin with a reasoning passage, returned as `reasoning_content`, before the visible text, and it uses part of `max_tokens`.

## What changes for clients

| | v1.8.x | v2.0.2 |
|---|---|---|
| API address | rank 0, port 8888 | rank 0, port 8002, loopback by default (`API_PORT`, `API_HOST` in `cluster.env`) |
| Model name listed | `glm-5.3-flash` | `glm53` (the `model` field of a request is not checked, so old clients keep working) |
| Known differences | v1.8.x notes | [LIMITATIONS.md](LIMITATIONS.md) |

To keep clients unchanged, set `API_PORT=8888` in rank 0's `cluster.env`.

If your client reads `usage.prompt_tokens_details.cached_tokens`, read `tensorfold.cached` instead (known issue 15 in
[LIMITATIONS.md](LIMITATIONS.md)); when streaming, it is in the final chunk.

v1.8.4's engine honored the `cache_salt` request field; in v2.0.2 it is ignored (known issue 16 in
[LIMITATIONS.md](LIMITATIONS.md)).

## Upgrade

1. **Stop v1.8.x** with its own controller, from the v1.8.x checkout, as its operations guide describes. You may
   keep its stopped containers for a rollback: v2.0.2's scripts never stop, remove or reuse a container they did not
   start, and refuse a name clash ([INSTALL.md](INSTALL.md#other-containers-on-the-box); `CONTAINER_PREFIX`).
2. **Check that the GPUs and ports are free**: `nvidia-smi` shows no serving process on any box, and nothing listens
   on v2.0.2's ports. `python3 scripts/preflight.py` in v2.0.2 checks the ports and its own container name.
3. **Install v2.0.2 in its own folders**: a new checkout and a new `DATA` folder ([INSTALL.md](INSTALL.md)). It
   shares nothing with v1.8.x: no weights, images, caches or configuration are reused. Check the free disk first;
   both releases' weights side by side need room for both.
4. **Start v2.0.2** (ranks 2 and 1, then 0) and run `scripts/smoke.sh`.

## Going back to v1.8.4

1. On each box, stop v2.0.2: `scripts/stop.sh`.
2. Get v1.8.4 at its tag, in your v1.8.4 checkout:

   ```bash
   git fetch --tags && git checkout v1.8.4
   ```

   or as a fresh copy:

   ```bash
   git clone --branch v1.8.4 https://github.com/jakejharris/jspark3.git jspark3-v1.8.4
   ```

   Then start v1.8.4 from that checkout with its own controller, through its preflight and qualification, as its
   install guide describes. Nothing in v2.0.2's folders is needed.
3. When you no longer need v2.0.2, and with all three boxes stopped, delete its `DATA` folder with `sudo rm -rf` (the
   containers write some of its files as root); `scripts/clear-sessions.sh --all` first if you want the session tier
   gone before the rest.

## Upgrading to a later v2.0.x

Stop all three boxes, update the checkout, and run the install steps again; each step skips what is already
verified. The session tier's folder is named after the engine build, so a new engine starts a new folder and never
reads the old one; `scripts/clear-sessions.sh --all` removes old ones.
