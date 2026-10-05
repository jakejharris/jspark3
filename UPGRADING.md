> **Current release: v2.0.2.** Use the [v2.0.2 upgrade and rollback guide](https://github.com/jakejharris/jspark3/blob/v2.0.2/UPGRADING.md). The live image-cache acceptance passed on the prepared runtime. A clean installation of the final v2.0.2 public recipe has not been demonstrated; the installation and performance receipts remain v2.0.1 evidence.
> The v2.0.1 guide below is retained for historical installs and rollback.

# Upgrading from v1.8.x, and going back to v1.8.4

This maintained guide accompanies the v2.0.1 tagged recipe. See [INSTALL.md](INSTALL.md) for the historical v2.0.1
installation instructions; commands below run from the indicated release checkout.

v2.0.1 is a new installation, not an in-place upgrade. The engine changes from vLLM to a fork of TensorFold 0.3.6.2
(MIT), and the weights change from the v1.8.x EXL3 files to public 4-bit MLX-format weights split across the three
hosts. Stop v1.8.x before you start v2.0.1, and keep your v1.8.4 checkout and weights if you might roll back.

v1.8.4 stays available; see rolling back. With base weights, two measured cases favour it (three DGX Sparks; v1.8.4 at its default, reasoning off, and v2.0.1 at reasoning effort low; an appliance comparison with different model IDs, not a same-weights claim). On prose replies, v1.8.4 shows the first visible text about 0.1 s sooner, because v2.0.1 writes a short reasoning passage first (known issue 9). With base weights and a single client on short code replies, the first visible text arrives in about the same time, with v1.8.4 about 2% faster. If you keep very many idle keep-alive clients connected, read known issue 13 first; that fix is outside v2.0.2; no target version is assigned.

Reasoning is now always on. v1.8.4 had it off by default and honoured requests to turn it off; the server runs a request with no reasoning setting at High and treats a request to turn it off as low (known issue 9). Replies begin with a reasoning passage, returned as `reasoning_content`, before the visible text, and it uses part of `max_tokens`.

## What changes for clients

| | v1.8.x | v2.0.1 |
|---|---|---|
| API address | rank 0, port 8888 | rank 0, port 8002, loopback by default (`API_PORT`, `API_HOST` in `cluster.env`) |
| Model name listed | `glm-5.3-flash` | `glm53` (the `model` field of a request is not checked, so old clients keep working) |
| Known differences | v1.8.x notes | [LIMITATIONS.md](https://github.com/jakejharris/jspark3/blob/v2.0.1/LIMITATIONS.md) |

To keep clients unchanged, set `API_PORT=8888` in rank 0's `cluster.env`.

If your client reads `usage.prompt_tokens_details.cached_tokens`, read `tensorfold.cached` instead (known issue 15 in
[LIMITATIONS.md](https://github.com/jakejharris/jspark3/blob/v2.0.1/LIMITATIONS.md)); when streaming, it is in the final chunk.

v1.8.4's engine honored the `cache_salt` request field; in v2.0.1 it is ignored (known issue 16 in
[LIMITATIONS.md](https://github.com/jakejharris/jspark3/blob/v2.0.1/LIMITATIONS.md)).

## Upgrade

1. **Stop v1.8.x** with its own controller, from the v1.8.x checkout, as its operations guide describes. You may
   keep its stopped containers for a rollback: v2.0.1's scripts never stop, remove or reuse a container they did not
   start, and refuse a name clash ([INSTALL.md](INSTALL.md#other-containers-on-the-box); `CONTAINER_PREFIX`).
2. **Check that the GPUs and ports are free**: `nvidia-smi` shows no serving process on any box, and nothing listens
   on v2.0.1's ports. `python3 scripts/preflight.py` in v2.0.1 checks the ports and its own container name.
3. **Install v2.0.1 in its own folders**: a new checkout and a new `DATA` folder ([INSTALL.md](INSTALL.md)). It
   shares nothing with v1.8.x: no weights, images, caches or configuration are reused. Check the free disk first;
   both releases' weights side by side need room for both.
4. **Start v2.0.1** (ranks 2 and 1, then 0) and run `scripts/smoke.sh`.

## Going back to v1.8.4

1. On each box, stop v2.0.1: `scripts/stop.sh`.
2. Get v1.8.4 at its tag, in your v1.8.4 checkout:

   ```bash
   git fetch --tags && git checkout v1.8.4
   ```

   or as a fresh copy:

   ```bash
   git clone --branch v1.8.4 https://github.com/jakejharris/jspark3.git jspark3-v1.8.4
   ```

   **Weight source correction (2026-10-04):** the
   [v1.8.4 installation guide](https://github.com/jakejharris/jspark3/blob/v1.8.4/docs/INSTALL.md) still names
   `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`, which the boot walk found returning anonymous HTTP 401.
   Keep using your existing verified v1.8.4 weights if you have them. Otherwise, in step 4 of that guide replace
   only the Mia `hf download` command with the pinned
   [JSpark3 mirror](https://huggingface.co/jakejharris/jspark3/tree/e6cb0b09b3bf9f2ce35721426c570c8e714c5fc4):

   ```bash
   export HF_HUB_DISABLE_XET=1
   hf download jakejharris/jspark3 \
     --revision e6cb0b09b3bf9f2ce35721426c570c8e714c5fc4 \
     --local-dir "$JSPARK_MODEL_ROOT/Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw-25a44fdb"
   sha256sum "$JSPARK_MODEL_ROOT/Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw-25a44fdb/SHA256SUMS"
   # expected: cb0da1f97a53aebc3fbc5478f19c82b25586b0bf8533c99fb4ed5321a48f5342
   ```

   Set `JSPARK_MODEL_ROOT` as the guide instructs. Keep the directory name: v1.8.4's scripts expect it. Use that
   exact revision, not the mirror's `main`, where files and the checksum list have since moved or changed.
   Keep the guide's separate drafter download and remaining steps. Its `validate_checkpoint.py` must hash the
   files and report `"serving_checkpoint_pass": true`; the checksum-list check above alone does not verify them.
   This mirror supplies v1.8.4's EXL3 weights, not v2.0.1's MLX weights.

   Then start v1.8.4 from that checkout with its own controller, through its preflight and qualification, as the
   guide describes. Nothing in v2.0.1's folders is needed.
3. When you no longer need v2.0.1, and with all three boxes stopped, delete its `DATA` folder with `sudo rm -rf` (the
   containers write some of its files as root); `scripts/clear-sessions.sh --all` first if you want the session tier
   gone before the rest.

## Upgrading to a later v2.0.x

Stop all three boxes, update the checkout, and run the install steps again; each step skips what is already
verified. The session tier's folder is named after the engine build, so a new engine starts a new folder and never
reads the old one; `scripts/clear-sessions.sh --all` removes old ones.
