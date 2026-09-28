# Operations and admission

Keep ordinary traffic blocked through your firewall or gateway until the
operator admission receipt passes. The API listens without authentication on
rank 0, port 8888; these tools check admission but do not implement routing.
Run gates without other client traffic so request counters and latency belong
to the measured workload. A dry-run or `/health` response is not admission.

## Qualify one boot

From the validated source export, after [installation](INSTALL.md), run:

```sh
python3 -B tools/v16/qualify_runtime.py \
  --recipe "$JSPARK_RUNTIME/recipe" --env-file "$JSPARK_RUNTIME/operator.env" \
  --manifest "$JSPARK_RUNTIME/service.json" --output "$JSPARK_RUNTIME/qualification"
```

Use a new output directory for every attempt. The command performs these steps:

1. Bind exact container IDs, start times, environment, recipe, image and runtime
   epoch files. This default-profile path requires no runtime epoch overrides.
   Require unedited `production-stock`, ABLIT=0, APC-on and qualified coop-on
   (or an explicit coop-off diagnostic boot). Bind the component seal, native
   binary, measured policy and actual operator image into both receipts.
2. Run `fleetctl verify`, including native `verify_stock.py` **inside every
   container**, loader/capture checks, arithmetic, focused and long-context
   witnesses, and effective no-swap resource checks. The receipt must contain
   `status=VERIFY_PASS` and `production_stock.status=PASS`.
3. Run the first prefill pass. Its warmup seeds the required kernel shapes using
   actual requests; its performance result is recorded only. Quick is diagnostic.
4. Apply `posix_fadvise(DONTNEED)` to regular files in each bound container's
   model, recipe, Fly source, evidence and kernel-cache mounts. This reads files
   and advises page-cache eviction; it deletes nothing and never invokes global
   `drop_caches`. It runs as container root through existing Docker access so
   root-owned cache/receipt files are readable. Preserve each rank's roots,
   file/byte counts and errors. Any error blocks this automated path.
5. Run the post-hygiene prefill gate: its 1100 tok/s floor and canaries are
   blocking. Run prefix-cache parity/noise checks with `--expect finehit` (also
   the default). This matches the shipped 640-token reuse; historical `lru`
   expects whole 2560-token blocks and can falsely fail this runtime.
6. Verify again; inspect every rank's logs for compilation or engine errors
   since the first pass's warmup ended. Require the native TRIAR-inactive proof,
   unchanged boot/environment/epochs, and matching first/final receipt hashes.
   Write `admission.json` only after all steps pass.

The shipped producer supplies the formerly missing first-prompt and finalize
receipts. Recheck the saved admission decision with:

```sh
python3 -B tools/v16/admission_gate.py \
  --first-prompt "$JSPARK_RUNTIME/qualification/first-prompt.json" \
  --finalize "$JSPARK_RUNTIME/qualification/finalize.json" \
  --out "$JSPARK_RUNTIME/qualification/admission-recheck.json"
```

It checks matching configuration/boot identities and the retained evidence
hashes. A saved PASS applies to that observed boot, not a later restart or
changed configuration. A failed attempt leaves its evidence for diagnosis and
does not produce a new passing finalization. Re-run into a new directory after
fixing the cause; never relabel a failed post-hygiene result as a first pass.

Standalone diagnostics remain available from the source export:

```sh
python3 -B tools/v16/prefill_gate.py --base-url http://RANK0_ADDR:8888 \
  --env-file "$JSPARK_RUNTIME/operator.env" --out ../prefill-diagnostic.json
python3 -B tools/v16/apc_gate.py --base-url http://RANK0_ADDR:8888 \
  --env-file "$JSPARK_RUNTIME/operator.env" --expect finehit \
  --fixtures ../apc-fixtures.json --out ../apc-diagnostic.json
```

These isolated diagnostics do not replace the ordered, same-boot admission run.
No maintainer cache snapshot or private campaign driver is needed. The warmup
is the cache preparation; zero subsequent compilation remains a gate.

## TRIAR and runtime settings

Leave `JSPARK3_TRIAR=1` and `JSPARK3_TRIAR_P2P=auto` as prepared. The flag loads
TRIAR code; it does not prove active execution. In this pinned recipe, Cadence's
B4 capture hook supersedes the dual-bank hook and captures the original NCCL
banks. Admission reads the installed hashes, profile/serving graph census,
activation and adaptive-capture receipts, and graph dump hashes on every rank.
An ON epoch, partial/aliased graph bank, source drift or any dual-graph activity
refuses. The passing state is `INACTIVE_NCCL_B45`; no active TRIAR or thirds
performance is claimed, and no activation command is supported here.

[Final binding](../manifests/final-binding.json) records the historical v1.8.0
measured configuration, including coop on. The prepared operator environment
has coop off. Epoch changes and mode changes invalidate admission. Keep requests
drained and restart/requalify for source, image, weight mode, memory layout or
runtime-setting changes. Live stock/edited switching is not qualified.

## Stop, restart and upgrade

Keep the **old** prepared controller, env and service manifest until its
containers are removed. It verifies the exact IDs before stopping anything:

```sh
cd "$JSPARK_RUNTIME/recipe"
python3 -B scripts/fleetctl.py stop --env-file ../operator.env \
  --manifest ../service.json --confirm STOP-JSPARK3
```

This preserves containers and logs for inspection. After retaining needed
receipts/logs, remove only that manifest's containers:

```sh
python3 -B scripts/fleetctl.py stop --env-file ../operator.env --manifest ../service.json \
  --confirm STOP-JSPARK3 --remove --remove-confirm REMOVE-JSPARK3
mv ../service.json ../service-stopped.json
```

Archive the old receipts, then build/stage the new release in separate recipe
and work directories. Run new preflight, start, verify and qualification. The
container names are fixed; leaving old stopped containers causes preflight to
refuse. Do not restart old containers with stale capture/cache receipts, or
use a new release's controller to reinterpret an old manifest.

## Host changes and troubleshooting

Effective `memory.swap.max` must stay zero. Reverify after service-manager
changes; daemon reloads can reset limits. On hosts using snapd and runc 1.2.5,
hold automatic refreshes during the serving window with `sudo snap refresh
--hold`, record the change, and restore your update policy afterward. Check
`snap refresh --time`; this is host maintenance, not an admission waiver.

| Refusal | Action |
|---|---|
| Source privacy scan | Inspect the named file, including untracked files; keep private work/evidence outside the source. Git history does not affect this check. |
| `scripts/__pycache__` in recipe | Use a fresh prepared recipe; wrappers now disable bytecode writes. Do not edit checksum inventories. |
| Checkpoint serving-byte gate | Run the four-path `validate_checkpoint.py` command in INSTALL on the failing rank; its stderr identifies the missing view, ledger or shard. |
| Display host state | Make `full` hosts headless as INSTALL specifies, or choose `display0` and remove all DRM keys before a fresh start. |
| Coop seal/build-image drift | Stop admission; verify the release-pinned component seal and rebuild matching bytes. Never switch coop under captured graphs. |
| Readiness timeout | Inspect `status` and the retained rank log; raise `--ready-timeout` only for a still-loading healthy boot. |
| First prefill dip | Retain it; hygiene and the repeat decide. A repeat below 1100 still blocks admission. |
| Hygiene errors | Inspect per-rank paths/errors; fix missing mounts/read access and repeat the whole qualification into a new output directory. |
| APC fails in `lru` mode | Use `finehit`; it matches the shipped cache granularity. |
| Existing name/service manifest | Stop/remove with the old bound manifest, archive it, then start fresh. |
| TRIAR-inactive proof fails | Retain the proof/logs and keep admission closed; do not enable TRIAR or weaken the graph checks. |

## Terms

Admission is the decision to allow client traffic after gates pass.
Qualification is the measured gate run on one exact fleet boot. A receipt is a
local JSON evidence record with hashes, not a signed external certification.
Hygiene advises eviction of file-backed page cache after loading. Pi-shaped
requests model coding-agent sessions with long shared prefixes, tools and new
turns. Coop is the optional cooperative-MoE kernel. Adaptive-K selects draft
width; dense FP8 controls trunk computation. TRIAR is experimental collective
code retained inactive. `production-stock` uses unedited weights; `qa` is a
test profile and does not establish production admission.
