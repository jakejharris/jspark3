# Image checkpoint and GIF fixes: v2.0.2 evidence

**Cache root cause: VERIFIED offline. Live cache validation: failed on the first attempt; retry pending.**
**GIF: offline HTTP reproduction changed from 400 to 200; the first live GIF check passed.**
This is a sanitized summary of the investigation's `ROOTCAUSE.md`, not a new benchmark.

## Image-history checkpoint

The client keeps its newest eight images and replaces older image blocks with archive text. Its OpenAI
converter moves tool images into a following user message, so rotation changes the preceding tool message
too. A checkpoint at the image token is already too late. With general intermediate checkpoints disabled,
the available valid state can remain an old text-only prompt. Exact prompt and full image-digest matching
correctly reject state after changed input.

Fix `bd7a45f0b2c7b4a03e08d23563a98a3d5cc673ea` retains one intermediate checkpoint before the
assistant/tool turn containing the oldest remaining image. It uses the existing persistence and rank
coordination paths. General dense checkpoints remain off, and matching rules are unchanged.

The CPU regression exercises sliced and unsliced prefill, memory cache on and off, target and draft state,
and disk restore across four image rotations. Synthetic resume positions advance `0 -> 5 -> 13 -> 21`;
the next retained anchor is 29. Restored state hashes and outputs equal fresh prefill. These fixture
positions are not production throughput measurements or proof of CUDA equivalence.

## GIF decoder

The decoder allowed JPEG, PNG and WEBP but rejected GIF. The supplied valid palette-mode animation
repeatedly failed when replayed in image history. Fix `258d1e839e62a759808eb15b8938019cadcce4dd` allows
GIF, explicitly decodes frame zero, preserves transparency compositing, and retains size and error limits.
It interprets a still frame; it does not interpret motion.

The supplied-file HTTP-handler reproduction used in-memory streams and stubbed generation, without a
model call. It returned 400 before the fix and 200 after it at both image-history positions 8 and 7.
The prepared GIF tensor equaled a PNG of frame zero. Corrupt GIFs still return a numbered 400.

## Offline verification

The original before check had 11 failing and 2 passing regressions. The relevant patched suite passed
305 tests. This release branch independently repeated that suite: **305 passed**. It covers image
preparation and lifetime, request errors, session disk chains, writers, exact replay and rolling images.
It uses deterministic CPU arithmetic, not a live serving benchmark.

## First live swap, 2026-10-05

- The swap started at **17:37:27Z**. Readiness passed at 17:43:42Z, then smoke passed at 17:43:51Z.
- The supplied GIF completed successfully and identified the pictured multi-tiered pagoda.
- The eight-image request had 510 prompt tokens and **resumed 0 tokens**. It returned a tool call
  instead of the expected newest-image color. The check failed; the nine- and ten-image checks did not run.
- The swap exited unsuccessfully at 17:43:54Z and was rolled back. The original v2.0.1 ring
  was restored, with identity checks and smoke passing. No successful advancing-cache-boundary check is claimed.

The first cold request's zero reuse alone does not diagnose the live failure. The unsuccessful check
provides no live proof of the cache fix. Diagnosis may revise or replace that commit. **The PR stays draft
until a live retry passes**, including advancing restored boundaries after image rotation.

## Savings estimates from existing logs

These are **log estimates, not measurements of this fix**:

| Capture | Gross estimated avoided prefill | Basis |
|---|---:|---|
| Run 4 | 762.9 seconds, about 12.7 minutes | 26 affected requests; 1,750.6 summed server seconds, about 29 minutes |
| Run 3 | 227.0 seconds, about 3.8 minutes | 13 affected requests; concurrent request time is not wall-clock duration |

The calculation is observed prefill time multiplied by the newly reusable fraction of the prompt suffix.
It assumes useful checkpoints were written and remain available. It does not measure checkpoint overhead,
disk restore time, image-tower time, IO pressure or changed suffix throughput. The first request after an
upgrade is cold. No new speed or storage claim follows from these estimates.

## Retained source evidence

The full investigation contains private operational context and is retained by the maintainer. These
SHA-256 values identify the records used for this summary; only sanitized findings are shipped here.

| Investigation record | SHA-256 |
|---|---|
| `ROOTCAUSE.md` | `e095f89d0db36270741baaedc15c474d7b1cd00dcf7ecefd29f4ed4f7165a6dd` |
| `runtime/pagoda-before.log` | `cb9154a6bf4f13ad8a791badd63ea98ca9887488d87a1a53f9f4047adfad22e4` |
| `runtime/pagoda-after.log` | `6cd1b598defc3936593478480143397d4bbce0b22d14fc9ffb7f765546e920ac` |
| `runtime/test-final.log` | `b8ae9ac3d20a055cc4a9d7bd78f3009b8d64170b24a3d3c2e7741a193877b839` |
| `prep/swap-state/check-outcome.json` | `880e7e3f3905c37ea751e20df0266302739bb8233c125e62c447dfd641c04655` |
| `prep/swap-state/swap-exit.json` | `09139068db95733d17ae5c912e2361cd379b2898e3b1c7738c96db957ca16f58` |
| `analysis/run3-savings.json` | `ef0c934c7aad40b7eb643444743bf8739db1863509fed8ff53d10fcf7c07a3b0` |
| `analysis/run4-savings.json` | `a864dc1e16dd8c6a53138574558b57f6067604a5ff7e608c4951326d661e76cf` |
| `prep/swap-state/rollback.log` | `80accab5bfecd039e82b743eeeede6edf2d0915c69a1413fb81de751712b1f64` |
