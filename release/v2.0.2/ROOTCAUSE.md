# Image checkpoint and GIF fixes: v2.0.2 evidence

**Cache root cause: VERIFIED offline. Live retry: PASS at 18:37Z on 2026-10-05, engine unchanged.**
**GIF: offline HTTP reproduction changed from 400 to 200; both live GIF checks passed. Smoke: 6/6.**
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
  was restored, with identity checks and smoke passing. That attempt did not reach the advancing-cache-boundary checks.

The first cold request's zero reuse was valid. The fixture supplied historical tool calls without a final
question closing that sequence; the model continued it with raw tool-call text. The unchanged v2.0.1
engine reproduced that answer, so it was not evidence of an image-cache regression. The revised fixture
marks the log complete and asks explicitly about the last image. It also waits for checkpoint durability
before testing reuse. No engine code, fix commit or wheel content changed.

## Passing live retry, 2026-10-05 at 18:37Z

| Observations | Prompt tokens | Answer | Resumed tokens | Cache source | New durable boundary |
|---:|---:|---|---:|---|---:|
| 8 | 549 | Blue | 0 | cold | 68 |
| 9 | 659 | Green | 68 | disk | 174 |
| 10 | 761 | Red | 174 | disk | 276 |

The newest eight images remained in each request; older images were replaced with archive text.
Both the boundary and full-prompt checkpoints were verified durable on all three ranks after every
request, including snapshot-key, header and token checksums. Readiness and runtime identity passed,
stock smoke passed **6/6**, and the supplied GIF correctly identified the pagoda without a tool call.
Acceptance passed at **18:37:38Z**, the swap exited zero, and the candidate was released for Pi traffic.

The same fixes are retained: `bd7a45f` / `258d1e8`, applied here as `e44eab4` / `9d376eb`.
Wheel content remains `35b0ccd7ee67e4f0a1b22db95566c8924425045552a8e08b3ca49454d3571e60`.
The [sanitized acceptance receipt](live-retry.json) binds these results to that wheel and the retained
source records. This proves the small CUDA sequence on the prepared runtime with matching engine content,
not a clean installation of the final public recipe or a long-context latency benchmark.

## Persistence and backpressure limit

The existing bounded writer refuses another optional snapshot batch while a write is pending. Its quiet
gate requires **0.5 seconds** without model work or request preparation. Continuous traffic without that
opportunity can therefore skip optional disk saves. The first attempt's cumulative count of nine was
real skipped snapshots, seven already accumulated before its rolling request; it was not nine new drops
from that last request. A CPU reproduction matched that behavior with no disk errors.

The successful retry waited for a stable idle store after smoke and for durability between rolling
requests. Its cumulative dropped-anchor counter was four after smoke and stayed four through the rolling
sequence and GIF: **zero additional drops in the rolling sequence**. This does not promise lossless
checkpoint persistence under sustained backpressure. The two fixes do not change that writer policy.

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
| `prep/RETRY-RESULT.md` | `c1033e2745c26891026a71737e2fd0977ecdc611c9a8fae375252382f16d00f7` |
| `retry/RESULT.md` | `f9c6650f6e9fabf93bcc28a8853bef1fa6a52b02a0093a02ea8a13c07ecf99e2` |
| `prep/retry-state/postboot-check.jsonl` | `7c72c75be695fa0b5976de3ee065b47b21b80213037bb58e12edc61f24f2adea` |
| `prep/retry-state/retry-exit.json` | `4473cf720be17a8d3d76601c59b33faa9341b1b4f917834319c5826603c575d5` |
| `prep/retry-state/retry-timed.jsonl` | `c7abf3efc8fd6ef55e0fc28ae10f0b0e91abf47593809d5ce8dccbb0f1baf330` |
| `prep/retry-state/final-verification.json` | `089ca7cc55699b340b49babfe0f528480a90b881c07dd1dae7b3204cf5644989` |
