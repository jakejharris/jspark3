# JSpark3 v2.0.2 release candidate

**Ready for review after a passing live retry. Publication still waits for the release owner's approval.**

This candidate is v2.0.1 plus two engine fixes: retain a checkpoint before the oldest image's tool turn,
and accept GIF inputs by decoding frame zero. Weights, native kernels and draft policies are unchanged.

The GIF reproduction changed from HTTP 400 to 200 offline, and the first live GIF check passed. The cache
root cause is VERIFIED offline. The first live swap at 17:37Z on 2026-10-05 failed its check: the eight-image
request resumed 0 tokens and returned a tool call instead of the expected color. The fixture failure was
reproduced on v2.0.1, and the swap was rolled back. At 18:37Z the corrected fixture passed with the same
engine commits: 8/9/10-image resumes `0 -> 68 -> 174`, both nonzero hits from disk, GIF pass and smoke 6/6.
Boundary and full-prompt checkpoints were durable on all three ranks. Acceptance waited for durability;
continuous traffic can still skip optional saves under the existing bounded writer policy. This is not
lossless persistence under load or a long-context latency benchmark. See [ROOTCAUSE.md](v2.0.2/ROOTCAUSE.md).

There are no new performance measurements. Estimated savings from the existing logs are about 12.7 of
29 summed server minutes in run 4 and about 3.8 minutes in run 3. These are log estimates with unmeasured
overheads, not measured speedups. Historical benchmark files remain labeled v2.0.1.

Install and rollback instructions are in [UPGRADING.md](../UPGRADING.md). The candidate uses a separate
session namespace derived from its wheel digest; it starts with cold sessions.
