# DRAFT: JSpark3 v1, one GLM-5.3 Flash endpoint on three DGX Sparks

Status: draft, not published. The v1.0.0 source and release links are final;
publishing this announcement is a separate maintainer action.

---

GLM-5.3 Flash runs well on two DGX Sparks; several community recipes prove
it. We wanted to know what a third Spark buys when it is used as a full
tensor-parallel and expert-parallel peer rather than a spare, and we wanted
the answer to be something other people could check. JSpark3 v1 is the
result: a serving recipe that pins every input, refuses to start anything it
did not measure, and publishes its evidence with the misses left in.

## Three is awkward

Two-way tensor parallelism divides the model's geometry cleanly. Three-way
does not: 64 attention heads, a 154,880-row vocabulary, and a 2,048-wide
shared expert are not multiples of three. The FlyCockpit three-Spark lineage
showed the fix, padding heads to 66, the vocabulary to 154,944, and the shared
expert to 2,112 with inert rows, then narrowing loaded weights back into
place. The checkpoint bytes never change; only the runtime layout does.

We did not ship a patched vLLM. We ship five transform programs that
reconstruct the pinned upstream changes inside a disposable container at
start, checking every file's hash before and after and refusing on drift. The
transforms are transactions: an interrupted apply is completed or rolled back
from observed hashes next time. Two of the five are ours, batching the linear
attention projections; the others carry MiaAI-Lab's, FlyCockpit's, and
vcruz305's work, credited and pinned.

## The overlay

With experts in EXL3 4-bpw, the BF16 trunk (attention, shared and dense MLP,
LM head) is a disproportionate share of each rank's weight memory. JSpark3 v1
converts 169 trunk modules to INT8 with Marlin kernels at load time, with a
128, 64, 32 group ladder, leaving experts and the KDA f/g modules alone. That
frees 1,595,392,320 bytes per rank, about 4.46 GiB across the cluster, which
flows into the KV cache and CUDA graph pools.

## What was already public

Before this release, a Spark owner had real choices, and the honest way to
place JSpark3 v1 is beside them rather than beside our own unpublished
predecessor. On our own frozen 24-request screen, thinking off, temperature 0,
400 max tokens, warm server, JSpark3 v1's medians across three batteries were
structured count 81.962, code 66.257, and prose 29.049 tok/s on a per-stream
estimator. FlyCockpit's three-Spark TP3 recipe reports structured count of
69.0, 68.5, and 71.2 tok/s and code of 52.3, 58.7, and 58.2 tok/s. Mia's
two-Spark TP2 recipe reports a sparkDash C1 decode of 62.9 tok/s and lab
medians of 65.1 structured and 27.1 prose. jetnet's three-Spark NVFP4 recipe
reports 35.2 tok/s at 512K with MTP-4, and 47.2 tok/s on its DFlash2 lane with
thinking on. Those are their authors' numbers, on their hardware, with their
harnesses, at different node counts, quantization lanes, contexts, and
estimators, so we do not turn any of them into a percentage against ourselves.
Four measurements on one page is not a scoreboard, and we have not built one.

We did run three of those recipes here. All three are adapted, and we say so:
the historical Mia TP2 recipe site and safety adapted, the current Mia TP2
source made runnable by a single compatibility repair, and a Fly-derived build
with our own correctness and safety repairs. On a shared agent prompt they
came in at 24.913, 24.728, and 29.042 tok/s aggregate decode, on independent
trajectories that make them product evidence rather than engine-rate
comparisons. We never ran jetnet at all.

## What the overlay changed

Separately, and inside our own development, we ran a matched A/B: the same
recipe with the overlay disabled, same fleet, byte-identical request sets.
That control is an unreleased internal build, not a competitor and not a
market comparison, but it is the only place where every other variable is
pinned, so it is where the overlay's effect actually shows.

Single-stream decode medians rose: code 63.861 to 66.257 tok/s (+3.75%),
structured count 77.510 to 81.962 (+5.74%), prose 28.308 to 29.049 (+2.62%);
in a strict same-day pairing the gains were +7.27%, +6.63%, and +8.35%. Token
pacing got smoother: the median inter-token interval fell from 98.645 to
91.912 ms and the worst interval from 364.416 to 148.344 ms. At 48 concurrent
streams, aggregate service throughput rose from 229.966 to 237.946 tok/s
(+3.47%).

Two things went the other way, and they are in the benchmarks page next to
the wins. The 113,908-token prefill proxy fell from 1277.443 to 1234.246
tok/s (-3.38%). The three-stream wave was noisy and lost its strict pairing
(-21.20%). And the build missed two of our own promotion gates: a campaign
code-median floor of 67.0 tok/s, by 0.743 tok/s (1.11%), and a sustained
pacing rule in an agent demonstration, where the longest uncompensated slow
run was 14 against a limit below 5. Nothing failed for correctness or
stability; we kept the misses because re-running until a gate passes is a way
to manufacture a number.

## What it does not claim

JSpark3 v1 is not a model. The GitHub recipe and release assets contain no
checkpoint weight objects; the separate public Hugging Face release carries an
attributed, byte-identical target mirror. Its exact payload and completion
receipt were remotely verified before maintainer merge into immutable public
main revision `e7c34dba923916754cfcb0bdf6c2c75a9b7ff1fc`.
The checkpoint is Brandon M. Music's ShapleyMcg EXL3/TR3 quantization
published by Mia-AiLab, attribution-required; the draft is Inco AI's DFlash2,
CC BY-NC-ND 4.0 for research and evaluation use. The assembled endpoint is
neither unrestricted open source nor commercial-ready, and the recipe's own
Apache-2.0 license covers only our code and prose. The container redistribution
audit is complete and blocks a JSpark3 GHCR image for v1.0.0; downstream users
still need to review the upstream terms for their own use. All evidence comes
from one fleet run by us; a third-party reproduction is the next step.

## Try it

Three DGX Sparks, cabled as a two-leg RoCE-v2 triangle, Docker and
`rdma-core`, the pinned checkpoints and image, and about twenty minutes of
reading in `docs/INSTALL.md`. The preflight will tell you exactly what is
wrong with your fleet before anything starts, which is the feature.

Release: `https://github.com/jakejharris/jspark3/releases/tag/v1.0.0`.
