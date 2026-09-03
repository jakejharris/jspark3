# JSpark3 v1: a technical report

Serving GLM-5.3 Flash across three NVIDIA DGX Sparks with tensor and expert
parallelism, a hash-gated runtime adaptation, and a selective INT8 trunk
overlay, compared with the public recipes that came before it and measured
against its own internal control, with the misses left in.

## 1. What this is and what we contributed

GLM-5.3 Flash is a large mixture-of-experts model. In its EXL3/TR3 4-bpw form
it fits on two DGX Sparks and several community recipes serve it that way.
JSpark3 v1 serves it on three: every rank holds one tensor-parallel shard of
the dense trunk and one third of the routed experts, the three nodes talk
over a RoCE-v2 triangle, and one OpenAI-compatible endpoint fronts the fleet.

The contribution is the three-Spark architecture, the runtime adaptation that
makes vLLM's GLM integration work at TP 3, the operating envelope, the W8A16
overlay, the measurement campaign, and a serving recipe that refuses to run
anything other than the measured construction. We did not train, fine-tune,
or quantize the model. The checkpoint is Brandon M. Music's ShapleyMcg EXL3/TR3
quantization published by Mia-AiLab; the draft is Inco AI's DFlash2; the
serving image and its GLM integration are MiaAI-Lab's; the TP3 technique
lineage is FlyCockpit's; the K-pool tail correction is vcruz305's. Each is
pinned by revision or digest and credited in `THIRD_PARTY_NOTICES.md`.

The Hugging Face side of this release re-hosts the target weights so operators
can fetch the checkpoint from one place. That mirror is an exact,
hash-verifiable copy of `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` at its pinned
revision, itself a byte-identical re-host of the quantization author's own
repository. Mirroring is not authorship: no weight byte is modified, the
checkpoint's own license and attribution travel with it, and the DFlash2 draft
is left where it is as a separately pinned dependency. All 123 allowlisted Git
LFS payloads and the exact completion receipt were remotely verified before
maintainer merge into public Hub main at immutable revision
`e7c34dba923916754cfcb0bdf6c2c75a9b7ff1fc`.

## 2. The problem with three

Two-way tensor parallelism divides GLM-5.3 Flash's geometry cleanly. Three-way
does not: 64 attention and KV heads, a 154,880-row vocabulary, and a
2,048-wide shared expert are not multiples of three. The FlyCockpit lineage
showed the way through: pad heads 64 to 66 with two inert heads, pad the
vocabulary to 154,944 with 64 inert tail rows, and widen the shared expert to
2,112, then narrow loaded weights back into place so the checkpoint bytes
never change. The DFlash2 draft has the same problem with its 32/8
query/KV heads, padded to 36/9. FlashInfer's SM120 kernels need the padded
head count for both decode and prefill.

JSpark3 v1 does not ship a patched vLLM. It ships five transform programs
that reconstruct the pinned upstream changes inside a disposable container at
start, verifying the hash of every file before and after and refusing on any
drift. `apply_tp3_overlay.py` carries the padding and the EXL3 expert-map
sharding; `apply_image_glm_dflash.py` carries MiaAI-Lab's GLM, DFlash2,
scheduler, and reasoning integration, including DFlash2's padded KV
slot-sharing, a decode-floor mixed-prefill policy, and a guard that
suppresses stop sequences inside reasoning; `apply_kpool_tail.py` carries the
K-pool hybrid-position and persistent tail-slot correction; the two KDA
programs are ours, batching the linear-attention f/g projections and
constructing mixed-output blocks. Together they touch 25 files with pinned
before and after hashes and exact seam cardinalities
(`recipe/transforms/README.md`).

The transforms are transactions. Each writes temporary siblings and rollback
copies, journals `PREPARED`, renames into place, verifies the terminal hashes,
and then removes the journal. If a container dies mid-apply, the next run
completes or rolls back from observed hashes. Mixed stages, unknown bytes, a
missing or duplicated seam, or a syntax failure in the generated file all
refuse without admitting the tree.

## 3. The operating envelope

- Serving image pinned by manifest and config digest; vLLM build 487ecf187
  inside it.
- TP 3, PP 1, EP on, multiprocessing executor.
- EXL3 target quantization; DFlash2 drafting 7 tokens per step with
  probabilistic rejection sampling.
- FP8 KV cache, prefix caching, 1,000,000-token configured maximum model
  length, 32 sequences, 8,192 batched tokens, full-decode-only CUDA graphs at
  8, 16, 24, 32, and 48, GPU memory utilization 0.83.
- Tool parser `glm47`, reasoning parser `glm45`, automatic tool choice,
  thinking disabled by default and switchable per request.
- Per rank: host network and IPC, 32 GiB shared memory, a private cgroup at
  64 GiB with swap disabled, no restart policy, all GPUs, `IPC_LOCK`,
  `/dev/infiniband`. Model, recipe, and technique sources read-only; only
  evidence and caches writable.
- NCCL pinned to IB transport with subnet-aware routing, cross-NIC and NIC
  merging disabled, one common RoCE-v2 IPv4 GID index, and an explicit refusal
  of `NCCL_PROTO`, `NCCL_ALGO`, and `NCCL_IB_ADDR_RANGE` overrides. No
  FlashInfer autotune.

The envelope is a hash-bound profile, not a set of defaults. The preflight
rows, the release manifest, the image receipt, and the entrypoint all check
it.

## 4. The W8A16 trunk overlay

With experts in EXL3 4-bpw, the remaining BF16 trunk (attention projections,
the shared and dense MLP, the LM head) is a disproportionate share of each
rank's weight memory. The overlay converts it at load time to INT8 with Marlin
kernels, leaving experts untouched.

Coverage is a census, not a heuristic: 169 runtime modules and 225 logical
tensors, enumerated in `results/evidence/candidate/overlay-census.json`. The
34 KDA f/g modules are excluded because their shapes do not fit the group
ladder. Group size follows 128, then 64, then 32, whichever first divides the
input dimension; the K=704 shapes, which 128 does not divide, are forced to
group 64 and the entrypoint refuses if that rule is not in the environment.
The conversion frees 1,595,392,320 bytes per rank, about 4.46 GiB across the
cluster, which flows to the KV cache and graph pools.

The overlay is installed by a loader hook: a small patcher rewrites vLLM's
base loader inside the container, and the recipe pins the overlay module
hash, the patcher hash, and the loader file's hash before and after. The
public overlay files differ from the measured ones only by identifier renames;
`manifests/derivation.json` records both hash sets and the recomputed
after-hash so a reviewer holding the private bundle can confirm the mapping.

## 5. Measurement method

Evidence in this report belongs to one of three classes and the classes are
never merged. **Published reference recipes** are recipes that were publicly
available before this release, quoted with the numbers their own authors
reported. **Local reproductions of a published recipe** are those recipes run
on this fleet, each with a pinned source revision and every adaptation listed;
none of them replays a source's own harness, so none is exact. **Internal
ablation** is the matched three-Spark control (same recipe, overlay disabled),
an unreleased internal development build that exists only to isolate what the
overlay changed.

The ablation ran on the same fleet with byte-identical request sets, so the
only variable is the overlay. Four instruments:

1. **Single-stream batteries (C1).** A frozen 24-request plan: 4 warm-up
   requests, then 20 scored across code, structured count, prose, a
   three-stream wave, and a six-stream wave. Thinking off, temperature 0,
   top-p 1, seed 20260830, 400 max tokens. Per-request rate is (completion
   tokens minus one) over the interval between the first and last visible
   streamed token; phase value is the median. Three candidate batteries, one
   earlier control battery, and three same-day matched controls.
2. **Pacing analysis.** From the visible token stream: a slow step is the
   request's own median interval plus 20 ms; compensation is credited when an
   adjacent interval's deficit covers at least half the excess; post-idle
   ramps and final steps are not interior. Nine gates, including a maximum
   uncompensated interior run below 5.
3. **Concurrency waves.** C12, C24, C48, 84 requests, aggregate service
   throughput over the wave's service window, DFlash2 acceptance from a quiet
   global counter delta, slowest-over-fastest fairness, TTFT percentiles.
4. **Long prefill proxy.** One identical 113,908-token scored prompt after one
   identical warm request; prompt tokens over time to first visible token.

Every run carried safety brackets: cgroup memory and swap counters, OOM
events, restart counts, GPU throttle bits, fabric counter reconciliation on
both HCAs of every rank, and all three containers running throughout. A run
that tripped a bracket would have been invalid; none did.

## 6. Results

### 6a. Published reference recipes (author-reported)

What a DGX Spark owner could obtain publicly before this release. Every number
here was reported by that recipe's author on that author's hardware with that
author's harness; none was measured on this fleet, and **no delta or ranking
against JSpark3 v1 is computed from any of them**. FlyCockpit's three-Spark
TP3 recipe reports structured count 69.0, 68.5, and 71.2 tok/s and code 52.3,
58.7, and 58.2 tok/s over three runs at temperature 0 with thinking off, at a
global draft acceptance of 81.5%. Mia's two-Spark TP2 recipe reports a
sparkDash C1 decode of 62.9 tok/s and lab medians of 65.1 tok/s structured and
27.1 tok/s prose. jetnet's three-Spark NVFP4 recipe reports 35.2 tok/s with
MTP-4 at 512K and 47.2 tok/s on its DFlash2 lane with thinking on. Node
counts, quantization lanes, speculation, contexts, and estimators differ
across all three; the minimum fields and caveats are in
[BENCHMARKS.md](BENCHMARKS.md).

### 6b. Local reproductions of a published recipe

Three of those recipes were run here. The historical Mia TP2 recipe at
`0e2e78f3de83624e6733b918724da27fc9040156`, site and safety adapted, ran a
same-prompt agent trajectory at 24.913 tok/s aggregate decode over 105,198
generated tokens with a 3.862 s mean time to first token. The current Mia TP2
source at `c190db1ae17ba8dff20129ed1f308d10c63cf37d`, made runnable under this
safety envelope by the single compatibility repair
`GLM53_INDEXER_WORKSPACE=rightsize`, ran the same prompt at 24.728 tok/s over
76,540 tokens with a 2.413 s mean time to first token. The Fly-derived local
build at `9093765c757bd1976372196e44af84a67cf86bad`, minimal-correctness and
safety adapted, ran it at 29.042 tok/s over 130,971 tokens with a 9.444 s mean
time to first token. All three trajectories are independent, so these describe
what each run did and are not engine-rate comparisons. The exact current-Mia
attempt never reached HTTP under this envelope and sent no request, so it
carries no number at all.

The adapted current-Mia arm also ran this project's own frozen 24-request
screen, with 24 requests, zero retries, and all counter brackets passing:
20.038576 tok/s prose, 44.562552 code, and 57.969782 structured count on the
single-stream phases, 31.362767 per-stream and 25.366765 service-window at C3,
and 37.985219 per-stream and 26.665269 service-window at C6. **That screen's
own plan omits the broader publication quality battery and marks rate-claim
eligibility false**, which is why those figures appear here and in
`results.json` and nowhere else in this release. The same arm's cold-prefill
ladder measured 540.933 tok/s at a cold 8K prompt in 14.793 s, 3937.381 tok/s
on an 8K follow-up in 2.035 s with 7,168 prefix-cache hits, then 1208.076
tok/s in 9.935 s at 12K, 1222.544 in 13.089 s at 16K, 1152.929 in 86.737 s at
100K, 1174.210 in 218.021 s at 256K, and 1182.742 in 253.650 s at 300K, every
one returning HTTP 200 with an exact stop. The rightsize repair reclaimed
about 4.91 GiB and raised KV capacity to 1,033,101 tokens.

### 6c. Internal ablation against the matched three-Spark control

The denominator in this subsection is the matched three-Spark control (same
recipe, overlay disabled), an unreleased internal development build. It is not
a product and not a market comparison.

Single-stream decode improved. Against the earlier control battery the
campaign medians were 66.257 versus 63.861 tok/s for code (+3.75%), 81.962
versus 77.510 for structured count (+5.74%), and 29.049 versus 28.308 for
prose (+2.62%). In the strict same-day pairing of candidate battery r3 with
matched control r6 the gaps were wider: +7.27% code, +6.63% count, +8.35%
prose. The three matched controls agreed with each other closely (code 62.257,
61.863, 61.768), which is why we trust the pairing more than the older
control battery.

Pacing got smoother. In the pairing, the median inter-token interval fell from
98.645 to 91.912 ms (-6.83%), p95 from 106.164 to 99.615 ms (-6.17%), p99 from
120.472 to 108.105 ms (-10.27%), and the worst interval from 364.416 to
148.344 ms (-59.29%). Slow intervals fell from 1.101% to 0.693% of steps and
uncompensated interior slow intervals from 0.293% to 0.154%; the longest run
fell from 2 to 1 and spikes of at least 250 ms from 1 to 0. Every candidate
battery passed all nine gates.

Concurrency helped at the top of the range. Aggregate service throughput was
155.986 versus 155.733 tok/s at C12 (+0.16%), 208.723 versus 206.235 at C24
(+1.21%), and 237.946 versus 229.966 at C48 (+3.47%), with DFlash2 acceptance
of 64.323%, 65.935%, and 64.562%. Fairness did not improve (0.303, 0.188,
0.148), and the C48 p90 time to first token was 96.722 s.

Prefill got worse. The 113,908-token proxy fell from 1277.443 to 1234.246
tok/s (-3.38%) and time to first token rose from 89.169 to 92.290 s (+3.50%).

The three-stream wave was the noisy instrument. Candidate C3 per-stream
medians were 69.634, 72.421, and 51.382 across the three batteries; the
strict pairing lost, 51.382 against 65.208 (-21.20%). The six-stream wave was
stable and slightly positive in the pairing (+2.91%). The large C6 gain
against the older control battery (+46.01%) is mostly that battery's state,
not the overlay, and we say so wherever it appears.

Full tables and estimator definitions: [BENCHMARKS.md](BENCHMARKS.md).

## 7. The gates we missed

The campaign had two predeclared promotion criteria beyond correctness and
safety. The first was a campaign code-median floor of 67.0 tok/s; the median
came in at 66.257, short by 0.743 tok/s (1.11%), and the campaign was
rejected. The second applied to a subsequent agent demonstration: the longest
uncompensated interior slow run had to stay below 5; it reached 14, and the
demonstration failed that gate even though the agent finished its task
correctly with fewer tokens and less elapsed time than the control run.

We kept both results. Re-running until a gate passes is a way to manufacture
a number, and the point of this release is that its numbers can be traced.
The single-stream batteries themselves passed every pacing gate; the misses
were a throughput floor set slightly above what the overlay delivers and a
pacing criterion on a cached, tool-using workload that the campaign's
synthetic batteries did not exercise.

## 8. Demonstration

Both arms ran the same agent prompt (identical hash) to produce a WebGL voxel
scene through a coding agent with prompt caching, independently. The JSpark3
v1 run completed in 132 requests and 32,618 generation tokens at 44.583 tok/s
aggregate decode with a 3.704 s mean time to first token over 28.7 minutes,
producing a correct artifact. The matched-control run took 168 requests and
88,823 tokens at 47.377 tok/s. These are demonstrations of a real workload; they
share a prompt but not a trajectory and are not a controlled comparison.

## 9. Making it reproducible

The pinned identities, all of which the recipe verifies before serving:

| Input | Identity |
|---|---|
| Target checkpoint | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` at `25a44fdbf16862a46b7cc9921142c6c81350af2f` (declared byte-identical to `brandonmusic/GLM-5.3-Flash-tr3-4bpw` at `5ab363a8dcf6405955fd5f99671e01a1c9fb124b`) |
| Draft checkpoint | `incoai/GLM-5.3-Flash-DFlash2` at `dc77ff1c99eeb2df044ee3d4f0094eb033fee410` |
| Serving image | `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58` (config `sha256:ad0cdd86d1ddd15ee758f519d16da15ac237f7f0648a5c52fbc20f9554944263`) |
| Technique sources | FlyCockpit `9093765c757bd1976372196e44af84a67cf86bad`; vcruz305 `622cb878d66f703c597bd6baaa2423caa1786f99` |
| Overlay | module `5aeff0cf92e715094d737faded2bf35000f7ce586213c495431b5a4805f7307d`; loader hook patcher `c84bdfbf69f7b1d3841155d35f73a06a601f2bcb33ae9e1d8423178dc31139b4`; loader before `a7e925f232ad3eebbee7ab37d3aba724c24465c3078da29489da0438664c6b08`, after `3205bff77aac34785167f5b21306048b9dc916b2c0691bf774bb3d9202bbd8da` |

Reproducing a rate is hard; reproducing a construction can be exact. The
recipe pins the checkpoints by revision, config hashes, shard inventory, and
byte totals; the image by two digests; the transforms by before and after
hashes of every file; the overlay by three hashes; the envelope by a profile;
and its own files by `SHA256SUMS`. The controller compares each rank's
preflight row byte for byte against the expected row, binds the start to the
preflight's checksum, mints an image receipt per rank that the entrypoint
must find and match, and starts ranks in a fixed order. Every command renders
as a dry-run without touching a host, and the release validator runs those
dry-runs alongside checksum, privacy, and claim checks.

The public evidence is a sanitized subset of the campaign's receipts with a
single machine-readable summary; the validator reconciles every number quoted
in this repository against that summary.

## 10. What is next

- A third-party reproduction on a separate fleet, which would move the
  evidence beyond a single operator.
- Continued use of the exact upstream image by digest. The prepared labeled
  derivative is retained for local reproduction only and is not a v1.0.0
  publication surface; redistributing it requires independent satisfaction of
  NVIDIA and upstream terms.
- A prefill-neutral overlay variant, since the prefill cost is the clearest
  regression in the trade.
- A C3-stable configuration or an explanation of the C3 variability.

## Appendix: reading the numbers

Rates in this report use the estimators named beside them. Per-stream medians
exclude admission holds; service-window aggregates include them; prefill
proxies divide prompt tokens by time to first token. A number from one
estimator cannot be compared with a number from another, and rates measured
on other fleets, other request sets, or with thinking enabled are not
comparable with these.
