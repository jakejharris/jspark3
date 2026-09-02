# Benchmarks

Every number on this page is taken from `results/results.json` and is
reconciled against it by the release validator. Each table states its
workload, estimator, sample size, and conditions, because rates measured
under different conventions are not comparable, and most of the differences
below are exactly that kind.

The evidence is kept in three classes, and they are never merged into one
table, one delta, or one ranking:

| Class | What it is | What may be said about it |
|---|---|---|
| **Published reference recipes** | Recipes a DGX Spark owner could obtain publicly before this release, with the numbers their authors reported | Quoted with their conditions and sources; no delta or rank against JSpark3 v1 |
| **Local reproduction of a published recipe** | A published recipe run on this fleet, with its pinned source identity and every adaptation listed | Reported with its fidelity qualifier; never described as exact unless the source's own harness was replayed |
| **Internal ablation** | The matched three-Spark control: the identical recipe with the overlay disabled, an unreleased internal development build | The project's causal A/B for the overlay; not a market comparison and not a competitor |

Grade of all locally measured evidence: `ENGINEERING-EVIDENCE`. It was
produced by the project's own campaign on its own fleet. No third-party
reproduction exists yet.

## Minimum fields beside every published number

| Field | Why it is required |
|---|---|
| Hardware and node count | Two Sparks and three Sparks are different machines |
| Checkpoint and quantization lane | EXL3/TR3 and NVFP4 are different products |
| Source or runtime revision | A rate belongs to a commit, not to a project name |
| Context | A 512K envelope and a 1,000,000-token envelope allocate memory differently |
| Tensor and expert parallel | Topology changes the collective cost |
| Speculation and draft parallel as actually loaded | A configured setting the loader ignores is not what ran |
| Prompt and workload | Structured-count rates are not prose rates |
| Thinking and sampling | Thinking on and thinking off are different workloads |
| Concurrency | A per-stream rate and an aggregate service rate are different quantities |
| Sample count | One run is a witness, not a distribution |
| Estimator | Per-stream, service-window, and wall-clock estimators disagree by design |
| Safety state | Swap, throttling, and memory pressure move rates |
| Local or external origin | Author-reported and locally measured are different evidence |
| Exact receipt or immutable URL | An untied number is unverified and stays out of this repository |
| Adaptations and limitations | An adapted run is not the published recipe |

## 1. Published reference recipes

These are the strongest defensible public references available before this
release: the closest public three-Spark recipes, plus Mia's two-Spark recipe
with its different node count disclosed. **Every number in this section is
author-reported from the source named in its row. None was measured on this
fleet, and no percentage, delta, or ranking against JSpark3 v1 is computed
from any of them**, because prompts, quantization lanes, speculation,
context, clocking, safety envelope, and estimators all differ.

### FlyCockpit TP3

Three NVIDIA DGX Sparks; EXL3/TR3 4-bpw target at revision
`25a44fdbf16862a46b7cc9921142c6c81350af2f`; FlyCockpit recipe at
`9093765c757bd1976372196e44af84a67cf86bad` with a mesh interconnect and
`NCCL_PROTO=LL`; 1,000,000-token context; TP 3 with EP; DFlash2 k=7. The
source advertises draft TP 1, but a pinned-source audit found that the loader
ignores that setting and builds the padded draft over world TP 3, so the
loaded behaviour is TP 3. Thinking off, temperature 0; single stream; three
runs per prompt; author-reported per-run decode rate.

| Prompt | Run 1 | Run 2 | Run 3 | Reported TTFT (s) |
|---|---:|---:|---:|---:|
| Structured count 1-200 (length 200) | 69.0 | 68.5 | 71.2 | 0.36 |
| `is_prime` code (length 200) | 52.3 | 58.7 | 58.2 | 0.37 |
| Short hello (17-token stop) | 37.9 | 36.9 | 37.3 | 0.35 |

Author-reported global draft acceptance for that session: 81.5%. Source:
<https://github.com/FlyCockpit/GLM-5.3-Flash-EXL3-3x-DGX-Sparks/tree/9093765c757bd1976372196e44af84a67cf86bad>.
Caveats stated by the source or found in the local audit: the measured
configuration ran at `gpu_memory_utilization=0.87`, where the source records
about 6 GiB of cgroup swap, and the safer 0.83 profile was not rate-measured
there; there is no prose row, no fixed JSpark3 request set, no raw local
receipt, and no estimator-equivalent comparison.

### Mia TP2

Two NVIDIA DGX Sparks, a different node count from this release; the same
EXL3/TR3 4-bpw target revision; Mia recipe at
`c190db1ae17ba8dff20129ed1f308d10c63cf37d`; 1,000,000-token context; TP 2;
DFlash2 k=7. Thinking off, temperature 0, 400 tokens, warm server with empty
KV, graphs and fused EXL3 MoE.

| Harness | Concurrency | Draft TP setting | Reported | Reported TTFT |
|---|---|---|---|---|
| `sparkDash Decode` | C1 | 1 | 62.9 tok/s | 719 ms |
| `sparkDash Decode` | C2 | 1 | 51.7 per stream, 103.3 aggregate | 6.62 s |
| `sparkDash Decode` | C4 | 1 | 37.1 per stream, 146.5 aggregate | 6.30 s |
| `tests/bench_decode.py`, median of five 400-token runs | C4 | 2 | structured 65.1, prose hash-map 27.1 | not reported |
| `tests/bench_decode.py`, median of five 400-token runs | C4 | 1 | structured 61.7, prose hash-map 26.9 | not reported |

Reported draft acceptance for the lab rows: 0.959 with 6.71 accepted tokens
per step on the structured prompt, 0.341 with 2.39 on the prose prompt. The
source also reports 24-27 tok/s on long or mixed prompts with roughly 60K to
100K of KV, and about 24.6 tok/s with MTP at k=2. Source:
<https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/tree/c190db1ae17ba8dff20129ed1f308d10c63cf37d>.
Caveats: prompt sensitivity is the point of this source, so structured rates
are not prose rates, and the per-stream and aggregate estimators differ
between rows and must not be mixed.

### jetnet TP3

Three NVIDIA DGX Sparks; a LibertAIDAI NVFP4 checkpoint with Marlin W4A16, a
different quantization lane from this release; eager execution at a 1500 MHz
clock cap; 512K context; TP 3; MTP at k=4 on main, with a DFlash2 lane added
on the current main. The current source states the model always thinks.

| Source | Reported |
|---|---|
| Historical and current main, MTP-4 | 35.2 tok/s, reported range 32.1 to 39.0; TTFT 0.26 s; roughly 1,718 to 1,800 prefill tok/s; needle retrieval pass at 471,813 tokens in 274.6 s |
| Current main, DFlash2 with thinking on | 47.2 tok/s, with the author warning that acceptance is workload-dependent |
| Pull request 1 | structured count 46-65, code 57-67, freeform 29-32 tok/s |
| Author's NVIDIA forum post, `tool-eval-bench` | 90 of 100, 74 pass, 11 partial, 3 fail, in 1,883.6 s at default reasoning |

Sources:
<https://github.com/jetnet/glm53-flash-nvfp4-tp3/tree/bfc820ec45cf7d30b8a1430207db3b653907eca0>,
<https://github.com/jetnet/glm53-flash-nvfp4-tp3/pull/1>,
<https://github.com/jetnet/glm53-flash-nvfp4-tp3/tree/4fdba0041e838299b5da581a84843f594861aea7>,
and
<https://forums.developer.nvidia.com/t/glm-5-3-flash-nvfp4-on-3x-dgx-spark-tp-3-512k-context-35-tok-s/381534>.
Caveats: a different target and speculation lane, a 512K rather than
1,000,000-token envelope, a clock-capped and prompt-dependent headline, and
no exact request match or local reproduction. The tool-eval receipt appears
only in the later forum post and must not be attributed to the earlier pinned
commit. The historical checkpoint also carried a scale concern documented by
the local audit.

## 2. Local reproduction of a published recipe

Runs of a published recipe on this fleet. Each carries its pinned source
identity and a fidelity qualifier. **None of these is an exact reproduction,
and none replays a source's own published harness**, so no row here restates
an author-reported figure from section 1. All three agent rows are
same-prompt product runs with independent trajectories: they describe what
each run did, not which recipe is faster.

| Lineage | Fidelity | Nodes | TP | Agent aggregate decode (tok/s) | Generated tokens | Mean TTFT (s) |
|---|---|---:|---:|---:|---:|---:|
| `mia-tp2-historical-0e2e78f` | site/safety-adapted | 2 | 2 | 24.913 | 105,198 | 3.862 |
| `mia-tp2-current-c190db1a-adapted` | compatibility-adapted | 2 | 2 | 24.728 | 76,540 | 2.413 |
| `fly-derived-9093765c-adapted` | minimal-correctness/safety-adapted | 3 | 3 | 29.042 | 130,971 | 9.444 |

Estimator for all three: an exclusive idle-to-idle server-counter delta over
the agent trajectory, with the aggregate decode rate being generated tokens
over summed decode seconds. High-effort thinking, provider default sampling.
Receipts: `results/evidence/reference/<lineage>/`.

**`mia-tp2-historical-0e2e78f`** is the pinned historical Mia recipe at
`0e2e78f3de83624e6733b918724da27fc9040156`, run locally with disclosed site
adaptations: local interface and GID mapping, existing read-only checkpoint
mounts, an API alias and port, isolated caches and evidence directories, and
hard no-swap containment. The six runtime patches are bytes from the pinned
source commit, not new model or runtime repairs by this project. Its
trajectory processed 3,317,035 prompt tokens of which 3,236,352 were cache
hits, spent 161.617 s in prefill and 4,222.682 s in decode, and passed 23 of
23 artifact checks on three consecutive runs.

**`mia-tp2-current-c190db1a-adapted`** is the current Mia source at
`c190db1ae17ba8dff20129ed1f308d10c63cf37d`. Its sole model or runtime
deviation is `GLM53_INDEXER_WORKSPACE=rightsize`, which reclaimed about
4.91 GiB and raised KV capacity to 1,033,101 tokens. That repair is what
makes the full 1,000,000-token current source runnable under this safety
envelope, which is precisely why this is an adapted local reproduction and
not an exact one. Its 24-request rapid screen and its cold-prefill ladder are
in [TECHNICAL-REPORT.md](TECHNICAL-REPORT.md); the screen's own plan marks
rate-claim eligibility false, so it is reported there and nowhere else.

**`fly-derived-9093765c-adapted`** pinned FlyCockpit at
`9093765c757bd1976372196e44af84a67cf86bad` and its mesh plugin, but
deliberately retained mandatory immutable checkpoint views, TP3 geometry
corrections, sparse-prefill and K-pool repairs, corrected drafter slot
sharing, explicit draft attention, the exact fail-closed lifecycle, and the
64 GiB no-swap envelope. It is lineage and code reuse with local correctness
and safety repairs, not a reproduction of the published performance. Its
trajectory reported an effective prompt rate of 9,962 tok/s including cache
hits.

**The exact current-Mia attempt** did not reach HTTP under this safety
envelope and sent no request. It carries no throughput, time-to-first-token,
prefill, quality, or agent number of any kind; it exists only to explain why
the adapted run above cannot be called exact.

**What does not exist here:** no literal FlyCockpit run, and no jetnet run at
all. jetnet was read and studied statically; no jetnet Spark-run receipt
exists in this project.

## 3. Internal ablation (unreleased development control)

This is the project's causal A/B for the selective W8A16 Marlin trunk
overlay. The denominator throughout is **the matched three-Spark control
(same recipe, overlay disabled), an unreleased internal development build**.
It was created during private development, it is not a product, it was never
published, and it is not a market comparison. It appears here because it is
the only comparison in this repository where hardware, topology, checkpoint,
draft, image, serving envelope, workload, estimator, and safety contract are
all matched, so the overlay is the intended material variable.

### Conditions common to every measurement in this section

- Hardware: three NVIDIA DGX Sparks (GB10, SM 12.1), two RoCE-v2 legs per node
  at MTU 9000, one endpoint with TP 3 and EP 3.
- Software: the pinned image, checkpoint, and draft listed in
  `manifests/dependencies.json`; the serving envelope in
  `recipe/config/profile.json`.
- Server state: warm. Each battery starts with warm-up requests that are not
  scored.
- Safety brackets for every run: zero OOM events, zero swap, zero restarts,
  zero GPU throttle bits, all three ranks running throughout. Maximum GPU
  temperature observed in the single-stream campaign was 72 C.

### Single-stream decode (C1)

Workload: a frozen 24-request plan, 4 warm-up then 20 scored requests, in
fixed order: code (3 requests), structured count (3), prose (4), a
three-stream wave (C3), and a six-stream wave (C6). Thinking disabled,
temperature 0, top-p 1, seed 20260830, 400 max tokens. Requests and prompts
are byte-identical across every battery and arm.

Estimator: per request, (completion tokens minus one) divided by the interval
between the first and last visible streamed token. Phase value: median of the
scored requests in that phase. For C3 and C6 the per-stream value is the
median across the concurrent streams (admission holds excluded).

#### Campaign medians against the earlier control battery

Three candidate batteries (r1, r2, r3) on separate runs; the earlier control
battery was measured on the same fleet and plan before the campaign.

| Phase | Earlier control | r1 | r2 | r3 | Median | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Code | 63.861 | 65.793 | 68.678 | 66.257 | 66.257 | +3.75% |
| Structured count | 77.510 | 82.075 | 81.381 | 81.962 | 81.962 | +5.74% |
| Prose | 28.308 | 27.769 | 30.194 | 29.049 | 29.049 | +2.62% |
| C3 per-stream median | 67.591 | 69.634 | 72.421 | 51.382 | 69.634 | +3.02% |
| C6 per-stream median | 37.460 | 53.411 | 55.840 | 54.694 | 54.694 | +46.01% |

Read the C6 row with care. The earlier control battery's C6 value is far
below the three same-day matched controls in the next table (51 to 54 tok/s),
so the +46.01% mostly reflects the state of that earlier battery rather than
an overlay effect. The paired comparison below (+2.91%) is the credible C6
number. C3 is variable in both directions: two candidate batteries exceeded
the earlier control and the third fell to 51.382.

#### Strict same-day pairing

Candidate battery r3 against matched control r6, the overlay-disabled
configuration re-measured on the same day, same fleet, same frozen plan. All
three matched controls (r4, r5, r6) are in
`results/evidence/internal-ablation-control/`.

| Phase | Control r4 | Control r5 | Control r6 | JSpark3 v1 r3 | Paired delta (r3 vs r6) |
|---|---:|---:|---:|---:|---:|
| Code | 62.257 | 61.863 | 61.768 | 66.257 | +7.27% |
| Structured count | 77.068 | 76.944 | 76.863 | 81.962 | +6.63% |
| Prose | 27.559 | 27.661 | 26.810 | 29.049 | +8.35% |
| C3 per-stream median | 60.837 | 61.636 | 65.208 | 51.382 | -21.20% |
| C6 per-stream median | 54.405 | 51.482 | 53.149 | 54.694 | +2.91% |

The C3 loss is real in this pairing and is disclosed as such; see
[LIMITATIONS.md](LIMITATIONS.md).

#### Service-window estimator for the waves

For readers who prefer a wave's aggregate over its serialized service window
(admission holds included): the earlier control battery's service-window
values were 20.852 tok/s (C3) and 56.789 tok/s (C6); candidate battery r3's
were 41.753 tok/s (C3) and 56.543 tok/s (C6). These are a different estimator
from the per-stream medians above and must not be mixed with them.

### Token pacing (paired r3 vs control r6)

Source: the visible streamed token intervals of the scored single-stream
requests in the paired batteries, analyzed with `tools/analyze_tail.py`
(method in `results/evidence/candidate/c1-battery-r3/pacing-analysis.json`).
Slow step: the request's own median inter-token interval plus 20 ms.
Compensated: an adjacent interval's deficit covers at least half of the
excess. Interior: not a post-idle ramp step and not the final step.

| Metric | Control r6 | JSpark3 v1 r3 | Delta |
|---|---:|---:|---:|
| Median interval (ms) | 98.645 | 91.912 | -6.83% |
| p95 (ms) | 106.164 | 99.615 | -6.17% |
| p99 (ms) | 120.472 | 108.105 | -10.27% |
| Maximum interval (ms) | 364.416 | 148.344 | -59.29% |
| Median absolute deviation (ms) | 2.306 | 2.307 | +0.04% |
| Slow intervals | 1.101% | 0.693% | |
| Uncompensated interior slow intervals | 0.293% | 0.154% | |
| Longest uncompensated interior run | 2 | 1 | |
| Intervals of at least 250 ms | 1 | 0 | |

All nine analyzer gates passed for every candidate battery (longest
uncompensated interior run 1, 0, and 1 for r1, r2, r3).

### Matched concurrency waves (scheduler)

Workload: frozen C12, C24, and C48 waves, 84 requests total, byte-identical
request set on both arms, one wave each, thinking disabled. All 84 requests
returned HTTP 200 with complete streams and usage. This harness does not
evaluate semantic code correctness.

Estimator: aggregate service throughput is the sum over the wave of
(completion tokens minus one) divided by the interval from the earliest
request start to the latest request end. DFlash2 acceptance is taken from a
quiet global counter delta around the wave. Fairness is the slowest stream's
rate over the fastest stream's rate.

| Wave | Matched control aggregate | JSpark3 v1 aggregate | Delta | DFlash2 acceptance | Fairness | TTFT p90 (s) |
|---|---:|---:|---:|---:|---:|---:|
| C12 | 155.733 | 155.986 | +0.16% | 64.323% | 0.303 | 14.636 |
| C24 | 206.235 | 208.723 | +1.21% | 65.935% | 0.188 | 18.937 |
| C48 | 229.966 | 237.946 | +3.47% | 64.562% | 0.148 | 96.722 |

Fairness did not improve over the matched control in any wave. The C48 time
to first token at p90 is a limitation for interactive use and is discussed in
[LIMITATIONS.md](LIMITATIONS.md).

### Matched long prefill

Workload: one identical 113,908-token scored prompt after one identical warm
request, same request set on both arms. Estimator: server-reported prompt
tokens divided by time to first visible token. Both responses were exactly
`DONE`.

| Metric | Matched control | JSpark3 v1 | Delta |
|---|---:|---:|---:|
| Prefill proxy (tok/s) | 1277.443 | 1234.246 | -3.38% |
| Time to first token (s) | 89.169 | 92.290 | +3.50% |

The overlay costs prefill throughput. This is a measured regression and part
of the trade.

### Agent demonstration (not a controlled comparison)

Both arms ran the same agent task from the same prompt (identical prompt
hash), producing a WebGL voxel artifact through a tool-using coding agent
with prompt caching. The trajectories were independent, so request counts,
token counts, and cache behaviour differ; the numbers describe what each run
did, not a ranking. The JSpark3 v1 run used high-effort thinking. The control
figures come from a server counter delta against an attribution reference
point.

| | Matched control run | JSpark3 v1 run |
|---|---:|---:|
| Completed requests | 168 | 132 |
| Generation tokens | 88,823 | 32,618 |
| Aggregate decode (tok/s) | 47.377 | 44.583 |
| Mean time to first token (s) | 4.038 | 3.704 |
| Effective prompt tok/s including cache hits | 20,319 | |
| Uncached prompt-compute tok/s | 1,110 | |
| Peak 10-second generation (tok/s) | 71.8 | |
| Wall time (min) | | 28.7 |
| Median inter-token interval (ms) | | 94.900 |
| Longest uncompensated interior slow run | | 14 |

The JSpark3 v1 run finished the task correctly with fewer tokens and less
elapsed time, and its artifact passed the WebGL check. It also failed the
internal sustained-pacing gate (longest run 14 against a limit below 5). Both
facts stand.

### Internal gates the measured build did not clear

| Gate | Rule | Observed | Verdict |
|---|---|---:|---|
| Campaign code-median floor | median C1 code across three batteries at least 67.0 tok/s | 66.257 | REJECT (short by 0.743 tok/s, 1.11%) |
| Sustained pacing in the demonstration | longest uncompensated interior slow run below 5 | 14 | FAIL |

These were the project's own promotion criteria. The recipe is published as
measured, with both misses preserved, rather than re-run until it passed.

## Editorial rules this page follows

- The three classes above are never combined into one table or one ranking.
- No percentage is ever computed between JSpark3 v1 and a published reference
  or a local reproduction; the validator refuses a percentage that shares a
  table row or a sentence with either class.
- The overlay-disabled control is always named as the matched three-Spark
  control and always marked as an unreleased internal development build.
- Same-prompt agent demonstrations are kept apart from fixed-request engine
  screens.
- An adapted run is never called exact, and a number that cannot be tied to a
  receipt or an immutable URL is not published at all.

## Evidence map

| Table | Receipts |
|---|---|
| Published reference recipes | `results/evidence/reference/published-references.json` (author-reported values, sources, and minimum fields) |
| Local reproductions | `results/evidence/reference/mia-tp2-historical-0e2e78f/`, `results/evidence/reference/mia-tp2-current-c190db1a-adapted/`, `results/evidence/reference/fly-derived-9093765c-adapted/`, `results/evidence/reference/mia-tp2-current-exact-attempt/finding.json` |
| C1 batteries | `results/evidence/candidate/c1-battery-r{1,2,3}/` (MATRIX, RUN, PLAN, REQUESTS, per-request results, windows, raw streams, counter deltas), `results/evidence/candidate/c1-campaign.json` |
| Internal ablation control batteries | `results/evidence/internal-ablation-control/c1-battery/`, `results/evidence/internal-ablation-control/c1-matched-control-r{4,5,6}/` |
| Pacing | `pacing-analysis.json` in each candidate battery; `tools/analyze_tail.py` |
| Scheduler and prefill | `results/evidence/candidate/scheduler-prefill/`, `results/evidence/internal-ablation-control/scheduler-prefill/` |
| Demonstration | `results/evidence/candidate/agent-demo/`, `results/evidence/internal-ablation-control/agent-demo/summary.json` |
| Overlay census and safety | `results/evidence/candidate/overlay-census.json`, `results/evidence/candidate/admission-safety.json` |
