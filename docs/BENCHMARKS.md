# Benchmark records

Decode is the aggregate rate of 1, 2, 4 or 8 code or prose requests (as labeled) started together (this build serves the Pi coding agent), each forced to 512 output tokens at temperature 0 with thinking off and no prefix-cache reuse, counted from the first to the last streamed token; prefill is the range over eight Pi-shaped coding-agent turns, each extending a cached prefix, measured right after a page-cache hygiene step.

Decode ladder figures are the recorded values to 0.1 tok/s (round half to even); the recorded values are in lo_exact and hi_exact. Every other figure is copied exactly as recorded.

## One operator clean-room observation

A v1.8.2 operator build on three DGX Sparks passed lifecycle verification,
post-hygiene prefill (1195.7–1270.6 tok/s) and the prefix-cache gate in `finehit`
mode. Single-stream code decode measured **49.05 tok/s median** across three
repetitions (45.70, 49.05, 50.23), 512 output tokens, temperature 0, coop off.
This is one measurement on one fleet, not a benchmark distribution or a full
admission result; that run did not execute an admission gate. A separate
coop-on boot differed in other settings, so these observations do not isolate
coop's speedup. The frozen files below are unchanged.

## Release, unedited EXL3 weights, production-stock, coop on

Build `v1.8.0`; weight mode `0` (`ABLIT=0`, unedited EXL3/TR3). Measured settings:
coop on, adaptive-K `ema`, dense FP8 `trunk`, display `full`, prefix-cache LRU on,
TRIAR resident but inactive. The operator default differs: **coop off**. Serving starts: 1; sweeps/repetitions: 2. Bands are within-start ranges, not confidence intervals.

| Metric | tok/s range |
|---|---|
| `prefill` | 1195.0 to 1262.6 |
| `decode_c1` | 68.2 to 73.3 |
| `decode_c2` | 101.0 to 103.9 |
| `decode_c4` | 136.9 to 141.7 |
| `decode_c8` | 174.0 to 179.7 |
| `decode_prose_c1` | 44.3 to 49.1 |
| `decode_prose_c2` | 61.6 to 63.5 |
| `decode_prose_c4` | 83.3 to 90.7 |
| `decode_prose_c8` | 110.1 to 110.8 |

Frozen for release. One production-stock serving start with two within-start sweeps. TRIAR is inactive; neither TRIAR nor thirds performance is claimed. Both sweeps and their ranges are retained; these are not confidence intervals. Structured workloads show substantial within-start variance. Quality observer capture was repaired and both quality panels passed on the same unchanged serving start.

## v1.7.4 base recipe, stock weights, qa profile

Build `v1.7.4`; weight mode `0`; frozen additional-toggle label `none` (not a claim that all runtime features were off). Serving starts: 1; sweeps/repetitions: 2. Bands are within-start ranges, not confidence intervals.

| Metric | tok/s range |
|---|---|
| `prefill` | 1196.7 to 1263.3 |
| `decode_c1` | 73.0 to 75.7 |
| `decode_c2` | 104.1 to 112.5 |
| `decode_c4` | 137.8 to 141.5 |
| `decode_c8` | 180.1 to 181.0 |
| `decode_prose_c1` | 44.8 to 45.7 |
| `decode_prose_c2` | 65.7 to 66.0 |
| `decode_prose_c4` | 85.9 to 92.6 |
| `decode_prose_c8` | 111.4 to 112.4 |

One qa-profile start with two sweeps. Both individual sweeps and ranges are retained. The production profile and resident collective toggle were not installed in this start.

## v1.7.4, edited weights (`ABLIT=1`, opt-in), no decode levers

Build `v1.7.4`; weight mode `1`; frozen additional-toggle label `none` (not a claim that all runtime features were off). Serving starts: 1; sweeps/repetitions: 2. Bands are within-start ranges, not confidence intervals.

| Metric | tok/s range |
|---|---|
| `prefill` | 1185.6 to 1266.2 |
| `decode_c1` | 62.7 to 64.4 |
| `decode_c2` | 106.5 to 108.0 |
| `decode_c4` | 126.8 to 136.5 |
| `decode_c8` | 168.3 to 172.5 |
| `decode_prose_c1` | 43.8 to 47.9 |
| `decode_prose_c2` | 58.0 to 60.3 |
| `decode_prose_c4` | 77.1 to 79.8 |
| `decode_prose_c8` | 101.6 to 106.4 |

An automated check held this run on one quality-panel answer that review found correct (the grader rejected an expression wrapped in inline backticks). Every decode wave completed without findings.

## v1.0.0 measured, published unchanged in v1.1.0

That build was measured with sparkDash, not this decode ladder. Its historical
rows remain in [RELEASE-NUMBERS](../release/RELEASE-NUMBERS.md).

All other prompt classes, repetitions, author-protocol rows and evidence hashes remain in [frozen results](https://github.com/jakejharris/jspark3/blob/v1.8.0/release/results-v1.8.0.json) and [measurement definitions](https://github.com/jakejharris/jspark3/blob/v1.8.0/release/RELEASE-NUMBERS.md).

The headline is combined throughput across concurrent streams, not single-chat
speed. The decode ladder is a stream-span measure, best of two sweeps; the
README's comparison table uses sparkDash DecodeBench prose, median of five.
Those instruments are distinct. The original campaign benchmark harness is not
part of this source package; the shipped gates measure readiness, correctness,
prefill and prefix-cache behavior, not an exact replay of the headline ladder.
Prompt class, sampling and speculative acceptance affect interactive speed.
Weight mode 0 means unedited quantized weights; mode 1 means donor-edited weights.
