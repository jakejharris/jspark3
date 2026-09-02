# JSpark3 v1 results summary

Generated from `results.json`. Every value here is machine-derived from the receipts under
`evidence/` and is reconciled by the release validator. The three evidence classes below are
kept separate on purpose and are never combined into one table or one ranking.

## 1. Published reference recipes

Recipes that a Spark owner could obtain publicly before this release. Every number is
**author-reported** from the source named in the row; none was measured on this fleet, and no
delta or ranking against JSpark3 v1 is computed from any of them. Instruments, prompts,
quantization lanes, contexts, and estimators differ between rows.

| Recipe | Nodes | Lane | Context | Reported decode (tok/s) | Basis |
|---|---:|---|---:|---|---|
| FlyCockpit TP3 `9093765c` | 3 | EXL3/TR3 4 bpw, DFlash2 | 1,000,000 | structured count 69.0 / 68.5 / 71.2; code 52.3 / 58.7 / 58.2; hello 37.9 / 36.9 / 37.3 | author-reported |
| Mia TP2 `c190db1a` | 2 | EXL3/TR3 4 bpw, DFlash2 | 1,000,000 | sparkDash C1 62.9; lab structured 65.1; lab prose 27.1 | author-reported |
| jetnet TP3 `bfc820ec` / `4fdba004` | 3 | NVFP4 with Marlin W4A16, MTP-4 or DFlash2 | 512K | MTP-4 35.2; DFlash2 thinking-on 47.2 | author-reported |

Minimum fields, sources, and caveats for each row: `docs/BENCHMARKS.md` and the
`published_references` block of `results.json`.

## 2. Local reproduction of a published recipe

Runs of a published recipe on this fleet, each with its pinned source identity and its
fidelity qualifier. None is an exact reproduction and none replays a source's own published
harness, so no number here is a like-for-like restatement of an author-reported figure. The
agent rows are same-prompt product runs with independent trajectories, not engine-rate
comparisons.

| Lineage | Fidelity | Nodes | Agent aggregate decode (tok/s) | Generated tokens | Mean TTFT (s) |
|---|---|---:|---:|---:|---:|
| `mia-tp2-historical-0e2e78f` | site/safety-adapted | 2 | 24.913 | 105,198 | 3.862 |
| `mia-tp2-current-c190db1a-adapted` | compatibility-adapted | 2 | 24.728 | 76,540 | 2.413 |
| `fly-derived-9093765c-adapted` | minimal-correctness/safety-adapted | 3 | 29.042 | 130,971 | 9.444 |

The exact current-Mia attempt did not reach HTTP under this safety envelope and sent no
request, so it carries no benchmark number of any kind. No literal FlyCockpit or jetnet run
exists here; jetnet was never run on this fleet at all.

## 3. Internal ablation (unreleased development control)

The denominator below is the matched three-Spark control (same recipe, overlay disabled), an
unreleased internal development build. It is this project's causal A/B for the overlay, not a
market comparison and not a competitor.

### Single-stream decode (C1), median of three candidate batteries vs the earlier control battery

| Phase | Matched three-Spark control | JSpark3 v1 | Delta |
|---|---:|---:|---:|
| Code | 63.861 | 66.257 | +3.75% |
| Structured count | 77.510 | 81.962 | +5.74% |
| Prose | 28.308 | 29.049 | +2.62% |

### Strict paired battery (candidate r3 vs same-day matched control)

| Phase | Matched three-Spark control | JSpark3 v1 | Delta |
|---|---:|---:|---:|
| Code | 61.768 | 66.257 | +7.27% |
| Structured count | 76.863 | 81.962 | +6.63% |
| Prose | 26.810 | 29.049 | +8.35% |
| C3 per-stream median | 65.208 | 51.382 | -21.20% |
| C6 per-stream median | 53.149 | 54.694 | +2.91% |

### Token pacing, paired r3 (visible SSE inter-token intervals)

| Metric | Matched three-Spark control | JSpark3 v1 | Delta |
|---|---:|---:|---:|
| Median interval (ms) | 98.645 | 91.912 | -6.83% |
| p95 (ms) | 106.164 | 99.615 | -6.17% |
| p99 (ms) | 120.472 | 108.105 | -10.27% |
| Max (ms) | 364.416 | 148.344 | -59.29% |

### Matched scheduler waves (aggregate service tok/s)

| Wave | Matched three-Spark control | JSpark3 v1 | Delta | DFlash2 acceptance | Fairness (slowest/fastest) |
|---|---:|---:|---:|---:|---:|
| C12 | 155.733 | 155.986 | +0.16% | 64.323% | 0.303 |
| C24 | 206.235 | 208.723 | +1.21% | 65.935% | 0.188 |
| C48 | 229.966 | 237.946 | +3.47% | 64.562% | 0.148 |

### Matched long prefill (113,908-token scored prompt)

| Metric | Matched three-Spark control | JSpark3 v1 | Delta |
|---|---:|---:|---:|
| Prefill proxy (tok/s) | 1277.443 | 1234.246 | -3.38% |
| TTFT (s) | 89.169 | 92.290 | +3.50% |

