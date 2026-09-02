# Reproducibility

Two different things can be reproduced from this repository, and they carry
different guarantees.

## 1. The construction (guaranteed, or the recipe refuses)

The recipe reproduces the exact runtime the evidence was measured on:
identical checkpoint bytes, identical image, identical transform outputs,
identical overlay, identical serving arguments and environment. Every input is
pinned and verified; a mismatch is a refusal, not a warning. Follow
[INSTALL.md](INSTALL.md), then confirm:

- `scripts/validate_checkpoint.py` printed `serving_checkpoint_pass: true`
  on every rank.
- Every container printed `JSPARK3_STARTUP_PATCH_PASS`.
- `verify.json` reports `PASS` with the fixed focused witness.

The recipe-level contract, with the complete list of pinned identities, is
[`recipe/docs/REPRODUCIBILITY.md`](../recipe/docs/REPRODUCIBILITY.md).

## 2. The measurements (reproducible method, not guaranteed numbers)

Token rates depend on ambient temperature, clocks, firmware, fabric placement,
cache state, and scheduler state. The method is fully specified so you can
repeat it and compare like with like.

Single-stream plan (C1). `results/evidence/candidate/c1-battery-r3/PLAN.json`
and `REQUESTS.json` carry the frozen 24-request plan: window order, per-request
prompts, sampling parameters (thinking disabled, temperature 0, top-p 1, seed
20260830, 400 max tokens), and the accounting contract. Per-request results and
the raw server-sent-event streams are under `results/`, `raw/`, and
`windows/` of each candidate battery, so the estimator can be recomputed from
the streams:

```
rate = (completion_tokens - 1) / (t_last_visible_token - t_first_visible_token)
phase value = median over the scored requests of that phase
```

Pacing. `tools/analyze_tail.py` implements the analyzer used for the pacing
tables: a slow step is the request's own median inter-token interval plus
20 ms; a slow step is compensated when an adjacent interval's deficit covers
at least half of its excess; post-idle ramps (the first five steps after a gap
of at least one second) and the final step of a request are not interior. Run
it on a battery's `raw/` directory to regenerate `pacing-analysis.json`.

Concurrency waves and long prefill. The scheduler and prefill receipts under
`results/evidence/*/scheduler-prefill/` include the request-set hashes, the
per-window service-window estimator, per-request timings, DFlash2 acceptance
from a quiet global counter delta, and the fabric and cgroup safety brackets.
The scored prefill prompt is 113,908 tokens after one identical warm request.

Agent demonstration. `results/evidence/candidate/agent-demo/` carries the
prompt, the server counter monitor stream, the pacing analysis, the produced
artifact, and screenshots; the matched control's demonstration summary is
beside it.
These are demonstrations of a real cached, tool-using workload, not controlled
comparisons.

## What a clean-room reproduction should report

- Preflight rows, release manifest, and `verify.json`, with hosts, addresses,
  and container identities redacted.
- The recipe manifest hash and overlay hash printed at startup.
- Per-phase C1 medians with the estimator above, and the pacing analysis.
- Whether the run cleared the internal gates recorded in
  [LIMITATIONS.md](LIMITATIONS.md), since the measured build did not.

A third-party reproduction on a separate three-Spark fleet is an open release
gate; none has been performed yet.
