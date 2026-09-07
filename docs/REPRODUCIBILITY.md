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
- `verify.json` reports `VERIFY_PASS`, including the fixed focused witness
  and strict long-context witness.

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
- For the v1.1 recipe: the width-controller and QKV-shadow route receipts
  from the launch audit, and the integrated single-stream witness above
  32,768 prompt tokens with at least 20 completion tokens and the pinned
  codeword. The accepted candidate's [sanitized evidence](../results/evidence/candidate/cadence-v11/README.md#integrated-live-verification)
  records 48,957 prompt tokens and 51 completion tokens. Keep the candidate
  and host verifier recipe identities separate when they differ; the final
  archive has not been cold-boot tested.

A third-party reproduction on a separate three-Spark fleet is an open release
gate; none has been performed yet.

## Pinned inputs

| Input | Identity |
|---|---|
| Target checkpoint | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` at `25a44fdbf16862a46b7cc9921142c6c81350af2f` (declared byte-identical to `brandonmusic/GLM-5.3-Flash-tr3-4bpw` at `5ab363a8dcf6405955fd5f99671e01a1c9fb124b`) |
| Draft checkpoint | `incoai/GLM-5.3-Flash-DFlash2` at `dc77ff1c99eeb2df044ee3d4f0094eb033fee410` |
| Serving image | `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58` |
| Technique sources | FlyCockpit `9093765c757bd1976372196e44af84a67cf86bad`, vcruz305 `622cb878d66f703c597bd6baaa2423caa1786f99` |

Everything is listed with hashes in
[manifests/dependencies.json](../manifests/dependencies.json) and
[manifests/sbom.cdx.json](../manifests/sbom.cdx.json).
