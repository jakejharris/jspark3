# JSpark3 release numbers

State: **final**, frozen 2026-09-27T11:17:19+00:00. Tag: v1.8.0. Mode switching: Story B.

Every figure here appears with the same digits in `results.json`, which records its class, instrument, evidence hash, weight mode and toggle configuration. Nothing is scaled, and no figure is carried from one set to another. Stock-weight and edited-weight figures are never combined.

**Conditions.** Decode is the aggregate rate of 1, 2, 4 or 8 code-writing requests started together (this build serves the Pi coding agent), each forced to 512 output tokens at temperature 0 with thinking off and no prefix-cache reuse, counted from the first to the last streamed token; prefill is the range over eight Pi-shaped coding-agent turns, each extending a cached prefix, measured right after a page-cache hygiene step.

**Decode prompt kind:** `code`. Code-writing prompts, because this build serves the Pi coding agent.

**Precision.** Decode ladder figures are the recorded values to 0.1 tok/s (round half to even); the recorded values are in lo_exact and hi_exact. Every other figure is copied exactly as recorded.

**Author comparison.** A set ran the author benchmark exactly as published; see the sparkDash rows.

## release_m0: Release, stock weights, production profile

v1.8.0, stock weights, one serving start, two sweeps. Toggles: none. Frozen 2026-09-27T11:17:19+00:00.

Run note: Frozen for release. One production-stock serving start with two within-start sweeps. TRIAR is inactive; neither TRIAR nor thirds performance is claimed. Both sweeps and their ranges are retained; these are not confidence intervals. Structured workloads show substantial within-start variance. Quality observer capture was repaired and both quality panels passed on the same unchanged serving start.

| Measure | Low | High | Unit | n | Class | Instrument |
|---|---:|---:|---|---:|---|---|
| Prefill, Pi-shaped turns after hygiene (`prefill`) | 1195.0 | 1262.6 | tok/s | 8 | measured | Pi-turn prefill gate, repeat right after page-cache hygiene |
| Decode, code, 1 stream (`decode_c1`) | 68.2 | 73.3 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, code prompts |
| Decode, code, 2 streams (`decode_c2`) | 101.0 | 103.9 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, code prompts |
| Decode, code, 4 streams (`decode_c4`) | 136.9 | 141.7 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, code prompts |
| Decode, code, 8 streams (`decode_c8`) | 174.0 | 179.7 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, code prompts |
| Cold prefill, one Pi-shaped prompt (`prefill_cold_pi`) | 1357.4 | 1416.3 | tok/s | 3 | measured | cold prefill ladder, no cached prefix, one Pi-shaped prompt |
| Cold prefill, about 8k tokens (`prefill_cold_8k`) | 1457.8 | 1465.4 | tok/s | 3 | measured | cold prefill ladder, no cached prefix, about 8k tokens |
| Cold prefill, about 32k tokens (`prefill_cold_32k`) | 1475.8 | 1482.4 | tok/s | 3 | measured | cold prefill ladder, no cached prefix, about 32k tokens |
| Decode, code, 3 streams (`decode_c3`) | 125.7 | 134.7 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, code prompts |
| Decode, code, 5 streams (`decode_c5`) | 143.2 | 145.9 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, code prompts |
| Decode, code, 6 streams (`decode_c6`) | 154.0 | 164.7 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, code prompts |
| Decode, code, 7 streams (`decode_c7`) | 180.3 | 182.3 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, code prompts |
| Decode, prose, 1 stream (`decode_prose_c1`) | 44.3 | 49.1 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, prose prompts |
| Decode, prose, 2 streams (`decode_prose_c2`) | 61.6 | 63.5 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, prose prompts |
| Decode, prose, 3 streams (`decode_prose_c3`) | 72.4 | 79.2 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, prose prompts |
| Decode, prose, 4 streams (`decode_prose_c4`) | 83.3 | 90.7 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, prose prompts |
| Decode, prose, 5 streams (`decode_prose_c5`) | 86.1 | 87.5 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, prose prompts |
| Decode, prose, 6 streams (`decode_prose_c6`) | 106.4 | 106.5 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, prose prompts |
| Decode, prose, 7 streams (`decode_prose_c7`) | 111.9 | 113.3 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, prose prompts |
| Decode, prose, 8 streams (`decode_prose_c8`) | 110.1 | 110.8 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, prose prompts |
| Decode, structured, 1 stream (`decode_structured_c1`) | 97.5 | 98.7 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, structured prompts |
| Decode, structured, 2 streams (`decode_structured_c2`) | 94.0 | 142.9 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, structured prompts |
| Decode, structured, 3 streams (`decode_structured_c3`) | 159.9 | 175.2 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, structured prompts |
| Decode, structured, 4 streams (`decode_structured_c4`) | 168.6 | 229.7 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, structured prompts |
| Decode, structured, 5 streams (`decode_structured_c5`) | 193.3 | 201.3 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, structured prompts |
| Decode, structured, 6 streams (`decode_structured_c6`) | 167.1 | 217.0 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, structured prompts |
| Decode, structured, 7 streams (`decode_structured_c7`) | 175.7 | 191.5 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, structured prompts |
| Decode, structured, 8 streams (`decode_structured_c8`) | 198.0 | 198.6 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, structured prompts |
| sparkDash 400-token, code, 1 stream (`sd400_code_c1`) | 98.89 | 99.31 | tok/s | 3 | measured | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, code prompts |
| sparkDash 400-token, code, 2 streams (`sd400_code_c2`) | 163.11 | 164.49 | tok/s | 3 | measured | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, code prompts |
| sparkDash 400-token, code, 4 streams (`sd400_code_c4`) | 232.48 | 267.27 | tok/s | 3 | measured | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, code prompts |
| sparkDash 400-token, json, 1 stream (`sd400_json_c1`) | 69 | 76.64 | tok/s | 3 | measured | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, json prompts |
| sparkDash 400-token, json, 2 streams (`sd400_json_c2`) | 99.07 | 109.66 | tok/s | 3 | measured | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, json prompts |
| sparkDash 400-token, json, 4 streams (`sd400_json_c4`) | 112.12 | 151.43 | tok/s | 3 | measured | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, json prompts |
| sparkDash 400-token, prose, 1 stream (`sd400_prose_c1`) | 45.65 | 47.05 | tok/s | 3 | measured | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, prose prompts |
| sparkDash 400-token, prose, 2 streams (`sd400_prose_c2`) | 61.75 | 63.25 | tok/s | 3 | measured | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, prose prompts |
| sparkDash 400-token, prose, 4 streams (`sd400_prose_c4`) | 85.5 | 95.07 | tok/s | 3 | measured | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, prose prompts |
| sparkDash 400-token, structured, 1 stream (`sd400_structured_c1`) | 99.63 | 101.63 | tok/s | 3 | measured | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, structured prompts |
| sparkDash 400-token, structured, 2 streams (`sd400_structured_c2`) | 131.09 | 139.62 | tok/s | 3 | measured | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, structured prompts |
| sparkDash 400-token, structured, 4 streams (`sd400_structured_c4`) | 199.27 | 240.44 | tok/s | 3 | measured | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, structured prompts |
| sparkDash 512-token, prose, 1 stream (`sd512_prose_c1`) | 43.34 | 48.6 | tok/s | 5 | measured | sparkDash DecodeBench, 512 tokens, author protocol, prose prompts |
| Author-reported, same benchmark (`sd512_prose_c1`) | 40.1 | 40.1 | tok/s | | author-reported | sparkDash Decode bench, prose, 512 tokens, thinking off, 1-4 concurrent |
| sparkDash 512-token, prose, 2 streams (`sd512_prose_c2`) | 60.12 | 67.59 | tok/s | 5 | measured | sparkDash DecodeBench, 512 tokens, author protocol, prose prompts |
| Author-reported, same benchmark (`sd512_prose_c2`) | 56.6 | 56.6 | tok/s | | author-reported | sparkDash Decode bench, prose, 512 tokens, thinking off, 1-4 concurrent |
| sparkDash 512-token, prose, 3 streams (`sd512_prose_c3`) | 72.66 | 84.27 | tok/s | 5 | measured | sparkDash DecodeBench, 512 tokens, author protocol, prose prompts |
| Author-reported, same benchmark (`sd512_prose_c3`) | 75.5 | 75.5 | tok/s | | author-reported | sparkDash Decode bench, prose, 512 tokens, thinking off, 1-4 concurrent |
| sparkDash 512-token, prose, 4 streams (`sd512_prose_c4`) | 83.16 | 87.09 | tok/s | 5 | measured | sparkDash DecodeBench, 512 tokens, author protocol, prose prompts |
| Author-reported, same benchmark (`sd512_prose_c4`) | 88.4 | 88.4 | tok/s | | author-reported | sparkDash Decode bench, prose, 512 tokens, thinking off, 1-4 concurrent |

## base_m0_q: v1.7.4 base recipe, stock weights, QA profile

v1.7.4, stock weights, one serving start, two sweeps. Toggles: none. Frozen 2026-09-26T22:30:16.274960Z.

Run note: One QA-profile start with two sweeps. Structured workloads vary: c4 spread is 17.09 percent and c6 is 13.77 percent relative to the first sweep; c6 range relative to the lower value is 15.96 percent. Both individual sweeps and ranges are retained. The production profile and resident collective toggle were not installed in this start.

| Measure | Low | High | Unit | n | Class | Instrument |
|---|---:|---:|---|---:|---|---|
| Prefill, Pi-shaped turns after hygiene (`prefill`) | 1196.7 | 1263.3 | tok/s | 8 | measured | Pi-turn prefill gate, repeat right after page-cache hygiene |
| Decode, code, 1 stream (`decode_c1`) | 73.0 | 75.7 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, code prompts |
| Decode, code, 2 streams (`decode_c2`) | 104.1 | 112.5 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, code prompts |
| Decode, code, 4 streams (`decode_c4`) | 137.8 | 141.5 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, code prompts |
| Decode, code, 8 streams (`decode_c8`) | 180.1 | 181.0 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, code prompts |
| Cold prefill, one Pi-shaped prompt (`prefill_cold_pi`) | 1357.3 | 1425.2 | tok/s | 3 | measured | cold prefill ladder, no cached prefix, one Pi-shaped prompt |
| Cold prefill, about 8k tokens (`prefill_cold_8k`) | 1462.4 | 1465.0 | tok/s | 3 | measured | cold prefill ladder, no cached prefix, about 8k tokens |
| Cold prefill, about 32k tokens (`prefill_cold_32k`) | 1477.8 | 1482.3 | tok/s | 3 | measured | cold prefill ladder, no cached prefix, about 32k tokens |
| Decode, code, 3 streams (`decode_c3`) | 133.6 | 133.7 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, code prompts |
| Decode, code, 5 streams (`decode_c5`) | 141.4 | 143.8 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, code prompts |
| Decode, code, 6 streams (`decode_c6`) | 159.1 | 172.6 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, code prompts |
| Decode, code, 7 streams (`decode_c7`) | 164.7 | 180.3 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, code prompts |
| Decode, prose, 1 stream (`decode_prose_c1`) | 44.8 | 45.7 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, prose prompts |
| Decode, prose, 2 streams (`decode_prose_c2`) | 65.7 | 66.0 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, prose prompts |
| Decode, prose, 3 streams (`decode_prose_c3`) | 73.6 | 78.9 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, prose prompts |
| Decode, prose, 4 streams (`decode_prose_c4`) | 85.9 | 92.6 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, prose prompts |
| Decode, prose, 5 streams (`decode_prose_c5`) | 85.3 | 88.8 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, prose prompts |
| Decode, prose, 6 streams (`decode_prose_c6`) | 108.7 | 112.0 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, prose prompts |
| Decode, prose, 7 streams (`decode_prose_c7`) | 112.6 | 115.4 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, prose prompts |
| Decode, prose, 8 streams (`decode_prose_c8`) | 111.4 | 112.4 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, prose prompts |
| Decode, structured, 1 stream (`decode_structured_c1`) | 96.0 | 97.5 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, structured prompts |
| Decode, structured, 2 streams (`decode_structured_c2`) | 135.0 | 135.2 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, structured prompts |
| Decode, structured, 3 streams (`decode_structured_c3`) | 154.9 | 161.7 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, structured prompts |
| Decode, structured, 4 streams (`decode_structured_c4`) | 162.2 | 195.6 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, structured prompts |
| Decode, structured, 5 streams (`decode_structured_c5`) | 167.7 | 168.6 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, structured prompts |
| Decode, structured, 6 streams (`decode_structured_c6`) | 162.1 | 188.0 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, structured prompts |
| Decode, structured, 7 streams (`decode_structured_c7`) | 188.6 | 191.2 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, structured prompts |
| Decode, structured, 8 streams (`decode_structured_c8`) | 197.6 | 202.5 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, structured prompts |
| sparkDash 400-token, code, 1 stream (`sd400_code_c1`) | 99.12 | 99.23 | tok/s | 3 | measured | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, code prompts |
| sparkDash 400-token, code, 2 streams (`sd400_code_c2`) | 164.82 | 164.87 | tok/s | 3 | measured | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, code prompts |
| sparkDash 400-token, code, 4 streams (`sd400_code_c4`) | 246.19 | 265.43 | tok/s | 3 | measured | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, code prompts |
| sparkDash 400-token, json, 1 stream (`sd400_json_c1`) | 77.03 | 79.27 | tok/s | 3 | measured | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, json prompts |
| sparkDash 400-token, json, 2 streams (`sd400_json_c2`) | 103.59 | 109.73 | tok/s | 3 | measured | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, json prompts |
| sparkDash 400-token, json, 4 streams (`sd400_json_c4`) | 154.39 | 164.55 | tok/s | 3 | measured | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, json prompts |
| sparkDash 400-token, prose, 1 stream (`sd400_prose_c1`) | 45.93 | 50.05 | tok/s | 3 | measured | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, prose prompts |
| sparkDash 400-token, prose, 2 streams (`sd400_prose_c2`) | 58.35 | 62.98 | tok/s | 3 | measured | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, prose prompts |
| sparkDash 400-token, prose, 4 streams (`sd400_prose_c4`) | 85.85 | 96.83 | tok/s | 3 | measured | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, prose prompts |
| sparkDash 400-token, structured, 1 stream (`sd400_structured_c1`) | 100.24 | 102.95 | tok/s | 3 | measured | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, structured prompts |
| sparkDash 400-token, structured, 2 streams (`sd400_structured_c2`) | 136.52 | 141.23 | tok/s | 3 | measured | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, structured prompts |
| sparkDash 400-token, structured, 4 streams (`sd400_structured_c4`) | 202.89 | 250.31 | tok/s | 3 | measured | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, structured prompts |
| sparkDash 512-token, prose, 1 stream (`sd512_prose_c1`) | 44.11 | 46.12 | tok/s | 5 | measured | sparkDash DecodeBench, 512 tokens, author protocol, prose prompts |
| Author-reported, same benchmark (`sd512_prose_c1`) | 40.1 | 40.1 | tok/s | | author-reported | sparkDash Decode bench, prose, 512 tokens, thinking off, 1-4 concurrent |
| sparkDash 512-token, prose, 2 streams (`sd512_prose_c2`) | 55.32 | 66.97 | tok/s | 5 | measured | sparkDash DecodeBench, 512 tokens, author protocol, prose prompts |
| Author-reported, same benchmark (`sd512_prose_c2`) | 56.6 | 56.6 | tok/s | | author-reported | sparkDash Decode bench, prose, 512 tokens, thinking off, 1-4 concurrent |
| sparkDash 512-token, prose, 3 streams (`sd512_prose_c3`) | 73.47 | 79.82 | tok/s | 5 | measured | sparkDash DecodeBench, 512 tokens, author protocol, prose prompts |
| Author-reported, same benchmark (`sd512_prose_c3`) | 75.5 | 75.5 | tok/s | | author-reported | sparkDash Decode bench, prose, 512 tokens, thinking off, 1-4 concurrent |
| sparkDash 512-token, prose, 4 streams (`sd512_prose_c4`) | 85.79 | 91.99 | tok/s | 5 | measured | sparkDash DecodeBench, 512 tokens, author protocol, prose prompts |
| Author-reported, same benchmark (`sd512_prose_c4`) | 88.4 | 88.4 | tok/s | | author-reported | sparkDash Decode bench, prose, 512 tokens, thinking off, 1-4 concurrent |

## opt_in_m1: v1.7.4 base recipe, no decode levers

v1.7.4, edited weights (opt-in), one serving start, two sweeps. Toggles: none. Frozen 2026-09-26T15:55:00Z.

Run note: An automated check held this run on one quality-panel answer that review found correct (the grader rejected an expression wrapped in inline backticks). Every decode wave completed without findings.

| Measure | Low | High | Unit | n | Class | Instrument |
|---|---:|---:|---|---:|---|---|
| Prefill, Pi-shaped turns after hygiene (`prefill`) | 1185.6 | 1266.2 | tok/s | 8 | measured | Pi-turn prefill gate, repeat right after page-cache hygiene |
| Decode, code, 1 stream (`decode_c1`) | 62.7 | 64.4 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, code prompts |
| Decode, code, 2 streams (`decode_c2`) | 106.5 | 108.0 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, code prompts |
| Decode, code, 4 streams (`decode_c4`) | 126.8 | 136.5 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, code prompts |
| Decode, code, 8 streams (`decode_c8`) | 168.3 | 172.5 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, code prompts |
| Prefill, Pi-shaped turns, first pass after readiness (`prefill_first_pass`) | 1186.8 | 1257.4 | tok/s | 8 | measured | Pi-turn prefill gate, first pass after readiness (recorded, not gating) |
| Cold prefill, one Pi-shaped prompt (`prefill_cold_pi`) | 1347.4 | 1411.7 | tok/s | 3 | measured | cold prefill ladder, no cached prefix, one Pi-shaped prompt |
| Cold prefill, about 8k tokens (`prefill_cold_8k`) | 1455.6 | 1465.0 | tok/s | 3 | measured | cold prefill ladder, no cached prefix, about 8k tokens |
| Cold prefill, about 32k tokens (`prefill_cold_32k`) | 1477.3 | 1480.6 | tok/s | 3 | measured | cold prefill ladder, no cached prefix, about 32k tokens |
| Decode, code, 3 streams (`decode_c3`) | 122.4 | 134.2 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, code prompts |
| Decode, code, 5 streams (`decode_c5`) | 137.0 | 139.9 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, code prompts |
| Decode, code, 6 streams (`decode_c6`) | 150.8 | 163.3 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, code prompts |
| Decode, code, 7 streams (`decode_c7`) | 161.8 | 166.0 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, code prompts |
| Decode, prose, 1 stream (`decode_prose_c1`) | 43.8 | 47.9 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, prose prompts |
| Decode, prose, 2 streams (`decode_prose_c2`) | 58.0 | 60.3 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, prose prompts |
| Decode, prose, 3 streams (`decode_prose_c3`) | 72.3 | 73.6 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, prose prompts |
| Decode, prose, 4 streams (`decode_prose_c4`) | 77.1 | 79.8 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, prose prompts |
| Decode, prose, 5 streams (`decode_prose_c5`) | 74.2 | 78.0 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, prose prompts |
| Decode, prose, 6 streams (`decode_prose_c6`) | 94.3 | 94.9 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, prose prompts |
| Decode, prose, 7 streams (`decode_prose_c7`) | 99.8 | 105.7 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, prose prompts |
| Decode, prose, 8 streams (`decode_prose_c8`) | 101.6 | 106.4 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, prose prompts |
| Decode, structured, 1 stream (`decode_structured_c1`) | 101.4 | 101.9 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, structured prompts |
| Decode, structured, 2 streams (`decode_structured_c2`) | 127.8 | 143.6 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, structured prompts |
| Decode, structured, 3 streams (`decode_structured_c3`) | 174.1 | 174.6 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, structured prompts |
| Decode, structured, 4 streams (`decode_structured_c4`) | 167.5 | 202.9 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, structured prompts |
| Decode, structured, 5 streams (`decode_structured_c5`) | 193.9 | 203.2 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, structured prompts |
| Decode, structured, 6 streams (`decode_structured_c6`) | 215.2 | 227.5 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, structured prompts |
| Decode, structured, 7 streams (`decode_structured_c7`) | 196.7 | 204.3 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, structured prompts |
| Decode, structured, 8 streams (`decode_structured_c8`) | 192.2 | 262.6 | tok/s | 2 | measured | stream-span decode ladder, 512 forced tokens, streams started together, structured prompts |

## v1_1: v1.0.0 measured, published unchanged in v1.1.0

v1.0.0, stock weights, one serving start, one sweep. Toggles: none.

| Measure | Low | High | Unit | n | Class | Instrument |
|---|---:|---:|---|---:|---|---|
| sparkDash 400-token, code, 1 stream (`sd400_code_c1`) | 84.47 | 84.47 | tok/s |  | historical | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, code prompts |
| sparkDash 400-token, code, 2 streams (`sd400_code_c2`) | 139.77 | 139.77 | tok/s |  | historical | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, code prompts |
| sparkDash 400-token, code, 4 streams (`sd400_code_c4`) | 251.13 | 251.13 | tok/s |  | historical | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, code prompts |
| sparkDash 400-token, json, 1 stream (`sd400_json_c1`) | 64.11 | 64.11 | tok/s |  | historical | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, json prompts |
| sparkDash 400-token, json, 2 streams (`sd400_json_c2`) | 59.26 | 59.26 | tok/s |  | historical | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, json prompts |
| sparkDash 400-token, json, 4 streams (`sd400_json_c4`) | 118.57 | 118.57 | tok/s |  | historical | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, json prompts |
| sparkDash 400-token, prose, 1 stream (`sd400_prose_c1`) | 37.95 | 37.95 | tok/s |  | historical | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, prose prompts |
| sparkDash 400-token, prose, 2 streams (`sd400_prose_c2`) | 32.9 | 32.9 | tok/s |  | historical | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, prose prompts |
| sparkDash 400-token, prose, 4 streams (`sd400_prose_c4`) | 75.08 | 75.08 | tok/s |  | historical | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, prose prompts |
| sparkDash 400-token, structured, 1 stream (`sd400_structured_c1`) | 86.56 | 86.56 | tok/s |  | historical | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, structured prompts |
| sparkDash 400-token, structured, 2 streams (`sd400_structured_c2`) | 76.95 | 76.95 | tok/s |  | historical | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, structured prompts |
| sparkDash 400-token, structured, 4 streams (`sd400_structured_c4`) | 200.29 | 200.29 | tok/s |  | historical | sparkDash DecodeBench, 400 tokens, v1.0.0 receipt protocol, structured prompts |

- `prefill` not reported: v1.0.0 was not measured on this instrument.
- `decode_c1` not reported: v1.0.0 was not measured on this instrument.
- `decode_c2` not reported: v1.0.0 was not measured on this instrument.
- `decode_c4` not reported: v1.0.0 was not measured on this instrument.
- `decode_c8` not reported: v1.0.0 was not measured on this instrument.

## Evidence

Each cell names its evidence by SHA-256. The maintainers keep the files behind these hashes.

