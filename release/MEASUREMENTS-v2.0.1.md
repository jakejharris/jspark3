# v2.0.1 measurement record

The structured [record](results-v2.0.1.json) holds every v2.0.1 figure with its result set, its configuration and its evidence hash. Historical v1.8.x results stay byte-identical in their own release files; they measured a different engine and different weights, so they do not transfer to this release.

## Results

| Measurement | base weights + draft model · base profile (default) | refusal-removed (ablit) weights + draft model · ablit profile | base weights, no draft model (commercial use) · base profile |
|---|---:|---:|---:|
| Decode per stream, short code replies (tok/s) | 80.9 | 73.3 | 63.6 |
| Decode per stream, short prose replies (tok/s) | 59.6 | 63.6 | 57.7 |
| Decode, one stream after a 32K-token prompt (tok/s) | 83.6 | 72.3 | not measured |
| Aggregate decode (tok/s): short prompts, 41-62 tokens, 1 concurrent | 59.8 | 69.8 | 59.0 |
| Aggregate decode (tok/s): short prompts, 41-62 tokens, 2 concurrent | 77.6 | 75.8 | not measured |
| Aggregate decode (tok/s): short prompts, 41-62 tokens, 4 concurrent | 97.5 | 96.4 | not measured |
| Aggregate decode (tok/s): short prompts, 41-62 tokens, 8 concurrent | 126.0 | 121.7 | 81.2 |
| Aggregate decode (tok/s): short prompts, 41-62 tokens, 16 concurrent | 106.8 | 110.0 | not measured |
| Time to first token, p50 (s): short prompts, 41-62 tokens, 8 concurrent | 0.54 (first visible text 1.5; 3 of 24 showed no text) | 0.54 (first visible text 1.2; 1 of 24 showed no text) | 1.9 (visible text 0.41 s after the first token, median over the replies with text; 3 of 24 showed no text) |
| First visible text, p50 (s): short prompts, 41-62 tokens, 8 concurrent | 1.5 | 1.2 | see the note below the table |
| Short replies that showed no text: short prompts, 41-62 tokens, 8 concurrent | 3 of 24 | 1 of 24 | 3 of 24 |
| Longest token gap, median (s): short prompts, 41-62 tokens, 8 concurrent, while a prompt of about 8,000 or 36,000 tokens joins | 0.52 | 0.49 | not measured |
| Longest token gap, max (s): short prompts, 41-62 tokens, 8 concurrent, while a prompt of about 8,000 or 36,000 tokens joins | 0.75 | 0.57 | not measured |
| Cold time to first token, 8K-token prompt (s) | 3.8 | 3.8 | 3.9 |
| Cold time to first token, 32K-token prompt (s) | 15.0 | 15.0 | 15.8 |
| Cold time to first token, 64K-token prompt (s) | 30.4 | 30.4 | not measured |
| Cold time to first token, 128K-token prompt (s) | 63.6 | 63.8 | not measured |
| Draft tokens accepted per verify step: short prompts, 41-62 tokens, 8 concurrent | 2.75 | 2.55 | not measured |
| Draft tokens accepted / proposed: short prompts, 41-62 tokens, 8 concurrent | 0.67 | 0.69 | not measured |
| Draft tokens accepted per verify step, one repeated prompt | 2.11 | 2.04 | not measured |
| Draft tokens accepted / proposed, one repeated prompt | 0.61 | 0.55 | not measured |

*Time to first token, p50, base weights + draft model · base profile:* First visible text at about 1.5 s (median); 3 of 24 short replies spent their 96-token limit on reasoning and showed no text. Measured at reasoning effort low.

*Time to first token, p50, refusal-removed (ablit) weights + draft model · ablit profile:* First visible text at about 1.2 s (median); 1 of 24 short replies spent its 96-token limit on reasoning and showed no text. Measured at reasoning effort low.

*Time to first token, p50, base weights, no draft model (commercial use) · base profile:* Visible text 0.41 s after the first token (median, p95 1.6 s) in the 21 replies that showed text; 3 of 24 short replies spent their 96-token limit on reasoning and showed no text. Measured at reasoning effort low.

The first-token notes use two measures: with the draft model, the median time to first visible text; without it, the median gap from first token to first visible text in the replies that showed text, because a median taken only over replies that showed text would come out below the first-token median taken over all replies.

Every set was measured on the same build, each on its weights variant's shipped settings, and no figure is a best run. Rates and times are medians, with the number of runs given below; the token gap is given as both a median and a maximum, and the context window is a setting, not a measurement. Short-reply decode is reported separately for code and for prose: the per-stream rate of replies capped at 256 tokens (median of 3 each). Long decode is one greedy code stream of up to 96 tokens after a 32K-token prompt (median of 3). Aggregate decode is the wall-clock rate of concurrent greedy replies, capped at 96 tokens, to a fixed mix of varied short prompts (41 to 62 tokens each) that is the same for every set (median of 3). Draft acceptance depends on the prompt, so it is also reported for a single repeated prompt. Cold time to first token uses exactly the stated number of prompt tokens with nothing cached (median of 2). The 8-request time to first token is the median wait for the first token when 8 short prompts from the mix (41 to 62 tokens) are sent at once; without the draft model, prompts that arrive together are not read together in one batch, and first tokens arrive later (see known issues). The 8-stream token gap is the longest pause seen by running streams while a prompt of about 8,000 or 36,000 tokens joins, reported as the median and the maximum of that pause across runs. Draft acceptance is the number of draft tokens accepted per verify step, not counting the token the model adds itself, with accepted over proposed tokens alongside, both measured at 8 concurrent requests. The no-draft-model set is a reduced run. The benchmark client runs on a separate machine on the same local network, so client-side times include one network hop. The benchmark figures in the result tables, including the stall bounds, come from chat requests at low reasoning effort. Installation, disk and startup figures are not chat measurements. Low effort still reasons before it answers, and decode and aggregate rates count reasoning tokens. A chat request that sets no reasoning effort runs at high effort, so its replies are longer and its rates can differ from these. Time to first token is measured to the first streamed token, reasoning or text. In the cold-prompt, newcomer and saved-session tests, that first token was visible text in all but two replies: one 64K cold-prompt reply with base weights and the saved-session return with refusal-removed (ablit) weights reached their eight-token limit on reasoning and showed no text. In the eight-client short-prompt test, most replies began with a short reasoning passage, so visible text arrives later than the first token, and some short replies spent their 96-token limit on reasoning and showed no text; each set's first-token figure is shown with its own visible-text note.

Each weight variant ships its own measured settings profile. Each column heading names the profile behind its figures. "not measured" means the set was not run for that cell. The default configuration is marked.

## Result sets

- **V-D, base weights + draft model · base profile:** weights `TensorFold/GLM-5.3-Flash-MLX-4bit-MTP@76add2a341a1cd90ad0e86bb69839ea9c35827c6`; settings profile `base` (file `config/profiles/base.env`, sha256 `d42a35d299095872c29610002e9e8507f3c4205098be8759dc704342c1d71191`); combined data-manifest digest `5ec35938b1f3e6ef81fbd8b94e3773fe8368f798a2237ace5ac3c1781d7f3497` (sha256 of the three rank manifests in `manifests/base/`; the command is `data_manifest_recipe` in `results-v2.0.1.json`); draft model `incoai/GLM-5.3-Flash-DFlash2@bf582e4eacc1810f76656d1811693ff6c6737d2a`; engine `509bfa8f60b14f956f14a4c5a58973b3f119c545`; env file `config/serve.env`; status measured (see known issue 10).
- **O-D, refusal-removed (ablit) weights + draft model · ablit profile:** weights `orcarouter/GLM-5.3-Flash-Uncensored-MLX@c02a5f6fa06f0aa444877b44d19fd5c96390329f`; settings profile `ablit` (file `config/profiles/ablit.env`, sha256 `e2f7d289c22c1b127b26b8827e44d7ec3f72ae2f859682b19dec366d3e4e1220`); measured data-manifest digest `3ced26475a36ecd398f679dff559c95a61ff6284e55433cb890c73838f772b2e`, recorded when the set was measured; draft model `incoai/GLM-5.3-Flash-DFlash2@bf582e4eacc1810f76656d1811693ff6c6737d2a`; engine `509bfa8f60b14f956f14a4c5a58973b3f119c545`; env file `config/serve.env`; status measured (see known issue 10). A fresh conversion matches the measured data file for file, except two bookkeeping files on each host that record which conversion produced them. (Fresh conversion manifest sha256 `b9f92cdf6ef55d642d20cb8db2c71fcd0c6925909ccfbb0e75ebe7a6328e0f33`, which `fresh_conversion_manifest_recipe` in `results-v2.0.1.json` reproduces from `manifests/ablit/`.)
- **V-N, base weights, no draft model (commercial use) · base profile:** weights `TensorFold/GLM-5.3-Flash-MLX-4bit-MTP@76add2a341a1cd90ad0e86bb69839ea9c35827c6`; settings profile `base` (file `config/profiles/base.env`, sha256 `d42a35d299095872c29610002e9e8507f3c4205098be8759dc704342c1d71191`); combined data-manifest digest `5ec35938b1f3e6ef81fbd8b94e3773fe8368f798a2237ace5ac3c1781d7f3497` (sha256 of the three rank manifests in `manifests/base/`; the command is `data_manifest_recipe` in `results-v2.0.1.json`); draft model `none`; engine `509bfa8f60b14f956f14a4c5a58973b3f119c545`; env file `config/serve.env`; status measured (see known issue 10).

**All sets.** Every set ran on the same build with the same benchmark harness and protocol, each on its weights variant's shipped settings profile. Each set's receipt records its label, the combined digest of the three hosts' data manifests, the draft model, the settings profile and the engine commit.

The varied prompt mix is listed in `release/PROMPT-MIX.md` (document sha256 `1c86b45f9b4897b4a18888135870d492411f137302c9bbe79acf1067b00342a2`, list sha256 `6b8a3b45354059d6a73065cdc2b99d77bfdafb3c3d73a314823a5505e7cf9ba3`); the prompt texts ship as `release/PROMPT-MIX.jsonl` (sha256 `aed79029271045acfddf673433d368441343a87cdb25b723c9f16147f3aafacf`).

## RigMark

One RigMark block per weight variant.

**base weights + draft model · base profile**

```text
MODEL      glm53
APPLIANCE  3x NVIDIA DGX Spark (GB10), 128 GB unified memory each
RUN        reasoning=low  •  protocol=1.1.0
SOURCE     git:c5a0db01b054  •  clean
WORKLOAD       DECODE EST.      LAST OUTPUT          RANGE          BASIC GATE
CODE              91.3 tok/s      21.3s last     88.2–92.8     ✓ 5/5
PROSE             51.6 tok/s      19.3s last     50.3–53.1     ✓ 5/5
STRUCTURED*      127.9 tok/s       3.8s last    127.4–128.3    ✓ 5/5
* predictable-output ceiling; not a proxy for agent speed
64K PREFILL   cold 2,124 tok/s  •  immediate replay 821,037 tok/s
AGGREGATE   C1 64.8  •  C2 84.6  •  C4 113.4 tok/s   (short code, end-to-end, 256-token cap per agent)
C4 OUTPUT STATE   normal stop 0/12  •  visible 12/12  •  reasoning may be included
JSON       sha256:05051f88e880384c…
```

**refusal-removed (ablit) weights + draft model · ablit profile**

```text
MODEL      glm53
APPLIANCE  3x NVIDIA DGX Spark (GB10), 128 GB unified memory each
RUN        reasoning=low  •  protocol=1.1.0
SOURCE     git:c5a0db01b054  •  clean
WORKLOAD       DECODE EST.      LAST OUTPUT          RANGE          BASIC GATE
CODE              90.1 tok/s      23.8s last     88.4–92.0     ✓ 5/5
PROSE             50.8 tok/s      19.3s last     50.6–51.6     ✓ 5/5
STRUCTURED*      127.4 tok/s       3.8s last    121.0–127.9    ✓ 5/5
* predictable-output ceiling; not a proxy for agent speed
64K PREFILL   cold 2,129 tok/s  •  immediate replay 812,054 tok/s
AGGREGATE   C1 67.9  •  C2 93.8  •  C4 128.4 tok/s   (short code, end-to-end, 256-token cap per agent)
C4 OUTPUT STATE   normal stop 0/12  •  visible 12/12  •  reasoning may be included
JSON       sha256:1b42ea87215719ef…
```

Appliance comparison: different model IDs, not a same-weights claim. v1.8.4 ran with reasoning off, its default; v2.0.1 ran at reasoning effort low. Cold prefill and replay rows use raw token-ID completions, where reasoning effort does not apply.

| RigMark row | v1.8.4 | base weights + draft model · base profile | refusal-removed (ablit) weights + draft model · ablit profile |
|---|---:|---:|---:|
| C1 per-stream time to first token, seconds (lower is better) | 0.396 | 0.405 | 0.361 |
| Prose time to first visible text, seconds (lower is better) | 0.380 | 0.484 | 0.497 |

With base weights, v1.8.4 is about 2% faster on this row. The first token is visible text in every reply in each column.

With base weights, v1.8.4 shows prose text about 0.10 s sooner; v2.0.1 takes about 1.27x as long. RigMark's own prose time to first token marks the first reasoning token, not visible text, so it is not shown.

Prose row: at reasoning effort low, v2.0.1 writes a short reasoning passage before the visible text (with base weights, 11 to 12 tokens, about 1% of each reply of about 1,000 tokens); v1.8.4, with reasoning off, wrote none. RigMark counts those tokens in the prose decode rate and in last output time.
