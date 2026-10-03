# Benchmarks

This page explains how every speed number for JSpark3 v2.0.1 (GLM-5.3 Flash) was measured. For each number it gives what was timed, which client sent the requests, what the prompts looked like, how many times the cell ran, and which statistic is reported.

The numbers themselves are in the results file at the release tag ([`release/results-v2.0.1.json`](https://github.com/jakejharris/jspark3/blob/v2.0.1/release/results-v2.0.1.json)). Every number on the README, the release notes and the model card comes from that file.

## Rules for every number

- **Every number names its weights and draft model.** A number without a set label (for example "base weights + draft model") is not one of ours.
- **The headline is the median, on the base weights with the draft model.** It is never the best run.
- **Ranges are the lowest and highest run.** They are not confidence intervals. Most cells ran 2 to 5 times, which is too few for an error bar.
- **Only the shipped configuration was measured.** Every set ran on the same build, each set on its weights variant's shipped settings. Each weight variant ships its own measured settings profile. `serve.sh` loads the profile from the weights switch ([OPERATIONS.md](OPERATIONS.md#settings-profiles)). Beyond that, each set changed only the switch that names it: the weights or the draft model. The other engine settings were as installed, and the session store was on. Any other change, including an edited profile, is a configuration we did not measure.
- **Speed only.** None of these numbers says whether an answer is right. `scripts/exactness.py` checks the draft model's output against plain decoding separately (see [OPERATIONS.md](OPERATIONS.md#is-it-ready-is-it-right)).
- **Three DGX Sparks on a direct high-speed (RDMA) link.** Other hardware, cabling, other work on the boxes or a different host memory state will give different numbers.

## The three result sets

Every set ran on the same build with the same benchmark harness and protocol, each on its weights variant's shipped settings profile. Each set's receipt records its label, the combined digest of the three hosts' data manifests, the draft model, the settings profile and the engine commit. That harness is not part of this source package. For each request in the concurrency rows (the varied prompt mix), the receipt also records the prompt's ID, the hash of its frozen messages and the server's `tensorfold.cached` count, and keeps the request body exactly as sent. The shipped scripts check each box, readiness, what is running and correctness (`scripts/preflight.py`, `scripts/wait-ready.sh`, `scripts/status.sh`, `scripts/smoke.sh`, `scripts/exactness.py`). `scripts/cache-check.py` checks that the next turn of a conversation reuses the earlier prompt and that a new prompt reuses nothing, and `scripts/prefill-check.py` checks that cold prompt reading at about 8,000 and 32,000 tokens meets a floor set in the script. None of these scripts produced the numbers on this page.

| Set | Weights | Draft model | Settings profile | What it measures | Why it exists |
|---|---|---|---|---|---|
| **base weights + draft model** | base (`TensorFold/GLM-5.3-Flash-MLX-4bit-MTP`, formerly `Vontra/GLM-5.3-Flash-MLX-4bit-MTP`, which redirects; MIT) | `incoai/GLM-5.3-Flash-DFlash2` (CC BY-NC-ND 4.0, non-commercial) | base | every metric below | the default install, and the headline |
| **refusal-removed (ablit) weights + draft model** | ablit (refusal-removed, converted locally from a gated source) | same draft model | ablit | every metric below | the opt-in install for ablit development, red-teaming and refusal research |
| **base weights, no draft model (commercial use)** | base | none: the weights' own built-in multi-token prediction head | base, with the built-in head's drafting setting | a reduced set: short decode, cold 8k and 32k, aggregate at 1 and 8 requests | the draft model's license is non-commercial; this set runs without it, which is the commercial path ([README, License](../README.md#license)) |

- The ablit numbers are measured with the same protocol and client as base, on the ablit settings profile, and published alongside them. Both draft-model sets use the same drafting setting (the two profiles differ only in the setting used without the draft model), so a difference between them comes from the weights and from how well the draft model's guesses match them (see [draft acceptance](#draft-acceptance)).
- The no-draft-model set is a separate start with `--drafter none`, not the default start with a request option.

## Conditions

These apply to every set.

- **Build:** the same release build for every set, at engine commit `509bfa8f60b14f956f14a4c5a58973b3f119c545`, started by `scripts/serve.sh` with no options beyond the set's weights and draft-model switches.
- **Weights data:** the base sets ran on base data whose combined per-host manifest has sha256 `5ec35938b1f3e6ef81fbd8b94e3773fe8368f798a2237ace5ac3c1781d7f3497`; the ablit set ran on data with `3ced26475a36ecd398f679dff559c95a61ff6284e55433cb890c73838f772b2e`. A fresh conversion's combined manifest is `b9f92cdf6ef55d642d20cb8db2c71fcd0c6925909ccfbb0e75ebe7a6328e0f33`. A fresh conversion matches the measured data file for file, except two bookkeeping files on each host that record which conversion produced them.
- **Quiet endpoint:** no other client used the server during a run.
  - The benchmark script reads the server's request counters before and after every phase. It stops if it sees a request it did not send.
  - Only one benchmark ran at a time.
- **Client:**
  - The benchmark client runs on a separate machine on the same local network, so client-side times include one network hop.
  - It reached the API on rank 0 through an SSH port forward.
  - The cells on this page came from our release benchmark script, a Python program that uses only the standard library's HTTP client. The script does not ship with the release; this page documents its protocol.
- **Requests:**
  - Every request from the benchmark script streamed its reply and set `"stream_options": {"include_usage": true}`, so token usage was reported at the end.
  - Sampling was greedy (temperature 0).
  - No request changed a server setting.
- **Reasoning effort:** this release always thinks; the lowest effort is Low (see [OPERATIONS.md](OPERATIONS.md#upgrading-from-v18x-and-rolling-back-to-v184)).
  - The benchmark figures in the result tables, including the stall bounds, come from chat requests at low reasoning effort. Installation, disk and startup figures are not chat measurements. Low effort still reasons before it answers, and decode and aggregate rates count reasoning tokens. A chat request that sets no reasoning effort runs at high effort, so its replies are longer and its rates can differ from these. Time to first token is measured to the first streamed token, reasoning or text. In the cold-prompt, newcomer and saved-session tests, that first token was visible text in all but two replies: one 64K cold-prompt reply with base weights and the saved-session return with refusal-removed (ablit) weights reached their eight-token limit on reasoning and showed no text. In the eight-client short-prompt test, most replies began with a short reasoning passage, so visible text arrives later than the first token, and some short replies spent their 96-token limit on reasoning and showed no text; each set's first-token figure is shown with its own visible-text note.
  - The low-effort cells send `"chat_template_kwargs": {"enable_thinking": false}`, which renders as Low.
  - The server reports no separate count of reasoning tokens, so no rate here can leave them out.
- **Order within a set:**
  - On the two draft-model sets, the cold prompts ran first and a separate end-to-end benchmark ran last, after every other cell.
  - Later cells therefore ran on a server that had already served requests.
  - Each cold prompt carries a unique salt and must report zero cached tokens in the server's `tensorfold.cached` count, so earlier traffic cannot help it.
- **Caches:** the server reuses work for a prompt that continues one it read recently, or, with the draft model on, exactly repeats one ([OPERATIONS.md](OPERATIONS.md#the-session-store-what-it-keeps-and-how-to-clear-it)).
  - Every row labeled cold, and every time to first token at a stated prompt length, must report zero cached tokens in the server's `tensorfold.cached` count, or the rep does not count.
  - The concurrency rows give every request a unique nonce and must report zero cached tokens in that count too. See [throughput](#throughput-with-several-requests-concurrency_aggregate_tok_s).
  - The decode-rate rows repeat their prompt, so they can be served from the cache.
- **Kernels were warm.** "Cold" on this page means nothing of the prompt was cached. It does not mean a first start: GPU code compiled during the first start is not in any number here. OPERATIONS gives first-start times.

## Metric definitions

The name in each heading is the metric's key in the results file. Each set has its own figures under that key. Where a metric has several figures, its sub-keys are listed. "Rep" means one repetition of the cell.

### Decode speed, short prompts (`decode_short_tok_s`)

- **What it answers:** once a single reply is streaming, how fast does the text come?
- **Two figures, never combined:**

  | Key | Prompt |
  |---|---|
  | `decode_short_tok_s.code` | a fixed short code-writing prompt |
  | `decode_short_tok_s.prose` | a fixed short prose prompt |

  - Prose usually decodes more slowly than code, because the draft model guesses it less well. One combined figure would hide the slower rate, so the two are always reported apart.
- **Shape:** one request at a time; output capped at 256 tokens; greedy.
- **Timed:** `(completion tokens − 1) / (time of the last streamed chunk − time of the first)`, for one stream.
  - It leaves out the time to the first token, so prompt reading is excluded.
  - Chunks can carry more than one token, so this is an estimate from the client side, not the server's own decode clock.
- **Reps and statistic:** 3 reps per prompt; each figure is the median of its 3.
- **Cache:** reps 2 and 3 send exactly the prompt of rep 1. The cache saves only prompt reading, and this rate leaves out the time to the first token.
- Single-request cells on other short prompts, capped at 96 tokens, are in the results file as secondary cells. They are not this number.

### Decode speed, long prompt (`decode_long_tok_s`)

- **What it answers:** how fast does a reply stream when the conversation is already long?
- **Prompt:** about 32,000 tokens of padding, then the fixed code prompt. The receipt records the actual prompt length.
- **Shape:** one request; greedy; output capped at 96 tokens. The rate covers the start of a reply, not a long answer.
- **Timed:** the same chunk-timed, single-stream rate as short decode.
- **Reps and statistic:** 3 reps; median.
- **Cache:** reps 2 and 3 send exactly the prompt of rep 1. The rate leaves out the time to the first token.

### Time to first token, cold (`cold_ttft_s`)

- **What it answers:** how long you wait for the first token when the server has none of your prompt cached.
- **Prompts:** exactly 8,000, 32,000, 64,000 and 128,000 prompt tokens. Sub-keys: `cold_ttft_s.8k`, `.32k`, `.64k`, `.128k`.
  - Each prompt is sized with the server's `/tokenize` endpoint before timing. The server's reported prompt length must match exactly.
  - Each prompt carries a unique salt. The server's `tensorfold.cached` count must be exactly 0, or the rep does not count.
- **Shape:** one request; output capped at 8 tokens.
- **Timed:** request start to the first streamed token, at the client.
- **Reps and statistic:** 2 different prompts per size; median of the 2.
- The results file also records:
  - the server's own prefill time;
  - prompt tokens over that time;
  - prompt tokens over the time to first token.
- There is no 16k cell.

### Throughput with several requests (`concurrency_aggregate_tok_s`)

- **What it answers:** with several users at once, how much text does the server produce in total?
- **Prompts:** short prompts (41 to 62 tokens, plus a 24-token prefix unique to each request), cache misses asserted. The prompts are a fixed mix, the same for every set. The mix is seeded, so each request gets the same prompt on every set. It is built from 16 original prompts, 8 coding and 8 chat. The 16 prompts were written for this benchmark and contain no private data. Their full texts ship with the results as `PROMPT-MIX.jsonl` (`release/PROMPT-MIX.jsonl`, sha256 `aed79029271045acfddf673433d368441343a87cdb25b723c9f16147f3aafacf`). `PROMPT-MIX.md` (`release/PROMPT-MIX.md`, sha256 `1c86b45f9b4897b4a18888135870d492411f137302c9bbe79acf1067b00342a2`) gives the exact composition.
- **Nonce:** Each request in the concurrency rows (the varied prompt mix) starts with a line specific to that request, added at the start of its first message as it is sent, so no two of those requests are identical. Some other rows reuse request bodies or prefixes on purpose (exact replay, and returning to a cached conversation); they are reported in their own rows and are not repetitions of the concurrency rows. Every request must report zero cached tokens in the server's `tensorfold.cached` count, or the rep does not count.
- **Why a mix:** how often the draft model's guesses are kept depends on the prompt. Repeating one prompt in every request turns that one prompt's rate into the whole result, so two weight variants can look further apart, or closer, than they are on varied traffic.
- **Shape:** 1, 2, 4, 8 or 16 requests released together; each capped at 96 tokens; greedy. Sub-keys: `concurrency_aggregate_tok_s.c1`, `.c2`, `.c4`, `.c8`, `.c16`.
- **Timed:** the group's total completion tokens, divided by its wall time from the common start to the end of the last reply. Time to first token and time spent waiting in the queue are included.
- **Reps and statistic:** 3 reps per level, c1 to c16; median.
- The server runs 8 requests at a time. At 16, the second 8 wait for slots, and that wait is part of the number. The 16-request cell measures the server under queueing, not 16 parallel streams.
- This is a total, not the speed one user sees. With 8 requests, each stream gets roughly an eighth of it.

### Time to first token with 8 requests (`c8_ttft_p50_s`)

- **What it answers:** at 8 requests at once, how long does a typical user wait for the first token?
- **Shape:** the 8-request throughput cell above: 8 short prompts from the mix (41 to 62 tokens, plus a 24-token prefix unique to each request) sent at once, cache misses asserted.
- **Statistic:** in each rep, the median time to first token across the 8 streams. The reported value is the median of the 3 reps' medians.
- With the draft model on, short text prompts that arrive together can be read together in one pass. Without it, they are not read together in one batch, and first tokens arrive later (see [request scheduling limitations](TROUBLESHOOTING.md#api-behaviour-clients-notice)).

### Stall when a long prompt arrives (`c8_stall_s`)

- **What it answers:** when someone sends a long prompt, how long do replies already in progress pause?
- **Shape:** 7 replies already streaming, then an 8th request arrives with a prompt of about 8,000 or 36,000 tokens.
  - The newcomer's prompt is uncached.
  - The 7 streams must still be producing tokens while the newcomer is read, or the rep does not count.
- **Timed:** the longest gap between two consecutive streamed chunks on any of the 7 streams.
- **Reps and statistic:** 3 reps per newcomer size. Sub-keys:
  - `c8_stall_s.median`: the median of the per-rep longest gaps;
  - `c8_stall_s.max`: the single longest gap seen.

### Draft acceptance

- **What it answers:** how many of the draft model's guessed tokens the main model keeps. More kept tokens means fewer slow steps per token, so faster decode.
- **Primary** (`accepted_per_verify_step`): accepted draft tokens per verification step. This does not count the token the main model adds itself at every step. Projects that count that token report a number 1 higher for the same run.
- **Also reported** (`accepted_over_proposed`): accepted draft tokens divided by proposed draft tokens.
- **How it is pooled:** both are totals over a whole rep (all accepted tokens over all verification steps, or over all proposed tokens), not an average of per-request figures.
- **Source:** the server's own drafting statistics. A client cannot see which tokens were drafted.
- **Where reported:** at 8 concurrent requests, once on the prompt mix and once on one short prompt asking for Python code, repeated in every request. Only acceptance is published for the repeated prompt, not its speed. It shows how far the weight variants differ on a single prompt. Sub-keys: `draft_acceptance.mix.*` and `draft_acceptance.single_prompt.*`, each with both figures above.
- **Statistic:** median over the cell's 3 reps.
- Acceptance is reported for the two sets that use the draft model. The no-draft-model set reports none.
- Acceptance depends on the text. Structured or repetitive output accepts more; open prose accepts less. That is why the throughput cells use a [prompt mix](#throughput-with-several-requests-concurrency_aggregate_tok_s).

### Context window (`max_context_tokens`)

- 262,144 tokens: the configured `--context` of the shipped start.
- This is a setting, not a measurement. A prompt near the full window can wait in the queue until other long requests finish (see [OPERATIONS.md](OPERATIONS.md#capacity)).

### Returning to a conversation that left memory (`session_tier.evicted_return_s`)

- **What it answers:** you come back to a long conversation after other long conversations pushed it out of memory. How long until the first token?
- **Measured on** the base weights + draft model set and the refusal-removed (ablit) weights + draft model set. The no-draft-model set does not have this cell. Without the draft model, a long conversation that includes images may not be saved to disk; that is a known issue ([TROUBLESHOOTING.md](TROUBLESHOOTING.md#a-long-conversations-next-reply-takes-as-long-as-the-first)), not a measured figure.
- **Shape:**
  1. Send a conversation of about 100,000 tokens that includes images, capped at 8 output tokens. Its time to first token is the fresh-read figure.
  2. Send three more distinct conversations of about 100,000 tokens. Together they are larger than the 5 GiB in-memory store on each box, so the first conversation leaves memory. Its saved copy stays on disk.
  3. Send the first conversation again with one more turn.
- **Timed:** the time to first token of step 3.
- **Checks:** on every run, step 3's `tensorfold.cached` count must be at least 100,000 and its `tensorfold.session_cache_source` must be `disk`. The receipt records both.
- **Reps:** one sequence per set.
- Our runs also include a **warm return** cell: a long conversation followed at once by its next turn, served from memory. That is a different situation and a different number. Do not quote it as the evicted return.

## Comparing with earlier releases

- v1.8.4's `docs/BENCHMARKS.md` used a different instrument, so none of its rows is the same measurement as a row here. Do not divide one by the other.
  - Its decode figures are totals across 1 to 8 requests, forced to 512 tokens with thinking off.
  - Its ranges come from two sweeps in one start.
  - Its prefill figure extends an already cached prefix, so it is not a cold prefill either.
- v2.0.0 was an internal build and was never published, so there are no v2.0.0 numbers to compare.
- No earlier build was measured on this prompt mix, so this page compares no throughput or 8-request time-to-first-token figure with an earlier release.
- RigMark is the only protocol run on both releases. See [RigMark](#rigmark), below.

## Running the checks yourself

- `scripts/smoke.sh` and `scripts/exactness.py` check correctness, not speed. See [OPERATIONS.md](OPERATIONS.md#is-it-ready-is-it-right).
- The release benchmark script does not ship. The definitions above are its protocol: prompt shape, cap, reps and statistic for every cell.
  - You can measure the same quantities with your own streaming client.
  - Your prompts will differ from ours, so treat the comparison as approximate. For throughput, use our published prompt mix (`PROMPT-MIX.jsonl`), or at least varied prompts rather than one prompt repeated, for the reason given under [throughput](#throughput-with-several-requests-concurrency_aggregate_tok_s). Give every request a unique prefix as `PROMPT-MIX.md` describes; otherwise repeated prompts are served from the cache.
  - Run on a quiet server with the shipped configuration, settings profile unedited, and check that `scripts/status.sh` on each box shows the weights, draft model and session store setting of the set you compare against.

## RigMark

RigMark is a public benchmark client for OpenAI-compatible servers (github.com/alexellis/rigmark). We ran it unmodified, with the same settings as our RigMark run of v1.8.4, so the two appliances can be compared on one protocol.

- RigMark is the public way to reproduce the RigMark blocks below; the release benchmark script that produced the results table does not ship.

### Protocol

| Setting | Value |
|---|---|
| Client | RigMark at commit `c5a0db01`, protocol 1.1.0, prompt corpus 1.0.0, Python 3.12, unmodified clean checkout |
| Where it ran | the same separate client machine as the other cells, through an SSH tunnel to rank 0 |
| Model | `--model auto` (resolves to `glm53`) |
| Request extra | `{"chat_template_kwargs":{"reasoning_effort":"low"}}`, so Low effort |
| Sampling | temperature 0, top_p 1, seed 20260905 |
| Decode workloads | code, prose and structured JSON; 5 runs each; up to 4,096 output tokens |
| Prefill | 8K, 32K and 64K token-ID prompts, sent as raw completions, which carry no reasoning-effort instruction; 3 cold / immediate-replay pairs each; 8 output tokens |
| Concurrency | 1, 2 and 4 requests at once; 3 rounds; short code prompt; 256-token cap per request |
| Statistic | median; RigMark never reports the best run as the headline |
| Comparison ID | `2026-09-05-glm-tp2-tp4-rigmark-v1`, as in the v1.8.4 run. RigMark derives each prompt's unique salt from it, so both runs sent the same prompts. |
| Quiet check | no running or waiting requests before and after; all three ranks up; no restarts |
| Timeout | 600 s per request |

### How to read the block

| Line | Meaning |
|---|---|
| `DECODE EST.` | median over 5 runs of `(completion tokens − 1) / (last output event − first output event)`. It excludes prompt reading and includes reasoning tokens. |
| `LAST OUTPUT` | median time from request start to the last streamed output, reasoning included. This is the wait a user sees. |
| `RANGE` | lowest and highest of the 5 decode runs. Not a confidence interval. |
| `BASIC GATE` | runs that returned a non-empty visible answer and stopped normally. Structured runs must also match every requested value. 15 gates in all: 5 runs × 3 workloads. It catches truncated, empty or reasoning-only replies. It does not check that code works or that prose is right. |
| `STRUCTURED*` | a ceiling: the output is predictable, so the draft model guesses well. It is not a measure of agent or chat speed. |
| `64K PREFILL cold` | prompt tokens divided by time to first token, for a 64K token-ID prompt with a unique salt so nothing is cached. "Cold" means cache-busting, not a cold start. |
| `immediate replay` | the same token IDs sent again right away. RigMark does not check that a cache hit happened; it reports the rate. With the draft model on, this release does not read an exact repeat again ([OPERATIONS.md](OPERATIONS.md#the-session-store-what-it-keeps-and-how-to-clear-it)), so this row times a replay, not a prompt read. |
| `AGGREGATE` | total tokens per second across 1, 2 or 4 requests released together, including time to first token and scheduling. |
| `C4 OUTPUT STATE` | how the 12 four-request replies ended: how many stopped normally rather than at the 256-token cap, and how many had visible text. |
| `JSON sha256` | a fingerprint of the full receipt file. With it you can check that you hold the same receipt we do. It is not a signature, and it does not prove who ran the test. |

### Blocks

- base weights + draft model:

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

- refusal-removed (ablit) weights + draft model:

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

- The no-draft-model set has no RigMark run.

### Against v1.8.4

Appliance comparison: different model IDs, not a same-weights claim. v1.8.4 ran with reasoning off, its default; v2.0.1 ran at reasoning effort low. Cold prefill and replay rows use raw token-ID completions, where reasoning effort does not apply.

| Measure | v1.8.4 | v2.0.1, base weights + draft model | v2.0.1, refusal-removed (ablit) weights + draft model | Note |
|---|---|---|---|---|
| C1 per-stream time to first token, seconds (lower is better) | 0.396 | 0.405 | 0.361 | With base weights, v1.8.4 is about 2% faster on this row. The first token is visible text in every reply in each column. |
| Prose time to first visible text, seconds (lower is better) | 0.380 | 0.484 | 0.497 | With base weights, v1.8.4 shows prose text about 0.10 s sooner; v2.0.1 takes about 1.27x as long. RigMark's own prose time to first token marks the first reasoning token, not visible text, so it is not shown. |

Prose row: at reasoning effort low, v2.0.1 writes a short reasoning passage before the visible text (with base weights, 11 to 12 tokens, about 1% of each reply of about 1,000 tokens); v1.8.4, with reasoning off, wrote none. RigMark counts those tokens in the prose decode rate and in last output time.

This compares two appliances, not two kernels. Between the runs, all of these changed:

- the serving engine: vLLM, then a fork of TensorFold 0.3.6.2 (MIT);
- the weights (EXL3 4-bit files, then 4-bit MLX-format weights split across the three boxes);
- the draft policy, the scheduler and the chat template.

Three differences change how the rows read:

- **Reasoning.** v1.8.4 had thinking off: its replies had no reasoning text. v2.0.1 always thinks, here at Low effort in the chat workloads.
  - Reasoning tokens count in v2.0.1's decode rate and in its last-output time.
  - Compare `LAST OUTPUT` for time to a finished answer. Compare `DECODE EST.` only as token speed.
- **Request timeout.** The v2.0.1 runs allowed each request 600 s; the v1.8.4 run allowed 1,800 s. A timeout is a failed run, never a fast one.
- **Repeated prompts.** Some RigMark rows send a prompt more than once, `immediate replay` among them, and RigMark does not report whether a reply was served from a cache. v1.8.4 reused the start of a repeated prompt; v2.0.1, with the draft model on, also skips an exact repeat. Either side's rows can include cache effects, so do not read the 1-request `AGGREGATE` row as a gain.

The protocol, prompts, seeds, caps and run counts are the same. The served model name differs (`glm-5.3-flash`, then `glm53`); `--model auto` picks it up.

### What RigMark does not measure

- Whether generated code works or answers are right. The basic gate is not a correctness check.
- High-effort requests, which are this release's default.
- Prompts longer than 64K, or decode on a long conversation. Use the long-prompt and cold-prompt cells above.
- More than 4 requests at once. Use the throughput cells above.
- Tool calls or images.
- Returning to a conversation that left memory. Use the [evicted-return cell](#returning-to-a-conversation-that-left-memory-session_tierevicted_return_s) above.

### Run RigMark yourself

Check out RigMark at commit `c5a0db01` and run it from a machine with an SSH tunnel to rank 0 (see [OPERATIONS.md](OPERATIONS.md#reaching-the-api-safely)):

```sh
./rigmark run --base-url http://127.0.0.1:8002 --model auto --label <your label> \
  --comparison-id 2026-09-05-glm-tp2-tp4-rigmark-v1 --metadata metadata.json \
  --extra-body '{"chat_template_kwargs":{"reasoning_effort":"low"}}'
```

- `metadata.json` describes your appliance. RigMark's `METADATA.md` lists its fields.
- Use that exact commit and our comparison ID. RigMark builds each prompt's salt from its protocol version and the comparison ID, so another version or ID sends different prompts.
- Run it on a quiet server, after the ranks have been up for a while.
- Your result is comparable with ours only if you serve the same configuration, with the settings profile unedited. Check that `scripts/status.sh` on each box shows the same weights, draft model and session store setting as the block you compare against.
