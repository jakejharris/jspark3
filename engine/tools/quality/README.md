# Quality checks

Capture a baseline before evaluating a candidate. Keep the request fixtures, tokenizer, and harness fixed, and retain both JSON receipts.

## Commands

Use the actual served model's `tokenizer.json` path. The baseline freezes full token IDs with the served tokenizer. The candidate replays those same IDs and scoring start, so tokenization cannot silently change.

```sh
python3 tools/quality/quality.py --mode baseline --tokenizer /path/to/served/tokenizer.json --url http://127.0.0.1:8002 --output baseline.json
python3 tools/quality/quality.py --mode candidate --tokenizer /path/to/served/tokenizer.json --url http://127.0.0.1:8002 --baseline baseline.json --output candidate.json
```

The default deadline is 1,800 s, with a 180 s request timeout. A timeout, missing scorer, malformed score, changed token IDs, tokenizer mismatch, or truncated completion stops the run. Never silently omit a domain. `--output` is written after all items complete. Do not run the long-context items outside an approved window. The desk fake can be launched with `python3 tools/quality/fake_server.py --port 18742`; point both commands to that port for a protocol dry run. It uses synthetic scores and deterministic answers, and has no GPU or model. `--tool-prose` and `--tool-stop` are negative controls: each returns correct tool arguments while violating the bare-call contract. The v2 fixture identity requires a fresh baseline before a candidate comparison.

## Scoring endpoint

The original server returned `logprobs: None` for text completions. This branch adds a default-off `TF_GLM_QUALITY_SCORE=1` hook and `POST /v1/quality/score` in the CUDA HTTP adapter. The route accepts loopback clients only. It uses the same prefill math and weights as serving, with no sampling or drafts, and returns true full-vocabulary teacher-forced NLL in **natural log units** for each scored token. Ranking/top-k logprobs cannot stand in. It accepts at most 33,024 input tokens and 256 scored continuation tokens per request and keeps the ordinary serving path unchanged when off. This is an instrumentation hook, not a serving configuration; TP3 hash parity and overhead must be checked before its baseline is trusted.

Request: `{ "model": "glm53", "token_ids": [ ... ], "score_start": N }`. Return `{ "token_ids": [ ... ], "score_start": N, "nll": [ ... ] }`. `nll[k]` is `-log p(token_ids[score_start+k] | token_ids[:score_start+k])`. Reject empty continuations, invalid IDs, and truncation. The harness tokenizes the full text with the actual served tokenizer, picks the first token changed at the prefix/continuation boundary, and freezes that exact sequence and boundary in its baseline artifact. This handles tokenizers that merge a trailing prefix space with the first continuation word. The scorer reports all scored rows, including special tokens if present, and scores its first target token from the preceding context. It does not round values before sending JSON. The harness checks the response shape, exact token IDs, and finite nonnegative NLL. The scorer reuses existing prefill buffers and projects one head row at a time; its extra FP32 vocabulary row and reduction temporaries are estimated below 1 MiB per rank.

## Frozen gate, decided before candidate data

The NLL set has 20 paired synthetic sequences held out from kernel tuning, four each in code, prose, reasoning, tool JSON, and 32k-context language. They are small to fit the window and are not a public benchmark. Each sequence contributes one mean NLL. For candidate minus baseline, a deterministic 5,000-resample paired bootstrap yields one-sided 95% upper bounds. The overall upper bound must be `<= +0.010` nats/token; every domain bound must be `<= +0.020`. Small groups make per-domain bounds conservative; preserve the frozen margins. These bounds are a practical screening rule, not proof of general quality.

Four coding problems run returned Python functions against hidden edge cases in a child process with AST restrictions, 2 CPU seconds and 256 MiB address space. This checks execution rather than textual resemblance. The gate needs at least 3/4 and no regression on any baseline pass. Four forced tool calls require a single bare `tool_calls` object with the exact name and parsed arguments: 4/4. Four frozen recall prompts comprise single/multiple needles at approximately 32k and 128k model tokens; answers are exact keys in order. The gate needs at least 3/4 and no regression on any baseline pass. Baseline should be 4/4 on coding, tools and recall; any weaker baseline is a diagnostic before comparisons. The local tokenizer counts and server `usage.prompt_tokens` must agree within 1% when usage is returned. The chat template overhead is included in that tolerance.
