# Benchmark prompt mix

The 16 prompts were written for this benchmark and contain no private data. They are fictional examples and general tasks. Their full message texts are in PROMPT-MIX.jsonl.

JSONL SHA256: `aed79029271045acfddf673433d368441343a87cdb25b723c9f16147f3aafacf`.
Canonical list SHA256: `6b8a3b45354059d6a73065cdc2b99d77bfdafb3c3d73a314823a5505e7cf9ba3`.

Canonical JSONL: one UTF-8 JSON object per line, sorted keys, compact separators, ensure_ascii=false, LF after every object. Canonical-list hash: SHA256 of that ordered array encoded with the same JSON settings but without a terminal newline. Each row also carries the SHA256 of its canonical messages array.

| Prompt ID | Topic |
|---|---|
| V-code/01-stable-unique | Python function |
| V-code/02-shell-pipeline | Shell explanation |
| V-code/03-sql-latest | SQL query |
| V-code/04-regex-slug | Regular expression |
| V-code/05-typescript-union | TypeScript type |
| V-code/06-mutable-default | Bug repair |
| V-code/07-json-tool-call | Structured JSON |
| V-code/08-unit-test-boundary | Unit test |
| V-chat/01-meeting-email | Email |
| V-chat/02-library-summary | Summary |
| V-chat/03-rainbow-explanation | Science explanation |
| V-chat/04-train-packing | Practical list |
| V-chat/05-translation | Translation |
| V-chat/06-ticket-arithmetic | Word problem |
| V-chat/07-rice-step | Cooking explanation |
| V-chat/08-opportunity-cost | Definition |

## Published concurrency assignment

The published c1/c2/c4/c8/c16 ladder uses all 16 prompts as its frozen source list. Seed = 20261002. For each family, sort the eight IDs by SHA256 of `20261002:<prompt-id>`, then interleave code/chat. For repetition r, rotate that 16-prompt list by `5*r mod 16` and take the first c prompts. This uses an explicit hash ordering, independent of Python random implementation, server build, policy or weights. Every published configuration uses exactly the same assignment for a given c and r. The frozen prompt texts in this file are never edited. Every benchmark request built from them gets a request-specific prefix at the start of its user message (no prompt in this mix has a system message). The form of the prefix differs between benchmark cells, and the stall test's newcomer requests also shorten or pad the prompt text to the prompt length being tested. In the concurrency rows, every request must report zero cached tokens in the server's `tensorfold.cached` count, and the receipts keep each request body exactly as sent.

Each c8 repetition has four code and four chat prompts; across its three repetitions all 16 prompts are covered. Each c16 repetition has all 16. Lower concurrency samples fewer prompts and is not an equal-total-work scaling experiment. Prompt IDs and hashes accompany raw receipts; the exact assignment appears in the results summary. Rates include all generated completion tokens, including reasoning, and aggregate timing includes TTFT and the full concurrent-wave wall time.

| Concurrency | Rep | Ordered prompt IDs |
|---:|---:|---|
| 1 | 0 | V-code/04-regex-slug |
| 1 | 1 | V-chat/01-meeting-email |
| 1 | 2 | V-code/06-mutable-default |
| 2 | 0 | V-code/04-regex-slug, V-chat/04-train-packing |
| 2 | 1 | V-chat/01-meeting-email, V-code/08-unit-test-boundary |
| 2 | 2 | V-code/06-mutable-default, V-chat/03-rainbow-explanation |
| 4 | 0 | V-code/04-regex-slug, V-chat/04-train-packing, V-code/05-typescript-union, V-chat/08-opportunity-cost |
| 4 | 1 | V-chat/01-meeting-email, V-code/08-unit-test-boundary, V-chat/06-ticket-arithmetic, V-code/03-sql-latest |
| 4 | 2 | V-code/06-mutable-default, V-chat/03-rainbow-explanation, V-code/01-stable-unique, V-chat/02-library-summary |
| 8 | 0 | V-code/04-regex-slug, V-chat/04-train-packing, V-code/05-typescript-union, V-chat/08-opportunity-cost, V-code/07-json-tool-call, V-chat/01-meeting-email, V-code/08-unit-test-boundary, V-chat/06-ticket-arithmetic |
| 8 | 1 | V-chat/01-meeting-email, V-code/08-unit-test-boundary, V-chat/06-ticket-arithmetic, V-code/03-sql-latest, V-chat/05-translation, V-code/06-mutable-default, V-chat/03-rainbow-explanation, V-code/01-stable-unique |
| 8 | 2 | V-code/06-mutable-default, V-chat/03-rainbow-explanation, V-code/01-stable-unique, V-chat/02-library-summary, V-code/02-shell-pipeline, V-chat/07-rice-step, V-code/04-regex-slug, V-chat/04-train-packing |
| 16 | 0 | V-code/04-regex-slug, V-chat/04-train-packing, V-code/05-typescript-union, V-chat/08-opportunity-cost, V-code/07-json-tool-call, V-chat/01-meeting-email, V-code/08-unit-test-boundary, V-chat/06-ticket-arithmetic, V-code/03-sql-latest, V-chat/05-translation, V-code/06-mutable-default, V-chat/03-rainbow-explanation, V-code/01-stable-unique, V-chat/02-library-summary, V-code/02-shell-pipeline, V-chat/07-rice-step |
| 16 | 1 | V-chat/01-meeting-email, V-code/08-unit-test-boundary, V-chat/06-ticket-arithmetic, V-code/03-sql-latest, V-chat/05-translation, V-code/06-mutable-default, V-chat/03-rainbow-explanation, V-code/01-stable-unique, V-chat/02-library-summary, V-code/02-shell-pipeline, V-chat/07-rice-step, V-code/04-regex-slug, V-chat/04-train-packing, V-code/05-typescript-union, V-chat/08-opportunity-cost, V-code/07-json-tool-call |
| 16 | 2 | V-code/06-mutable-default, V-chat/03-rainbow-explanation, V-code/01-stable-unique, V-chat/02-library-summary, V-code/02-shell-pipeline, V-chat/07-rice-step, V-code/04-regex-slug, V-chat/04-train-packing, V-code/05-typescript-union, V-chat/08-opportunity-cost, V-code/07-json-tool-call, V-chat/01-meeting-email, V-code/08-unit-test-boundary, V-chat/06-ticket-arithmetic, V-code/03-sql-latest, V-chat/05-translation |

## Regression cell

A separate single-prompt c8 cell (a CSV-parser coding prompt, three waves) is run as a regression check only. It is not part of the 16-prompt mix and has no published concurrency number. Its acceptance is reported next to the mix acceptance to show how acceptance depends on the workload.

Published concurrency numbers (`concurrency_aggregate_tok_s_c<n>`) and `c8_ttft_p50_s` come only from the 16-prompt mix; `c8_ttft_p50_s` is the median of the three per-repetition p50 TTFTs.
