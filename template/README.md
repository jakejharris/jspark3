# Chat template

`chat-template.jinja` is the stock `chat_template.jinja` shipped with the weights, plus one inserted block.

- Source: [TensorFold/GLM-5.3-Flash-MLX-4bit-MTP](https://huggingface.co/TensorFold/GLM-5.3-Flash-MLX-4bit-MTP)
  (formerly `Vontra/GLM-5.3-Flash-MLX-4bit-MTP`, which redirects) at revision
  `76add2a341a1cd90ad0e86bb69839ea9c35827c6`, file `chat_template.jinja`.
- Stock sha256: `34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891` (the same entry as in
  `manifests/inputs/base-weights.sha256`).
- License: MIT, Copyright (c) 2026 Z.AI Co., Ltd. `LICENSE` here is the weights repo's `LICENSE` at the same revision
  (sha256 `30b85b6b9659f2e78aa259f8faf5d920a68dee7c9ced3fa6dba1f19f2bc4fca1`).

Our block is six lines, lines 3 to 8 inclusive. Lines 3 to 6: when a request sets `enable_thinking` or `thinking` to
false (the model still reasons) and its effective reasoning effort would be Max (no `reasoning_effort` of `low` or
`high`), the effort becomes Low instead. Lines 7 and 8: a request that gives no `reasoning_effort` at all and sets
neither to false answers at High instead of Max. Every stock line is unchanged: deleting those six lines gives back
the stock file byte for byte.
`pins.env` pins the result as `TEMPLATE_SHA256`, and `tests/check-template.py` checks both.
