# Contributing

JSpark3 v1 is a measured, hash-pinned serving recipe. Contributions are welcome
when they keep that property.

## Ground rules

- Every change to `recipe/` must keep `python3 tools/validate_release.py .`
  passing. The validator regenerates `recipe/SHA256SUMS`, checks the identity
  contracts, scans for private data, and runs the lifecycle dry-runs.
- Do not add model weights, tokenizer files, container layers, patched vLLM
  trees, `.env` files, or measurement captures that contain hosts, addresses,
  credentials, or container identities. `tools/validate_release.py` refuses
  them; so should you.
- A performance claim needs a receipt. Add the sanitized machine-readable
  evidence under `results/evidence/`, regenerate `results/results.json`, and
  quote only values that exist in its `display` map.
- Keep upstream credit intact. `THIRD_PARTY_NOTICES.md` and
  `REQUIRED_ATTRIBUTION.md` are byte-stable; changes to them are a
  maintainer-only, legal-review item.

## Workflow

1. Fork, branch, change.
2. Run `python3 tools/validate_release.py . --report validation.json`.
3. Open a pull request. CI runs the same validator.
4. Recipe changes that alter serving behaviour need a fresh three-node
   verification receipt (`verify.json`) attached to the pull request, with
   hosts and addresses redacted.
