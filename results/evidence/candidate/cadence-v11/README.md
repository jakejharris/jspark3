# Cadence v1.1 current-claims evidence export

`CLAIMS.json` is the single machine-readable source of record for every
current (v1.1.0, Cadence) numeric claim quoted in public prose. It exists so
the current claims have a distinct, traceable evidence file instead of an
arbitrary allowlist, and so the frozen historical v1.0.0 results
(`results/results.json`, mirrored byte-for-byte at
`huggingface/RESULTS.json`) are never rewritten by a later release.

## Provenance

The values are exported verbatim from the frozen measured-evidence record of
the measured Cadence run (a private ops-runtime document; its SHA-256 is
declared in `CLAIMS.json` under `provenance.source_sha256` so a reviewer
holding the private bundle can confirm the mapping). The underlying
preregistered paired design, raw records, and burst/rebuild reports remain in
the private evidence bundle; only the values needed to substantiate current
public claims are exported. Credentials, private host paths and addresses,
the boot ledger, and the non-publishable experiments are excluded by rule.

## Reading the uncertainty

- The first serving start failed its sham resolution control, so every
  first-start paired effect is **diagnostic only**. The second start's
  formally predeclared sham passes; its first-64 diagnostic still fails.
- The paired code gain is **not replicated**: the second start's interval
  spans zero.
- Conditional single-feature contrasts are different contrasts and must not
  be added or multiplied into a combined effect.
- The service batteries and rebuild figures are descriptive, not paired
  causal estimates; the archived pre-Cadence arm is not the published v1.0.0
  package.
- The quality battery shows candidate-only delivered-answer failures and its
  semantic confidence is inconclusive at every endpoint.
- The kernel transform's pinned output is verified by an offline construction
  test; the live single-stream witness above 32,768 tokens on the assembled
  public build is **pending**.

## Validation

`tools/validate_release.py` recomputes every entry of the `display` map from
the structured fields in `CLAIMS.json`, enforces the sham/diagnostic and
non-replication facts, and admits exactly these display values (with their
evidence classes) into the public-prose claim reconciliation. Any drift
between prose, the display map, and the structured fields fails the release
validator.
