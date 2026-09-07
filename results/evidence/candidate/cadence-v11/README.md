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
  test. The later integrated live verification passed one pinned request past
  the old boundary; maximum-context and sustained-concurrency capacity remain
  unverified. This admission result is separate from the measured populations.

## Validation

`tools/validate_release.py` recomputes every entry of the `display` map from
the structured fields in `CLAIMS.json`, enforces the sham/diagnostic and
non-replication facts, and admits exactly these display values (with their
evidence classes) into the public-prose claim reconciliation. Any drift
between prose, the display map, and the structured fields fails the release
validator.

## Integrated live verification

[LIVE-VERIFY.json](LIVE-VERIFY.json) is a separate, sanitized projection of
one accepted full `fleetctl verify` receipt. It is admission evidence, not a
new benchmark population. The original remains in the private runtime
bundle with SHA-256
`1fbba426b8ac7ef9dc7b019241a70351335e82ebc8064c92a19c310e7db628d6`.
The export removes the start-manifest hash, original payload hash, per-rank
image-receipt hashes, and container names from the receipt projection.
Original receipt hashes remain explicitly labeled provenance for reviewers
holding the private bundle; they are not checksums of the sanitized file.
No prompts, generated content, host paths, addresses, container IDs, or raw
logs are distributed. Every remaining receipt field is copied verbatim.

The integrated result is `VERIFY_PASS`: all-rank runtime/image identity,
safety, complete Cadence capture evidence, load, health, arithmetic, focused
witness, and strict long-context witness passed. The pinned request reported
48,957 server prompt tokens and 51 completion tokens with the codeword
returned verbatim. The focused median was 73.8718 tok/s for its fixed
admission workload only. This is neither a Pi speed result nor a comparative
performance estimate. Pi slowdown remains unresolved; E3 is separate and
contributes no evidence to this release.

The running candidate is commit
`a729583fc1e286583023f4b9b92db87efefcb33a`, recipe SHA-256
`32784962571dfaa21634d8d290b011b3bbbe36ac8bdeaae0ceff5bd38fed7627`.
The host verifier is commit `456a2624d0c54e2096327b57fcb54e8333f10114`,
recipe SHA-256
`afd1fca1f58bcece36a441e2479bc41f39504fcc0c4cad45ae9905e629e62a43`.
[CANDIDATE-SHA256SUMS](CANDIDATE-SHA256SUMS) is the byte-exact checksum
manifest from the candidate commit. Comparing it with the packaged
`recipe/SHA256SUMS` proves that only `scripts/fleetctl.py` changed; the serving
entrypoint, modules, transforms, contracts, image/model pins, and serving
arguments remained identical. The host changes isolate remote JSON from
site hooks, bind an explicitly declared candidate recipe, recognize a real
generated module cache without relaxing source pins, and correct the stale
transform target-set aggregate. These are host controller/verifier changes;
they do not prove cold-start behavior of the final archive.

Earlier verification attempts refused on site-hook JSON contamination,
generated cache inventory handling, and the stale transform aggregate. Their
raw refusal receipts and logs remain in the private runtime bundle. The
stale aggregate was the exact v1.0.0 value
`ed7b0092e5a5a1d2aeb6dd2cbe9780783df89d70f733dff019dd05aa8cdd08bd`;
the candidate already had the correct current-contract value
`1f3beb88157da0a7782cc94d49bc5c8d93103fa708b620f8b3fb51f110a8f635`.
Fixing that verifier assumption required no candidate mutation. Earlier sham
failures, non-replicated code gains, quality failures, and historical
benchmark populations remain unchanged by this later admission pass.

The final package keeps the tested verifier recipe and adds sealing
evidence, documentation, and offline checks. Its source archive is exported
from the final commit; source, recipe, results, and SBOM asset checksums are
recorded together in the sealing handoff. It has not been cold-boot tested.
The configured maximum context, sustained concurrency, independent-fleet
reproduction remain unverified. The sealed candidate passed final review;
publication is authorized, with observed service writes recorded separately.

The release validator checks all integrated gate fields, recomputes the
focused median and public display values, checks the candidate checksum
manifest, and requires the packaged recipe to match the tested verifier
recipe. It also preserves the earlier uncertainty checks. A reviewer with
the private original can independently reproduce the documented projection;
the public export alone does not authenticate the original remote run.
