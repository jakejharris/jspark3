# v1.8.4 measurement record

The structured [record](results-v1.8.4.json) contains the v1.8.4 serving
measurements and their source, native, policy and boot identities. Historical
v1.8.0 results remain byte-identical in their existing files; those rates do
not transfer to this release's qualified native bytes.

## Parity with v1.8.0

v1.8.4 serves at parity with v1.8.0, with cooperative MoE on by default. Of the
12 measured cells below, 8 overlap the frozen v1.8.0 range, 2 are above it and
2 are below it. Code decode at two streams was not measured. The coop on/off
comparison is pending; no uplift is claimed.

| Cell | v1.8.4 measured (tok/s) | v1.8.0 frozen (tok/s) | Relation |
|---|---|---|---|
| structured c1 | 96.7-97.1 (n=2) | 97.5-98.7 (n=2) | below |
| structured c2 | 120.1-120.5 (n=2) | 94.0-142.9 (n=2) | overlaps |
| structured c4 | 170.9-213.9 (n=2) | 168.6-229.7 (n=2) | overlaps |
| structured c8 | 202.7-220.9 (n=2) | 198.0-198.6 (n=2) | above |
| prose c1 | 44.7-46.1 (n=2) | 44.3-49.1 (n=2) | overlaps |
| prose c2 | 63.7-65.5 (n=2) | 61.6-63.5 (n=2) | above |
| prose c4 | 86.0-88.4 (n=2) | 83.3-90.7 (n=2) | overlaps |
| prose c8 | 105.6-106.4 (n=2) | 110.1-110.8 (n=2) | below |
| code c1 | 65.2-84.7 (n=5) | 68.2-73.3 (n=2) | overlaps |
| code c2 | not measured | 101.0-103.9 (n=2) | n/a |
| code c4 | 136.2-139.5 (n=5) | 136.9-141.7 (n=2) | overlaps |
| code c8 | 164.2-180.9 (n=5) | 174.0-179.7 (n=2) | overlaps |
| prefill (Pi turns, post-hygiene) | 1197.1-1273.4 (n=8) | 1195.0-1262.6 (n=8) | overlaps |

Decode is aggregate tok/s, with 512 forced tokens per stream and streams started
together. Structured and prose ran 2 sweeps; code ran 5 repeats at temperature 0,
and c2 was not run. Prefill is tok/s per post-hygiene Pi turn. All v1.8.4 values
come from a single coop-on boot. Ranges are observed, not confidence intervals.

The v1.8.0 column is the frozen production-stock release cohort (`release_m0`)
in [results-v1.8.0.json](results-v1.8.0.json): one serving start, two sweeps per
decode cell and eight prefill turns. A cell is below or above when the whole
v1.8.4 range lies outside the v1.8.0 range; otherwise the ranges overlap. This
compares observed ranges; it is not a statistical equivalence test.

## Measured boot

Candidate commit `adda4a11fd58ec593cacbbfce4919868d8916e9b`; production-stock
weights (`ABLIT=0`), coop on, adaptive-K `ema`, dense FP8 `trunk`. The serving
start used the release component seal, pinned native and measured policy
recorded in the structured record. Thinking was off and prefix reuse was not
used for decode.

The boot passed same-boot admission. Its
[first-prompt receipt](v1.8.4-admission/first-prompt.json),
[finalization receipt](v1.8.4-admission/finalize.json) and the publishable
dependencies are included byte-exact. The full admission logs, and three JSON
records that contain private network addresses or paths, are private: they are
hash-bound in the finalization receipt and not published. The
[attestation](v1.8.4-admission/attestation.json) records the owner's full
private recheck of that receipt with the unchanged admission gate. Operators
must run full admission on their own boots.

Both full-model quality captures passed the quality panel gate on this serving
start; the record keeps their hashes and same-start agreement summary. No
cross-boot or coop on/off quality conclusion is drawn. Raw streams, request
payloads and logs are private; the record keeps their SHA-256 values.
