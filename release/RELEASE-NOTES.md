# JSpark3 v1.8.4: parity with v1.8.0, cooperative MoE on by default

v1.8.4 serves at parity with v1.8.0 and turns cooperative MoE on by default.
Of 12 measured cells from one admitted coop-on boot, 8 overlap the frozen v1.8.0
ranges, 2 are above and 2 are below. The coop on/off comparison is pending;
no uplift is claimed.

Cooperative MoE is on by default in the prepared runtime. Operators compile
the native library locally and must reproduce the same bytes that passed the
GPU component campaign and were sealed for this release. The source package
contains the build recipe, measured dispatch policy and qualification metadata;
it does not contain the compiled library, model weights or a container image.

The pinned cooperative-MoE SHA-256 is `3212a3b0a308e2ec3673878212fcb0504a463c5f7df84eced5db7bb301cc3c07`. The component seal is
`42ea2a36ab5b9de05e7e8b1d690bd1fe7014968ec7381ac4a62516143deb05ec`, and its measured dispatch policy is `3623acda4ea998cb53dbeb24c7ac815240a393a31d2e56997df2ce5291bc316b`. These values
come from the [release pin](https://github.com/jakejharris/jspark3/blob/v1.8.4/recipe/config/coop-release.json)
and [component BUILD record](https://github.com/jakejharris/jspark3/blob/v1.8.4/recipe/overlays/v16/coop/BUILD.json).

The completed component campaign ran memcheck and racecheck with candidate
launch-coverage checks, numerical comparisons, graph replay and policy checks.
A deliberate out-of-bounds CUDA access verified that the memcheck detector
still rejects a real invalid access. See the [reviewed campaign evidence](https://github.com/jakejharris/jspark3/blob/v1.8.4/release/component-v1.8.4.json)
and [qualification index](https://github.com/jakejharris/jspark3/blob/v1.8.4/recipe/overlays/v16/coop/QUALIFICATION.json).

**A/B status:** coop on/off comparison pending; no uplift claimed

## Measured throughput

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

The v1.8.0 column is the frozen production-stock release cohort in
[results-v1.8.0.json](https://github.com/jakejharris/jspark3/blob/v1.8.4/release/results-v1.8.0.json). The
[v1.8.4 record](https://github.com/jakejharris/jspark3/blob/v1.8.4/release/results-v1.8.4.json) keeps every figure and its
evidence hash; [v1.8.4 measurements](https://github.com/jakejharris/jspark3/blob/v1.8.4/release/MEASUREMENTS-v1.8.4.md)
describe the comparison. Historical measurements remain in their original files
and are labelled by their original release.

## Admission

The release boot passed same-boot admission. Its first-prompt and finalization
receipts and their publishable dependencies are in the source. The full admission
logs, and three JSON records that contain private network addresses or paths,
are private: they are hash-bound in the
[finalization receipt](https://github.com/jakejharris/jspark3/blob/v1.8.4/release/v1.8.4-admission/finalize.json), and the
[attestation](https://github.com/jakejharris/jspark3/blob/v1.8.4/release/v1.8.4-admission/attestation.json) records the owner's
full private recheck. `validate_release.py --require-final` checks the published
receipts and dependencies against those hashes. Component qualification and byte
reproducibility do not replace admission of the operator's own boot. Follow the
[installation guide](https://github.com/jakejharris/jspark3/blob/v1.8.4/docs/INSTALL.md)
and [operations guide](https://github.com/jakejharris/jspark3/blob/v1.8.4/docs/OPERATIONS.md)
before serving traffic. `prepare_runtime.py --coop-off` and
`build_native.py --display-only` are explicit diagnostic opt-outs from the
qualified coop-on default. Coop changes require a restart.

## Known issue

The first requests after a fresh install can be about 1 s slower once, while GPU kernels (DeepGEMM) compile.

## Licensing

The cooperative kernel builds with shared CUDA runtime linkage (`--cudart=shared`),
as recorded in the [builder](https://github.com/jakejharris/jspark3/blob/v1.8.4/recipe/overlays/v16/coop/build_repro.sh).
Its AGPL-3.0-only derivatives and MIT headers retain their notices and source.
Original recipe code and prose are Apache-2.0. Shared cudart linkage does not
change the component licenses or make the assembled service wholly Apache-2.0.
The target weights and DFlash2 dependency retain their separate terms, including
DFlash2's non-commercial research/evaluation restriction. See the
[license review](https://github.com/jakejharris/jspark3/blob/v1.8.4/manifests/license-review.json)
and [licensing guide](https://github.com/jakejharris/jspark3/blob/v1.8.4/docs/LICENSING.md).

## Credits

JSpark3 builds on work by Z.AI, Inco AI, z-lab, Mia's AI Lab,
FlyCockpit, vcruz305, sfxnz, Tony, turboderp, coolbho3k, gabewillen,
plotarmordev, outstandly, the vLLM project and the InstantTensor contributors.
See [THIRD_PARTY_NOTICES.md](https://github.com/jakejharris/jspark3/blob/v1.8.4/THIRD_PARTY_NOTICES.md) for component contributions,
source revisions and license notices.

The [ShapleyMcg](https://github.com/brandonmmusic-max/shapleymcg) attribution is reproduced below.

This work includes or was produced using ShapleyMcg, created by Brandon M. Music (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under the ShapleyMcg License v1.0, an attribution-required license that grants no rights to the person known as "0xSero." Use of ShapleyMcg without this attribution is unlicensed.

Schedule B citation:

```bibtex
@misc{music2026shapleymcg,
  author = {Music, Brandon M.},
  title  = {ShapleyMCG: An Auditable Calibration-to-Encoding Pipeline for
            Low-Bit Mixture-of-Experts Models},
  year   = {2026},
  url    = {https://github.com/brandonmmusic-max/shapleymcg},
  note   = {Licensed under the ShapleyMcg License v1.0}
}
```
