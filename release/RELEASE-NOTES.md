# v1.8.4 candidate: pending component qualification

This offline candidate adds the owner qualification/sealing workflow, a separate
qualification-image/operator-image schema, default display+coop builds and
qualified coop-on preparation/admission. It does not contain a measured policy
for the new binary and cannot pass final-release validation yet.

Native candidate: `3212a3b0a308e2ec3673878212fcb0504a463c5f7df84eced5db7bb301cc3c07`.
The builder, native sources and compiler flags remain unchanged.

The component campaign, final-source reproduction on two physical machines,
ordinary-clone stock boot admission, archive/clone parity and new measurements
must pass before release. An edited serving boot requires its separate owner
admission. The public producer continues to reject ABLIT=1.

Builder identities in the seal are operator-attested observations from a trusted
operator's two-board determinism check. Host-only collection, a fixed trusted
executable and a clean query environment reduce accidental identity mistakes.
Receipt self-hashes do not authenticate edits, defend against a malicious
operator or provide hardware attestation. Independently rebuilding and comparing
the pinned native SHA-256 remains the reproducible artifact integrity check.
See the [builder evidence field descriptions](../docs/COOP_REPRODUCIBILITY.md#builder-evidence-fields).

See [the owner workflow](../docs/COOP_REPRODUCIBILITY.md),
[installation](../docs/INSTALL.md) and [measurement scope](MEASUREMENTS-v1.8.4.md).
Historical results and notices remain preserved; compiled binaries are excluded.

## Credits

JSpark3 builds on work by Z.AI, Inco AI, z-lab, Mia's AI Lab,
FlyCockpit, vcruz305, sfxnz, Tony, turboderp, coolbho3k, gabewillen,
plotarmordev, outstandly, the vLLM project and the InstantTensor contributors.
See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for component contributions,
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
