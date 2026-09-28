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
