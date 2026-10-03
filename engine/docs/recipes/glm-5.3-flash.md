# GLM-5.3 Flash in JSpark3

Follow the enclosing release's [installation guide](../../../INSTALL.md) for
the selected weights, drafter terms, three-rank preparation, and serving commands.
The engine's supported flags are listed by `tensorfold serve --help`.

See the enclosing release's [README Results section](../../../README.md#results),
[docs/BENCHMARKS.md](../../../docs/BENCHMARKS.md) and
[release/MEASUREMENTS-v2.0.1.md](../../../release/MEASUREMENTS-v2.0.1.md)
for performance and qualification claims.
Model weights and drafters are separate downloads with their own terms.
The engine source and its license do not grant rights to those downloads.

For the optional Apple Silicon SSD streaming dependencies, run
`python -m pip install './engine[ssd]'` from the release root. This is an
inherited TensorFold feature, not a JSpark3 hardware qualification.
