# Engine development and maintenance

For JSpark3 v2.0.1 (GLM-5.3 Flash), follow the enclosing release's
[installation guide](../INSTALL.md). It owns the container, weight pins,
three-rank split, serving configuration, and start procedure.

From the release root, in the prepared environment:

```sh
python -m pip install ./engine
tensorfold --help
```

For editable development, use `python -m pip install -e ./engine`. To install
the optional Apple Silicon SSD streaming dependencies from this source, use
`python -m pip install './engine[ssd]'` from the release root.

See the [API reference](docs/api.md) for engine interfaces. Inherited family
recipes and the changelog are historical TensorFold reference material, not
qualification results for this JSpark3 release. See the release's
[README Results section](../README.md#results),
[docs/BENCHMARKS.md](../docs/BENCHMARKS.md) and
[release/MEASUREMENTS-v2.0.1.md](../release/MEASUREMENTS-v2.0.1.md)
for measured results and supported configuration claims.

Automatic update checks are disabled in this vendored engine. `tensorfold update`
prints the JSpark3 installation link and makes no changes. Updates must use the
complete JSpark3 release so the engine and recipe stay together.
