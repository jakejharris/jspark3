# JSpark3 v2.0.1 (GLM-5.3 Flash) engine

This is the engine vendored by [JSpark3](https://github.com/jakejharris/jspark3/tree/release/v2.0.1).
It is a modified TensorFold 0.3.6.2 source snapshot, with JSpark3's distributed
GLM serving changes. The Python distribution retains the name `tensorfold`
and the inherited package version; the enclosing recipe identifies the release.

Follow the release's [installation guide](../INSTALL.md) for the container,
model preparation, three-rank configuration, and launch instructions. See the
enclosing release's `RELEASE-FACTS.md` for measured results and qualification.
Other inherited engine documentation describes TensorFold features and historical
measurements; it does not qualify those configurations for this release.

To build or install this source inside the recipe's prepared environment, from
the release root:

```sh
python -m pip install ./engine
```

For development, use `python -m pip install -e ./engine`. Optional Apple Silicon
SSD streaming dependencies can be installed with `python -m pip install './engine[ssd]'`.
See [RUNBOOK.md](RUNBOOK.md) and the [API reference](docs/api.md).

The vendored engine does not check for or install upstream updates. The
`tensorfold update` command points to the enclosing release's installation
guide; update the complete JSpark3 recipe together.

The engine retains its MIT [LICENSE](LICENSE), with the Apache-2.0 port and
other third-party components identified in [NOTICE](NOTICE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Upstream TensorFold is credited
at <https://github.com/ashhart/TensorFold>. Model weights and drafters are fetched
separately under their own licenses; none are included in this source tree.
