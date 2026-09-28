# JSpark3

A serving recipe for GLM-5.3 Flash on three NVIDIA DGX Sparks.

**Current release: v1.8.2.** Start with the [v1.8.2 release](https://github.com/jakejharris/jspark3/releases/tag/v1.8.2) and the [v1.8.2 installation guide](https://github.com/jakejharris/jspark3/blob/v1.8.2/docs/INSTALL.md).

Do not install v1.8.0: it requires a container image that was never published.

Download the [v1.8.2 recipe tarball](https://github.com/jakejharris/jspark3/releases/download/v1.8.2/jspark3-recipe-v1.8.2.tar.gz), extract it, and run the checksum and validator commands below from the extracted `jspark3` directory. Or clone the [v1.8.2 source tag](https://github.com/jakejharris/jspark3/tree/v1.8.2):

Use a shallow clone: a full `git clone` fails v1.8.2's privacy scan on old commit history.

```sh
git clone --depth 1 --branch v1.8.2 https://github.com/jakejharris/jspark3.git
cd jspark3
sha256sum -c SHA256SUMS
python3 -B tools/validate_release.py .
```

Then follow the installation guide to build and verify your own local image and native binaries. There is no JSpark3 image to pull from GHCR. Keep the default `JSPARK3_V16_COOP=0` and complete the guide's three-host qualification before serving traffic.

For older deployments, see the [historical v1.1.0 (Cadence) guide](https://github.com/jakejharris/jspark3/blob/v1.1.0/README.md).

For license boundaries and notices, see the [v1.8.2 licensing guide](https://github.com/jakejharris/jspark3/blob/v1.8.2/docs/LICENSING.md).

## Attribution

This work includes or was produced using ShapleyMcg, created by Brandon M. Music (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under the ShapleyMcg License v1.0, an attribution-required license that grants no rights to the person known as "0xSero." Use of ShapleyMcg without this attribution is unlicensed.
