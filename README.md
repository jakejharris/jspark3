# JSpark3

A serving recipe for GLM-5.3 Flash on three NVIDIA DGX Sparks.

**Current release: v1.8.4.** Start with the [v1.8.4 release](https://github.com/jakejharris/jspark3/releases/tag/v1.8.4) and the [v1.8.4 installation guide](https://github.com/jakejharris/jspark3/blob/v1.8.4/docs/INSTALL.md).

Do not install v1.8.0: it requires a container image that was never published.

Download the [v1.8.4 recipe tarball](https://github.com/jakejharris/jspark3/releases/download/v1.8.4/jspark3-recipe-v1.8.4.tar.gz), extract it, and run the checksum and validator commands below from the extracted `jspark3` directory. Or clone the [v1.8.4 source tag](https://github.com/jakejharris/jspark3/tree/v1.8.4):

```sh
git clone --branch v1.8.4 https://github.com/jakejharris/jspark3.git
cd jspark3
sha256sum -c SHA256SUMS
python3 -B tools/validate_release.py . --require-final
```

Then follow the installation guide to build and verify your own local image and native binaries. There is no JSpark3 image to pull from GHCR. Keep the prepared default `JSPARK3_V16_COOP=1` and complete the guide's three-host qualification before serving traffic.

v1.8.4 serves at parity with v1.8.0, with cooperative MoE on by default; see the [v1.8.4 measurements](https://github.com/jakejharris/jspark3/blob/v1.8.4/release/MEASUREMENTS-v1.8.4.md). The coop on/off comparison is pending; no uplift is claimed.

Cooperative MoE is on by default. Operators compile the byte-identical native library that passed the component GPU campaign and was sealed for this release. The pin is `3212a3b0a308e2ec3673878212fcb0504a463c5f7df84eced5db7bb301cc3c07` and the component seal is `42ea2a36ab5b9de05e7e8b1d690bd1fe7014968ec7381ac4a62516143deb05ec`. See the [campaign evidence](https://github.com/jakejharris/jspark3/blob/v1.8.4/release/component-v1.8.4.json).

**A/B status:** coop on/off comparison pending; no uplift claimed

Shared cudart linkage and the component license boundaries are documented in the release licensing guide.

For older deployments, see the [historical v1.1.0 (Cadence) guide](https://github.com/jakejharris/jspark3/blob/v1.1.0/README.md).

For license boundaries and notices, see the [v1.8.4 licensing guide](https://github.com/jakejharris/jspark3/blob/v1.8.4/docs/LICENSING.md).

## Credits

JSpark3 builds on work by Brandon M. Music, who made the
[EXL3/TR3 4-bpw checkpoint](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
that every rank loads, and by Z.AI, Inco AI, z-lab, Mia's AI Lab,
FlyCockpit, vcruz305, sfxnz, Tony, turboderp, coolbho3k, gabewillen,
plotarmordev, outstandly, ratulsarna, nood-co1, knapcio, lilianmoraru,
the vLLM project and the InstantTensor contributors.
See [THIRD_PARTY_NOTICES.md](https://github.com/jakejharris/jspark3/blob/v1.8.4/THIRD_PARTY_NOTICES.md) for component contributions,
source revisions and license notices.

The [ShapleyMcg](https://github.com/brandonmmusic-max/shapleymcg) attribution is reproduced below.

This work includes or was produced using ShapleyMcg, created by Brandon M. Music (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under the ShapleyMcg License v1.0, an attribution-required license that grants no rights to the person known as "0xSero." Use of ShapleyMcg without this attribution is unlicensed.
