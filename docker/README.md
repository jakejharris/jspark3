# Local-only image reproduction

> Download clarification, 2026-09-14: Download the GLM model once, from Mia's copy or JSPARK3's copy of the same version. They contain the same model files. Cadence also needs the separate DFlash2 draft, a smaller model that helps generate answers faster.
> [Download sources and setup](https://github.com/jakejharris/jspark3/blob/main/docs/INSTALL.md).

JSpark3 v1.0.0 and v1.1.0 use the same upstream MiaAI-Lab image. Both
releases launch this exact digest:

```text
ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58
```

The lifecycle controller, per-rank preflight, transform contract, and
host-minted image receipt all bind that manifest digest and its config digest.
No JSpark3 GHCR image is published for either release, and no alternate image
is a release deliverable.

The mounted recipe applies its hash-checked runtime transforms before serving.
The v1.1.0 recipe supplies the Cadence modules and long-context kernel fix;
these changes do not require a new base image. The lifecycle launches the
upstream digest directly, not the optional local derivative below.

## Why the Dockerfile remains

`Dockerfile` reproduces the prepared thin derivative locally. It starts from
the exact upstream image, then adds OCI labels and the JSpark3 license and
notice files under `/opt/jspark3/`. It adds no weights, tokenizers, or runtime
transforms.

The completed license audit found this insufficient for redistribution. The
result still contains the NVIDIA-derived upstream layers, and labels or notice
files do not make it satisfy the NGC derived-container redistribution grant.
Accordingly, the Dockerfile is retained only so an operator can inspect and
reproduce the construction locally.

## Local build

On an arm64 host with Docker Buildx:

```bash
./docker/build.sh
```

This loads `jspark3-local:1.1.0` into the local Docker image store. The script
has no push mode, registry login, remote tag, or export path. The manual image
workflow performs the same non-pushing build check in CI.

## Redistribution boundary

Do not push, export, publish, or otherwise redistribute the local build without
independently satisfying NVIDIA's terms and every applicable upstream term.
Changing labels, adding notices, or assigning a new tag does not itself supply
those rights. If a future, independently cleared image is created, it is a new
release decision and requires fresh license review and three-node verification;
it is not part of v1.0.0 or v1.1.0.
