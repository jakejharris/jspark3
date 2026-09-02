# Local-only image reproduction

JSpark3 v1.0.0 was measured on, and always launches, the upstream MiaAI-Lab
image at this exact digest:

```text
ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58
```

The lifecycle controller, per-rank preflight, transform contract, and
host-minted image receipt all bind that manifest digest and its config digest.
No JSpark3 GHCR image is published for v1.0.0, and no alternate image is a
release deliverable.

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

This loads `jspark3-local:1.0.0` into the local Docker image store. The script
has no push mode, registry login, remote tag, or export path. The manual image
workflow performs the same non-pushing build check in CI.

## Redistribution boundary

Do not push, export, publish, or otherwise redistribute the local build without
independently satisfying NVIDIA's terms and every applicable upstream term.
Changing labels, adding notices, or assigning a new tag does not itself supply
those rights. If a future, independently cleared image is created, it is a new
release decision and requires fresh license review and three-node verification;
it is not part of v1.0.0.
