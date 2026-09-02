# syntax=docker/dockerfile:1
#
# JSpark3 v1 local-only image reproduction definition.
#
# v1.0.0 was measured on, and its lifecycle controller pins, the upstream
# MiaAI-Lab image by digest. This Dockerfile produces a thin, labeled
# derivative of that exact digest for local inspection and reproducibility. It
# adds provenance labels and the recipe's license set only. It does not bake
# weights, tokenizer files, or the runtime transforms: those are applied
# fail-closed at container start by the recipe, bound to a host-minted receipt.
#
# Do not push, export, or otherwise redistribute this derivative without
# independently satisfying NVIDIA and every applicable upstream term. No
# JSpark3 GHCR image is published for v1.0.0; the controller uses FROM's exact
# upstream digest directly.

FROM ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58

ARG JSPARK3_VERSION=1.0.0
ARG JSPARK3_SOURCE_REVISION=unknown

LABEL org.opencontainers.image.title="jspark3-local-reproduction" \
      org.opencontainers.image.version="${JSPARK3_VERSION}" \
      org.opencontainers.image.source="https://github.com/jakejharris/jspark3" \
      org.opencontainers.image.revision="${JSPARK3_SOURCE_REVISION}" \
      org.opencontainers.image.base.name="ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks" \
      org.opencontainers.image.base.digest="sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58" \
      org.opencontainers.image.description="Local-only JSpark3 reproduction of the pinned upstream GLM-5.3 Flash serving image plus JSpark3 notices. Not a v1.0.0 distribution artifact. Third-party terms apply; see /opt/jspark3/THIRD_PARTY_NOTICES.md."

COPY LICENSE NOTICE THIRD_PARTY_NOTICES.md REQUIRED_ATTRIBUTION.md /opt/jspark3/
