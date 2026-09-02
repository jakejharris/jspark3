#!/usr/bin/env bash
# Build the prepared derivative into the local Docker image store for
# reproducibility inspection. This command has no push or registry path.
set -euo pipefail

if (($#)); then
  echo "usage: ./docker/build.sh (local build only; no arguments)" >&2
  exit 2
fi

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
image=${JSPARK3_LOCAL_IMAGE:-jspark3-local}
version=${JSPARK3_VERSION:-1.0.0}
revision=$(git -C "$here" rev-parse HEAD 2>/dev/null || echo unknown)

docker buildx build --platform linux/arm64 --file "$here/docker/Dockerfile" \
  --build-arg "JSPARK3_VERSION=$version" \
  --build-arg "JSPARK3_SOURCE_REVISION=$revision" \
  --tag "$image:$version" --load "$here"

echo "built $image:$version in the local Docker image store"
echo "do not redistribute it without independently satisfying NVIDIA and upstream terms"
