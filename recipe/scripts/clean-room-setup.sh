#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
recipe_dir=$(cd -- "$script_dir/.." && pwd)
[[ -f "$recipe_dir/SHA256SUMS" ]] || { echo 'REFUSE: SHA256SUMS is missing' >&2; exit 9; }
(cd -- "$recipe_dir" && sha256sum -c SHA256SUMS)
exec "$script_dir/preflight.sh" "$@"
