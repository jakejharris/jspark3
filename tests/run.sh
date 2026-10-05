#!/usr/bin/env bash
# Every check that runs without the hardware: shell and Python syntax (and shellcheck when installed), each script's
# --help, the chat template, the per-box ring environment, the rendered serve command against the measured one,
# output hygiene, and every file against SHA256SUMS. Needs no data and starts no containers.
#
#   tests/run.sh
set -uo pipefail
# shellcheck source=clean-env.sh
source "$(dirname "$0")/clean-env.sh"
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE" || exit 2
failed=0
unrun=0
step() {  # name command...
  local name=$1 out; shift
  if out=$("$@" 2>&1); then
    echo "PASS  $name"
  else
    echo "FAIL  $name"
    printf '%s\n' "$out" | tail -20 | sed 's/^/      /'
    failed=$((failed + 1))
  fi
}

shell=(scripts/*.sh tests/*.sh)
# shellcheck disable=SC2329  # called through step, which shellcheck cannot follow
syntax() { local f; for f in "$@"; do bash -n "$f" || return 1; done; }
step "bash syntax" syntax "${shell[@]}"
if command -v shellcheck >/dev/null; then
  step "shellcheck" shellcheck "${shell[@]}"
else
  echo "SKIP  shellcheck (not installed)"
fi
step "python syntax" python3 -m py_compile scripts/*.py scripts/fabric/*.py tests/*.py tools/*.py
step "release identity" python3 scripts/check-release.py
# --help prints the script's header comment and nothing of its code.
# shellcheck disable=SC2329  # called through step
help_ok() { local out; out=$("$@" --help) && ! grep -qE '^(set -|source |shellcheck |#!)' <<<"$out"; }
for s in scripts/*.sh; do
  [[ $s == scripts/lib.sh ]] && continue
  step "$s --help" help_ok "$s"
done
for p in scripts/cache-check.py scripts/prefill-check.py scripts/preflight.py scripts/hf-version.py scripts/verify-split.py scripts/exactness.py scripts/wheel-content.py tools/payload.py tools/release-assets.py; do
  step "$p --help" python3 -I -S "$p" --help
done
step "chat template" python3 tests/check-template.py
step "ring environment" python3 tests/check-ring-env.py
step "serve render matches the measured boot" python3 tests/check-serve-render.py
# These fixtures need loopback sockets. Explicitly report them as unrun when
# the sandbox denies sockets; all other errors remain failures in the real tests.
python3 -I -S - <<'PY'
import socket
import sys
try:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
except PermissionError:
    sys.exit(77)
PY
socket_status=$?
if [[ $socket_status == 77 ]]; then
  for test in check-anonymous-fetch.py check-token-canary.py check-fetch-redirects.py check-fetch-gate-order.py check-api-gates.py; do
    echo "UNRUN tests/$test (loopback sockets denied; rerun outside the sandbox)"
    unrun=$((unrun + 1))
  done
else
  step "cache, prefill and streaming API gates" python3 tests/check-api-gates.py
  step "base weights and drafter fetch anonymously" python3 tests/check-anonymous-fetch.py
  step "no token reaches the console or the logs" python3 tests/check-token-canary.py
  step "authenticated fetch redirects stay within the selected origin" python3 tests/check-fetch-redirects.py
  step "ablit access is checked before every download" python3 tests/check-fetch-gate-order.py
fi
step "every downloader uses the selected endpoint" python3 tests/check-fetch-endpoint.py
step "CLI version checks read local metadata without interpreter startup" python3 tests/check-hf-version.py
step "CLI version checks reject startup, launcher and cwd injections" python3 tests/check-hf-version-isolation.py
step "CLI version checks reject malformed launcher bytes and symlink loops" python3 tests/check-hf-version-malformed.py
step "capacity checks preserve outputs when space is short" python3 tests/check-space-guards.py
step "verification rechecks revoke stale markers" python3 tests/check-verification-markers.py
step "ablit thirds are sealed as measured" python3 tests/check-split-seal.py
step "ablit conversion: runs offline, checks and marks" python3 tests/check-convert-ablit.py
step "preflight probes pass on a stock Spark" python3 tests/check-preflight-probes.py
# shellcheck disable=SC2329  # called through step
keep_parks() { local out; out=$(scripts/stop.sh --dry-run --keep 2>&1); [[ $out == *"docker stop"* && $out != *"docker rm"* ]]; }
step "stop.sh --keep stops without removing" keep_parks
step "output hygiene" bash tests/check-leaks.sh
step "files match SHA256SUMS" python3 tools/payload.py verify
echo "tests/run.sh: $failed failed"
if [[ $unrun != 0 ]]; then echo "tests/run.sh: $unrun unrun (loopback sockets denied)"; fi
exit $((failed > 0))
