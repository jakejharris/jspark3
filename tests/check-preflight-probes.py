#!/usr/bin/env python3
"""Check preflight's two container probes against a stand-in docker that behaves like a stock DGX Spark.

    python3 tests/check-preflight-probes.py

The stand-in has only the runc runtime, reaches the GPU through `--gpus all` (the NVIDIA Container Toolkit's hook),
and its image prints a banner on stdout unless the entrypoint is overridden, as the pinned NVIDIA image does.
Passes when:
  - on that box, "a container sees the GPU" and "containers get unlimited locked memory" both PASS;
  - the locked-memory check still passes when a banner precedes the limit;
  - without the toolkit, or with a limited memlock, the matching check FAILs with what to do;
  - both probes self-remove (--rm) and never pull (--pull never).
No GPU, no docker.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from cluster_fixture import select

HERE = Path(__file__).resolve().parent.parent
GPU = 'a container sees the GPU (docker run --gpus all)'
MEMLOCK = 'containers get unlimited locked memory'
failures = []

DOCKER = '''#!/usr/bin/env bash
printf '%s\\n' "$*" >>"$DOCKER_ARGS"
case $1 in
  info) echo '{"io.containerd.runc.v2":{},"runc":{}}' ;;
  run)
    entry='' gpus=0 limit=unlimited
    for a in "$@"; do
      case $prev in --entrypoint) entry=$a ;; --gpus) gpus=1 ;; --ulimit) [[ $a == memlock=-1 ]] || limit=64 ;; esac
      prev=$a
    done
    [[ -n $FAKE_MEMLOCK ]] && limit=$FAKE_MEMLOCK
    if [[ -z $entry || $BANNER == always ]]; then echo '== NVIDIA PyTorch =='; echo 'NVIDIA Release (build)'; fi
    if [[ $entry == nvidia-smi || ($gpus == 1 && -z $entry) ]]; then
      [[ $FAKE_TOOLKIT == 1 ]] || { echo 'could not select device driver "" with capabilities: [[gpu]]' >&2; exit 125; }
      echo 'GPU 0: NVIDIA GB10 (UUID: GPU-stand-in)'
    else
      echo "$limit"
    fi ;;
  *) exit 1 ;;
esac
'''


def check(ok, what):
    print(f'{"PASS" if ok else "FAIL"}  {what}')
    if not ok:
        failures.append(what)


def prepare(t):
    tree = t / 'tree'
    for part in ('scripts', 'config', 'manifests', 'template'):
        shutil.copytree(HERE / part, tree / part, symlinks=True)
    for part in ('pins.env', 'cluster.env.example', 'wheels.lock'):
        shutil.copy2(HERE / part, tree / part)
    (t / 'data').mkdir()
    (tree / 'cluster.env').write_text(re.sub(r'(?m)^DATA=\S*', f'DATA={t / "data"}',
                                             (tree / 'cluster.env.example').read_text()))
    (t / 'bin').mkdir()
    for name, body in (('docker', DOCKER), ('nvidia-smi', '#!/bin/sh\necho "GPU 0: NVIDIA GB10 (UUID: GPU-host)"\n')):
        (t / 'bin' / name).write_text(body)
        (t / 'bin' / name).chmod(0o755)
    return tree


def run(t, tree, **env):
    args = t / 'docker-args'
    args.unlink(missing_ok=True)
    e = dict(os.environ, PATH=f'{t / "bin"}:{os.environ["PATH"]}', DOCKER_ARGS=str(args), FAKE_TOOLKIT='1',
             BANNER='', FAKE_MEMLOCK='')
    e.update(env)
    select(e, tree, t)
    p = subprocess.run([sys.executable, str(tree / 'scripts/preflight.py'), '--for', 'serve'], capture_output=True,
                       text=True, env=e, timeout=300)
    lines = {}
    for line in p.stdout.splitlines():
        state, _, rest = line.partition('  ')
        lines[rest.split(': ', 1)[0] if rest.startswith((GPU, MEMLOCK)) else rest] = (state.strip(), rest)
    return lines, args.read_text().splitlines() if args.exists() else []


def main():
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        tree = prepare(t)
        lines, calls = run(t, tree)
        check(lines.get(GPU, ('',))[0] == 'PASS', 'stock Spark (runc only, toolkit hook): the GPU check passes')
        check(lines.get(MEMLOCK, ('',))[0] == 'PASS', 'stock Spark: the locked-memory check passes despite the banner')
        probes = [c for c in calls if c.startswith('run ')]
        check(len(probes) == 2 and all('--rm' in c.split() and '--pull never' in c for c in probes),
              'both probes self-remove (--rm) and never pull (--pull never)')
        lines, _ = run(t, tree, BANNER='always')
        check(lines.get(MEMLOCK, ('',))[0] == 'PASS', 'a banner before the limit still passes (the last line is read)')
        lines, _ = run(t, tree, FAKE_TOOLKIT='0')
        gpu = lines.get(GPU, ('', ''))
        check(gpu[0] == 'FAIL' and 'NVIDIA Container Toolkit' in gpu[1], 'without the toolkit the GPU check fails, naming it')
        lines, _ = run(t, tree, FAKE_MEMLOCK='8192')
        check(lines.get(MEMLOCK, ('',))[0] == 'FAIL', 'a limited memlock fails')
    print(f'check-preflight-probes: {len(failures)} failed')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
