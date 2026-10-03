#!/usr/bin/env python3
"""Run the real capacity/removal blocks at source scale, with stubbed size/free-space commands."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SIZE = 200100885293
failures = []


def check(ok, what):
    print(f'{"PASS" if ok else "FAIL"}  {what}')
    if not ok:
        failures.append(what)


def guard(script):
    source = (HERE / f'scripts/{script}.sh').read_text()
    start = source.index('  need=$(du') if script == 'convert-ablit' else source.index('    need=$(find')
    stop = source.index('\nfi\nwarn_if_serving', start) if script == 'convert-ablit' else source.index('\n  fi\n  warn_if_serving', start)
    return source[start:stop]


SETUP = '''set -euo pipefail
DATA=$1; out=$DATA/weights; work=$DATA/work; snapshot=$DATA/snapshot; what=fixture
variant_dir() { echo "$DATA"; }
rank_dir() { echo "$DATA/rank$1"; }
du() { printf '%s\\tfixture\\n' "$SOURCE_BYTES"; }
find() { printf '%s\\n' "$SOURCE_BYTES"; }
df() { printf 'Avail\\n%s\\n' "$FREE_BYTES"; }
die() { echo "$*"; exit 2; }
'''


def run_case(t, script, size, free, awk):
    root = t / 'data'
    root.mkdir()
    for folder in ('weights', 'weights.partial', 'work', 'rank0', 'rank1', 'rank2',
                   'rank0.partial', 'rank1.partial', 'rank2.partial'):
        directory = root / folder
        directory.mkdir()
        (directory / 'prior-output').write_bytes(b'keep this payload\n')
        Path(str(directory) + '.verified').write_bytes(b'keep this marker\n')
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    (t / 'bin').mkdir()
    (t / 'bin/awk').symlink_to(awk)
    env = dict(os.environ, PATH=f'{t / "bin"}:{os.environ["PATH"]}', SOURCE_BYTES=str(size), FREE_BYTES=str(free))
    # Use the production formatter as well as the guard, with every installed awk.
    formatter = next(line for line in (HERE / 'scripts/lib.sh').read_text().splitlines() if line.startswith('bc_div()'))
    p = subprocess.run(['bash', '-c', SETUP + formatter + '\n' + guard(script) + '\nprintf "need=%s\\n" "$need"',
                        'space-fixture', str(root)], env=env, capture_output=True, text=True, timeout=10)
    after = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    return p, before == after


def main():
    awks = sorted(Path('/usr/bin').glob('*awk'))
    check(bool(awks), 'enumerated /usr/bin/*awk')
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        for awk in awks:
            print(f'awk: {awk} -> {awk.resolve()}')
            for script, need in (('convert-ablit', SIZE * 2), ('split', (SIZE * 11 + 29) // 30)):
                for free in (10000000000, need - 1, need):
                    case = t / f'{awk.name}-{script}-{free}'
                    case.mkdir()
                    p, unchanged = run_case(case, script, SIZE, free, awk)
                    if free < need:
                        check(p.returncode == 2 and 'GB free' in p.stdout and unchanged,
                              f'{awk.name} {script}: {free} < {need} refuses and preserves every output/marker')
                    else:
                        check(p.returncode == 0 and f'need={need}\n' in p.stdout and not unchanged,
                              f'{awk.name} {script}: exactly {need} bytes permits replacement')
                check('awk' not in guard(script), f'{script}: capacity arithmetic is independent of awk')
        for size in (1, 30, 31):
            case = t / f'rounding-{size}'
            case.mkdir()
            need = (size * 11 + 29) // 30
            p, _ = run_case(case, 'split', size, need, awks[0])
            check(p.returncode == 0 and f'need={need}\n' in p.stdout, f'split rounds {size} bytes upward to {need}')
    print(f'check-space-guards: {len(failures)} failed')
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
