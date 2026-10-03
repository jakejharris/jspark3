#!/usr/bin/env python3
"""Check the ablit third's .complete marker and MANIFEST.sha256, which scripts/split.sh writes after the split.

    python3 tests/check-split-seal.py

Passes when:
  - the shipped ablit lists are consistent with that format: each lists .complete and a MANIFEST.sha256 whose
    sha256 is that of the list of every other file, in the same "<sha256>  <name>" form, sorted bytewise;
  - the shipped base lists have an empty .complete and no MANIFEST.sha256;
  - verify-split.py seal writes "<conversion id> rank=R world=3" and then that list, on a small stand-in third;
  - seal refuses a bad conversion id, a second seal and a third with a folder in it;
  - split.sh --dry-run seals an ablit third with ABLIT_CONVERSION_ID and only touches an empty marker for base.
No data, no docker.
"""
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SEAL = [sys.executable, str(HERE / 'scripts/verify-split.py'), 'seal']
EMPTY = hashlib.sha256(b'').hexdigest()
failures = []


def check(ok, what):
    print(f'{"PASS" if ok else "FAIL"}  {what}')
    if not ok:
        failures.append(what)


def entries(path):
    return dict(line.split('  ', 1)[::-1] for line in Path(path).read_text().splitlines() if line)


def listing(names):
    return ''.join(f'{names[n]}  {n}\n' for n in sorted(names, key=str.encode))


def shipped():
    for rank in range(3):
        ablit = entries(HERE / f'manifests/ablit/rank{rank}.sha256')
        inner = listing({n: d for n, d in ablit.items() if n != 'MANIFEST.sha256'})
        check('.complete' in ablit and ablit.get('MANIFEST.sha256') == hashlib.sha256(inner.encode()).hexdigest(),
              f'manifests/ablit/rank{rank}.sha256: MANIFEST.sha256 is the list of the other {len(ablit) - 1} files')
        base = entries(HERE / f'manifests/base/rank{rank}.sha256')
        check(base.get('.complete') == EMPTY and 'MANIFEST.sha256' not in base,
              f'manifests/base/rank{rank}.sha256: empty .complete, no MANIFEST.sha256')


def seal(t):
    third = t / 'rank1'
    third.mkdir()
    for name, body in (('config.json', b'{}'), ('model-00001-of-00002.rank1of3.safetensors', b'\0' * 4096),
                       ('chat_template.jinja', b'template\n'), ('Upper.json', b'[]')):
        (third / name).write_bytes(body)
    cid = hashlib.sha256(b'stand-in conversion').hexdigest()
    p = subprocess.run(SEAL + [str(third), '--rank', '1', '--conversion-id', cid], capture_output=True, text=True)
    marker = third / '.complete'
    check(p.returncode == 0 and marker.read_text() == f'{cid} rank=1 world=3\n',
          'seal writes .complete as "<conversion id> rank=1 world=3"')
    files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in third.iterdir() if f.name != 'MANIFEST.sha256'}
    check((third / 'MANIFEST.sha256').read_text() == listing(files) and '.complete' in files,
          'seal writes MANIFEST.sha256 last: every other file, .complete included, sorted bytewise')
    again = subprocess.run(SEAL + [str(third), '--rank', '1', '--conversion-id', cid], capture_output=True, text=True)
    check(again.returncode != 0 and 'already has MANIFEST.sha256' in again.stderr, 'seal refuses a sealed third')
    bad = subprocess.run(SEAL + [str(t), '--rank', '1', '--conversion-id', 'PENDING'], capture_output=True, text=True)
    check(bad.returncode != 0 and '64 lowercase hex' in bad.stderr, 'seal refuses a conversion id that is not pinned')
    nested = subprocess.run(SEAL + [str(t), '--rank', '1', '--conversion-id', cid], capture_output=True, text=True)
    check(nested.returncode != 0 and 'plain files only' in nested.stderr, 'seal refuses a third with a folder in it')


def dry_runs(t):
    tree = t / 'tree'
    for part in ('scripts', 'config', 'manifests'):
        shutil.copytree(HERE / part, tree / part, symlinks=True)
    for part in ('pins.env', 'cluster.env.example'):
        shutil.copy2(HERE / part, tree / part)
    cid = hashlib.sha256(b'stand-in conversion').hexdigest()
    pins = (tree / 'pins.env').read_text()
    pins = re.sub(r'(?m)^ABLIT_CONVERTER_SHA256=.*$', 'ABLIT_CONVERTER_SHA256=' + 'a' * 64, pins)
    pins = re.sub(r'(?m)^ABLIT_CONVERSION_ID=.*$', 'ABLIT_CONVERSION_ID=' + cid, pins)
    (tree / 'pins.env').write_text(pins)
    for name in ('ablit-source', 'ablit-weights'):
        (tree / f'manifests/inputs/{name}.sha256').write_text(f'{EMPTY}  stand-in.txt\n')
    data = t / 'data'
    data.mkdir()
    (tree / 'cluster.env').write_text(re.sub(r'(?m)^DATA=\S*', f'DATA={data}',
                                             (tree / 'cluster.env.example').read_text()))
    out = {}
    for weights in ('ablit', 'base'):
        p = subprocess.run([str(tree / 'scripts/split.sh'), '--weights', weights, '--rank', '1', '--dry-run'],
                           capture_output=True, text=True, timeout=120)
        out[weights] = (p.returncode, (p.stdout + p.stderr).replace('\\ ', ' '))  # the dry run prints %q quoting
    code, text = out['ablit']
    check(code == 0 and 'verify-split.py seal' in text and f'--conversion-id {cid}' in text
          and '.complete' not in text.split('verify-split.py seal')[0],
          'split.sh --weights ablit --dry-run seals the third with ABLIT_CONVERSION_ID and touches no empty marker')
    code, text = out['base']
    check(code == 0 and 'touch /data/rank1.partial/.complete' in text and 'seal' not in text,
          'split.sh --weights base --dry-run touches an empty .complete and does not seal')


def main():
    shipped()
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        seal(t)
        dry_runs(t)
    print(f'check-split-seal: {len(failures)} failed')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
