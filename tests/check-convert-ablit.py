#!/usr/bin/env python3
"""Check scripts/convert-ablit.sh end to end with a stand-in docker: what it runs, what it checks, what it marks.

    python3 tests/check-convert-ablit.py

Works on copies of this tree made ready for the ablit conversion: pins for the shipped converter and a stand-in
conversion id, a stand-in engine wheel, and stand-in output lists. The stand-in docker answers for the pinned image
and, for `docker run`, writes a prepared conversion output into $DATA/ablit/work the way the real one does.
Passes when:
  - a good conversion moves the weights and all three thirds into place and marks each verified; split.sh then has
    nothing to do, and a second run converts nothing;
  - the container runs with no network, the offline source inventory and no Hugging Face token;
  - a wrong conversion id, a third that differs, or a changed converter is refused and nothing is marked verified.
No data, no docker.
"""
import hashlib
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from cluster_fixture import select

HERE = Path(__file__).resolve().parent.parent
PINS = dict(re.findall(r'^([A-Z_0-9]+)=(\S*)', (HERE / 'pins.env').read_text(), re.M))
CID = hashlib.sha256(b'stand-in conversion').hexdigest()
TOKEN = 'hf_CANARY' + secrets.token_hex(12)
failures = []

DOCKER = '''#!/usr/bin/env bash
case $1 in
  image) echo "$FAKE_IMAGE_ID" ;;
  ps) ;;
  run)
    printf '%s\\n' "$*" >>"$DOCKER_ARGS"
    for a in "$@"; do [[ $a == *:/data ]] && data=${a%:/data}; done
    mkdir -p "$data/ablit/work" && cp -r "$FIXTURE/converted" "$data/ablit/work/" ;;
  *) exit 1 ;;
esac
'''


def check(ok, what):
    print(f'{"PASS" if ok else "FAIL"}  {what}')
    if not ok:
        failures.append(what)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_split(*args):
    subprocess.run([sys.executable, str(HERE / 'scripts/verify-split.py'), *map(str, args)], check=True,
                   capture_output=True)


def fixture(root, cid=CID):
    conv = root / 'converted'
    (conv / 'weights').mkdir(parents=True)
    (conv / 'weights/config.json').write_text('{}\n')
    (conv / 'weights/model-00001-of-00001.safetensors').write_bytes(b'\1' * 4096)
    (conv / 'CONVERSION-ID').write_text(cid + '\n')
    for rank in range(3):
        third = conv / f'rank{rank}'
        third.mkdir()
        (third / f'model-00001-of-00001.rank{rank}of3.safetensors').write_bytes(bytes([rank]) * 2048)
        (third / 'chat_template.jinja').write_text('template\n')
        verify_split('seal', third, '--rank', rank, '--conversion-id', CID)
    return conv


def prepare(t):
    tree, data = t / 'tree', t / 'data'
    for part in ('scripts', 'config', 'manifests'):
        shutil.copytree(HERE / part, tree / part, symlinks=True)
    for part in ('pins.env', 'cluster.env.example'):
        shutil.copy2(HERE / part, tree / part)
    conv = fixture(t / 'good')
    verify_split('write', conv / 'weights', tree / 'manifests/inputs/ablit-weights.sha256')
    for rank in range(3):
        verify_split('write', conv / f'rank{rank}', tree / f'manifests/ablit/rank{rank}.sha256')
    (tree / 'wheels').mkdir()
    wheel = tree / 'wheels' / PINS['ENGINE_WHEEL']
    with zipfile.ZipFile(wheel, 'w') as z:
        z.writestr('tensorfold/__init__.py', '')
    member = hashlib.sha256(b'').hexdigest()
    content = hashlib.sha256(f'{member}  tensorfold/__init__.py\n'.encode()).hexdigest()
    pins = (tree / 'pins.env').read_text()
    for key, value in (('ABLIT_CONVERTER_SHA256', sha(tree / 'scripts/ablit/convert-weights.py')),
                       ('ABLIT_CONVERSION_ID', CID), ('WHEEL_CONTENT_SHA256', content)):
        pins = re.sub(rf'(?m)^{key}=.*$', f'{key}={value}', pins)
    (tree / 'pins.env').write_text(pins)
    (tree / 'cluster.env').write_text(re.sub(r'(?m)^DATA=\S*', f'DATA={data}',
                                             (tree / 'cluster.env.example').read_text()))
    for folder, listed in (('base/weights', 'inputs/base-weights.sha256'), ('ablit/source', 'inputs/ablit-source.sha256')):
        (data / folder).mkdir(parents=True)
        (data / folder / 'stand-in.txt').write_text('x\n')
        Path(f'{data / folder}.verified').write_text(sha(tree / 'manifests' / listed) + '\n')
    (t / 'bin').mkdir()
    (t / 'bin/docker').write_text(DOCKER)
    (t / 'bin/docker').chmod(0o755)
    return tree, data


def run(t, tree, script, *args, fixture_root=None):
    env = dict(os.environ, PATH=f'{t / "bin"}:{os.environ["PATH"]}', HF_TOKEN=TOKEN, DOCKER_ARGS=str(t / 'docker-args'),
               FIXTURE=str(fixture_root or t / 'good'), FAKE_IMAGE_ID=PINS['IMAGE_ID'])
    select(env, tree, t)
    p = subprocess.run([str(tree / 'scripts' / script), *args], capture_output=True, text=True, env=env, timeout=300)
    return p.returncode, p.stdout + p.stderr


def docker_runs(t):
    path = t / 'docker-args'
    return path.read_text().splitlines() if path.exists() else []


def marked(tree, data):
    pairs = [(data / 'ablit/weights', tree / 'manifests/inputs/ablit-weights.sha256')]
    pairs += [(data / f'ablit/rank{r}', tree / f'manifests/ablit/rank{r}.sha256') for r in range(3)]
    return [Path(f'{d}.verified').exists() and Path(f'{d}.verified').read_text().strip() == sha(m) for d, m in pairs]


def good(t):
    tree, data = prepare(t)
    code, out = run(t, tree, 'convert-ablit.sh', '--dry-run')
    check(code == 0 and '--network none' in out and not docker_runs(t), 'the dry run prints the container and runs nothing')
    code, out = run(t, tree, 'convert-ablit.sh')
    check(code == 0 and all(marked(tree, data)),
          'a good conversion marks the weights and all three thirds verified against their lists')
    check(all((data / f'ablit/rank{r}/MANIFEST.sha256').exists() for r in range(3))
          and not any((data / f'ablit/work/converted/rank{r}').exists() for r in range(3)),
          'the thirds are moved into $DATA/ablit/rank<R>, sealed as converted')
    check((data / 'ablit/WEIGHTS-LICENSE.txt').exists() and (data / 'ablit/TEMPLATE-ADDITIONS-LICENSE.txt').exists(),
          'the two license files are kept beside the converted weights')
    args = ' '.join(docker_runs(t))
    check(len(docker_runs(t)) == 1 and '--network none' in args and f'--user {os.getuid()}:{os.getgid()}' in args
          and '--source-api /convert/ablit-source-api.json' in args and '/convert/reproduce-ablit.py /data/ablit/work' in args,
          'the container runs reproduce-ablit.py offline, as the host user, with the saved source inventory')
    logs = ''.join(f.read_text() for f in (tree / 'logs').glob('*.log'))
    check(TOKEN not in args + out + logs and 'HF_TOKEN' not in args, 'no Hugging Face token reaches the container, console or logs')
    code, out = run(t, tree, 'split.sh', '--weights', 'ablit', '--rank', '1')
    check(code == 0 and 'already split and verified' in out, 'split.sh --weights ablit then has nothing to do')
    code, out = run(t, tree, 'convert-ablit.sh')
    check(code == 0 and 'already built and verified' in out and len(docker_runs(t)) == 1,
          'a second run converts nothing')


def refused(t, what, expect, fixture_change=None, tree_change=None):
    tree, data = prepare(t)
    root = t / 'changed'
    shutil.copytree(t / 'good', root)
    if fixture_change:
        fixture_change(root / 'converted')
    if tree_change:
        tree_change(tree)
    code, out = run(t, tree, 'convert-ablit.sh', fixture_root=root)
    check(code != 0 and expect in out and not any(marked(tree, data)), f'{what} is refused; nothing is marked verified')
    return out


def main():
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        for name in ('good', 'id', 'third', 'converter'):
            (t / name).mkdir()
        good(t / 'good')
        refused(t / 'id', 'a conversion id other than ABLIT_CONVERSION_ID', 'not ABLIT_CONVERSION_ID',
                fixture_change=lambda c: (c / 'CONVERSION-ID').write_text('f' * 64 + '\n'))
        out = refused(t / 'third', 'a third that differs from its list', 'rank 2 third does not match',
                      fixture_change=lambda c: (c / 'rank2/chat_template.jinja').write_text('other\n'))
        check('DIFFERENT  chat_template.jinja' in out, 'the refusal names the file that differs')
        refused(t / 'converter', 'a converter that is not the pinned one', 'ABLIT_CONVERTER_SHA256',
                tree_change=lambda tree: (tree / 'scripts/ablit/convert-weights.py').open('a').write('#\n'))
        check(not docker_runs(t / 'converter'), 'a changed converter is refused before the container starts')
    print(f'check-convert-ablit: {len(failures)} failed')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
