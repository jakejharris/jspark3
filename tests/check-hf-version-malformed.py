#!/usr/bin/env python3
"""Refuse byte-altered launchers and looping venv layouts without executing them."""
import os
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from hf_fixture import install_hf

HERE = Path(__file__).resolve().parent.parent
FIXTURE = runpy.run_path(str(HERE / 'tests/check-token-canary.py'))
PIN = FIXTURE['PINS']['HF_HUB_VERSION']


def main():
    failures = []
    cases = 0
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        tree = FIXTURE['prepare'](t)
        data = t / 'data'
        data.mkdir()
        cluster = tree / 'cluster.env'
        cluster.write_text(re.sub(r'(?m)^DATA=\S*', f'DATA={data}', (tree / 'cluster.env.example').read_text()))
        bindir = t / 'bin'
        bindir.mkdir()
        hf = bindir / 'hf'
        marker = t / 'executed'
        body = f'open({str(marker)!r}, "w").write("executed")\n'
        packages = install_hf(hf, '#!/usr/bin/env python3\n' + body, PIN)
        home = bindir / 'hf-env'
        python = home / 'bin/python'
        # Pin the caller interpreter too: tests/run.sh's selected Python drives all three.
        (bindir / 'python3').symlink_to(sys.executable)
        env = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'HF_', 'HUGGINGFACE_'))}
        env.update(PATH=f'{bindir}:{home / "bin"}:/usr/bin:/bin', CLUSTER_ENV=str(cluster))
        preflight = (f'import runpy; p=runpy.run_path({str(tree / "scripts/preflight.py")!r}); '
                     f'p["check_fetch"]({{"HF_HUB_VERSION": {PIN!r}, "WEIGHTS": "base"}})')
        commands = {
            'helper': [sys.executable, '-I', '-S', str(tree / 'scripts/hf-version.py')],
            'fetch': [str(tree / 'scripts/fetch-weights.sh'), '--weights', 'base', '--drafter-only', '--drafter', 'none'],
            'preflight': [sys.executable, '-I', '-S', '-c', preflight],
        }
        install = (f'python3 -m venv ~/hf-cli && ~/hf-cli/bin/pip install "huggingface_hub=={PIN}", '
                   'then export PATH="$HOME/hf-cli/bin:$PATH" (INSTALL.md, \'What you need\')')

        def gates(label, version='unknown'):
            nonlocal cases
            cases += 1
            good = True
            expected = {
                'helper': (1, 'unknown\n', ''),
                'fetch': (2, '', f'fetch-weights.sh: the Hugging Face CLI is version unknown; this release pins '
                          f'HF_HUB_VERSION={PIN}. Install it in a virtual environment: {install}\n'),
                'preflight': (0, f"FAIL  the Hugging Face CLI ('hf') is the pinned version {PIN}: it is unknown: "
                              + 'install it in a virtual environment: '
                              + install.replace("(INSTALL.md, 'What you need')", '(INSTALL.md)') + '\n', ''),
            }
            if version == PIN:
                expected = {
                    'helper': (0, PIN + '\n', ''),
                    'fetch': (2, '', 'fetch-weights.sh: --drafter-only with --drafter none fetches nothing\n'),
                    'preflight': (0, f"PASS  the Hugging Face CLI ('hf') is the pinned version {PIN}\n", ''),
                }
            for name, argv in commands.items():
                p = subprocess.run(argv, env=env, cwd=t, capture_output=True, text=True, timeout=30)
                actual = (p.returncode, p.stdout, p.stderr)
                ok = actual == expected[name]
                print(f'{"PASS" if ok else "FAIL"}  {label}: {name}')
                if not ok:
                    print(repr(actual))
                good = good and ok
            if marker.exists():
                good = False
                print(f'FAIL  {label}: launcher executed')
            if not good:
                failures.append(label)

        direct = f'#!{python}\n'.encode()
        # Use an absolute env argument so it selects the fixture venv, not the caller.
        env_header = f'#!/usr/bin/env {python}\n'.encode()
        trampoline = f'#!/bin/sh\n\'\'\'exec\' "{python}" "$0" "$@"\n\' \'\'\'\n'.encode()
        for label, header in (('LF direct', direct), ('LF env', env_header), ('LF trampoline', trampoline)):
            hf.write_bytes(header + body.encode())
            gates(label, PIN)
        # Pip/distlib uses UTF-8 and double quotes for a path containing spaces.
        # An apostrophe inside those quotes and Unicode whitespace stay literal.
        for dirname in ('café', 'café home', "café user's home", 'café\u00a0home',
                        'café\u2003home', 'cafe\u0301', 'café🌍'):
            moved = t / dirname / 'hf-cli'
            moved.parent.mkdir()
            home.rename(moved)
            selected = moved / 'bin/python'
            if ' ' in str(selected):
                header = f"#!/bin/sh\n'''exec' \"{selected}\" \"$0\" \"$@\"\n' '''\n"
            else:
                header = f'#!{selected}\n'
            hf.write_bytes(header.encode('utf-8') + body.encode())
            gates(f'UTF-8 pip launcher {dirname!r}', PIN)
            if ' ' not in str(selected):
                # Pip uses this unquoted form for long interpreter paths.
                header = f"#!/bin/sh\n'''exec' {selected} \"$0\" \"$@\"\n' '''\n"
                hf.write_bytes(header.encode('utf-8') + body.encode())
                gates(f'UTF-8 unquoted trampoline {dirname!r}', PIN)
            for argument in (str(selected), f'-S "{selected}"'):
                hf.write_bytes(f'#!/usr/bin/env {argument}\n'.encode('utf-8') + body.encode())
                gates(f'UTF-8 env {argument!r}', PIN)
            moved.rename(home)

        # Negative controls already refused by the parent. Use real paths so a
        # permissive decoder/filter cannot pass by failing a later file lookup.
        for label, byte in (('invalid UTF-8', b'\xff'), ('DEL', b'\x7f'),
                            ('CR', b'\r'), ('control', b'\x01')):
            moved = t / os.fsdecode(b'bad' + byte) / 'hf-cli'
            moved.parent.mkdir()
            home.rename(moved)
            selected = os.fsencode(moved / 'bin/python')
            for form, header in (
                    ('direct', b'#!' + selected + b'\n'),
                    ('trampoline', trampoline.replace(os.fsencode(python), selected))):
                hf.write_bytes(header + body.encode())
                gates(f'{label} {form} path')
            moved.rename(home)
        hf.write_bytes(direct.replace(b'/hf-env/', b'/bad\x00/') + body.encode())
        gates('NUL path')
        # A real path makes a mistaken character-count limit observable.
        moved = t / ('é' * 110) / 'hf-cli'
        moved.parent.mkdir()
        home.rename(moved)
        header = f'#!{moved / "bin/python"}\n'
        assert len(header.rstrip('\n')) <= 255 < len(header.rstrip('\n').encode('utf-8'))
        hf.write_bytes(header.encode('utf-8') + body.encode())
        gates('UTF-8 shebang byte limit')
        moved.rename(home)
        for label, header in (
                ('CRLF direct', direct.replace(b'\n', b'\r\n')),
                ('CRLF env', env_header.replace(b'\n', b'\r\n')),
                ('CRLF trampoline', trampoline.replace(b'\n', b'\r\n')),
                ('CR trampoline exec line', trampoline.replace(b'"$@"\n', b'"$@"\r\n')),
                ('CR trampoline closing line', trampoline.replace(b"' '''\n", b"' '''\r\n")),
                ('vertical-tab shebang', direct.replace(b'\n', b'\v\n')),
                ('form-feed shebang', direct.replace(b'\n', b'\f\n'))):
            hf.write_bytes(header + body.encode())
            gates(label)
        hf.write_bytes(direct + body.encode())
        direct_url = packages / 'huggingface_hub-0.0.0.dist-info/direct_url.json'
        direct_url.symlink_to('direct_url.json')
        gates('optional direct_url loop')
        direct_url.unlink()
        # A real alternate site-packages plus a looping primary forces Path.resolve.
        alternate = home / 'lib64'
        if alternate.is_symlink():
            alternate.unlink()
        alternate.symlink_to('lib64')
        gates('optional lib64 loop')
        alternate.unlink()
        shutil.copytree(home / 'lib', alternate)
        (home / 'lib').rename(home / 'lib.saved')
        (home / 'lib').symlink_to('lib')
        gates('primary lib loop with real lib64')
    print(f'check-hf-version-malformed: {cases} cases, {len(failures)} failed')
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
