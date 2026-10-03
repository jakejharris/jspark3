#!/usr/bin/env python3
"""Attack the real version gate with active startup hooks, launchers and foreign metadata."""
import json
import os
import re
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

from hf_fixture import install_hf

HERE = Path(__file__).resolve().parent.parent
FIXTURE = runpy.run_path(str(HERE / 'tests/check-token-canary.py'))
PIN = FIXTURE['PINS']['HF_HUB_VERSION']
failures = []
HOOK = '''import json, os, socket, sys
with open(os.environ['HOOK_AUDIT'], 'a') as stream:
    stream.write(json.dumps({'event': 'hook', 'file': __file__}) + '\\n')
def audit(event, args):
    if event.startswith('socket.'):
        with open(os.environ['HOOK_AUDIT'], 'a') as stream:
            stream.write(json.dumps({'event': event, 'python': sys.executable}) + '\\n')
        raise RuntimeError('stopped before DNS or socket operation')
sys.addaudithook(audit)
try:
    socket.getaddrinfo('startup.example.invalid', 443)
except RuntimeError:
    pass
'''


def check(ok, what):
    print(f'{"PASS" if ok else "FAIL"}  {what}')
    if not ok:
        failures.append(what)


def main():
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
        packages = install_hf(hf, '#!/usr/bin/env python3\nraise RuntimeError("never execute hf")\n', PIN)
        hf_python = bindir / 'hf-env/bin/python'
        original = hf.read_text()
        metadata = packages / 'huggingface_hub-0.0.0.dist-info/METADATA'
        original_metadata = metadata.read_text()
        outer = t / 'outer'
        venv.EnvBuilder(with_pip=False, symlinks=True).create(outer)
        outer_python = outer / 'bin/python3'
        py_version = f'python{sys.version_info.major}.{sys.version_info.minor}'
        outer_packages = outer / 'lib' / py_version / 'site-packages'
        cwd = t / 'cwd'
        cwd.mkdir()
        audit = t / 'hook-audit'
        env = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'HF_', 'HUGGINGFACE_'))}
        env.update(PATH=f'{outer / "bin"}:{bindir}:/usr/bin:/bin', CLUSTER_ENV=str(cluster),
                   HOOK_AUDIT=str(audit), HF_TOKEN='fake-not-a-credential', HUGGINGFACE_CO_STAGING='1')
        helper = tree / 'scripts/hf-version.py'
        preflight = ('import runpy; p=runpy.run_path(' + repr(str(tree / 'scripts/preflight.py')) + '); '
                     + 'p["check_fetch"]({"HF_HUB_VERSION": ' + repr(PIN) + ', "WEIGHTS": "base"})')
        commands = {
            'helper': [str(outer_python), '-I', '-S', str(helper)],
            'fetch': [str(tree / 'scripts/fetch-weights.sh'), '--weights', 'base', '--drafter-only', '--drafter', 'none'],
            'preflight': [str(outer_python), '-I', '-S', '-c', preflight],
        }
        install = (f'python3 -m venv ~/hf-cli && ~/hf-cli/bin/pip install "huggingface_hub=={PIN}", '
                   'then export PATH="$HOME/hf-cli/bin:$PATH" (INSTALL.md, \'What you need\')')

        def run(argv):
            return subprocess.run(argv, env=env, cwd=cwd, capture_output=True, text=True, timeout=40)

        def gates(label, version=PIN):
            for name, argv in commands.items():
                p = run(argv)
                if name == 'helper':
                    ok = p.returncode == (1 if version == 'unknown' else 0) and p.stdout.strip() == version
                elif name == 'fetch':
                    expected = ('fetch-weights.sh: --drafter-only with --drafter none fetches nothing' if version == PIN else
                                f'fetch-weights.sh: the Hugging Face CLI is version {version}; this release pins '
                                f'HF_HUB_VERSION={PIN}. Install it in a virtual environment: {install}')
                    ok = p.returncode == 2 and p.stderr.strip() == expected
                else:
                    expected = (f"PASS  the Hugging Face CLI ('hf') is the pinned version {PIN}" if version == PIN else
                                f"FAIL  the Hugging Face CLI ('hf') is the pinned version {PIN}: it is {version}: "
                                + 'install it in a virtual environment: ' + install.replace("(INSTALL.md, 'What you need')", '(INSTALL.md)'))
                    ok = p.returncode == 0 and p.stdout.strip() == expected
                check(ok, f'{label}: {name} reports {version} with unchanged gate text')
                if not ok:
                    print(repr((p.returncode, p.stdout, p.stderr)))

        # Both possible interpreters have hooks that actively try to use the network.
        for label, python, location in (('hf', hf_python, packages), ('outer', outer_python, outer_packages)):
            (location / 'active_hook.py').write_text(HOOK)
            (location / 'active_hook.pth').write_text('import active_hook\n')
            gates(f'{label} .pth')
            check(not audit.exists(), f'{label} .pth: zero hook executions and zero DNS/socket attempts')
            run([str(python), '-c', 'pass'])
            events = [json.loads(line)['event'] for line in audit.read_text().splitlines()]
            check('hook' in events and 'socket.getaddrinfo' in events, f'{label} .pth: active negative control')
            audit.unlink()
            (location / 'active_hook.pth').unlink()

        injection = t / 'injection'
        injection.mkdir()
        userbase = t / 'userbase'
        user_packages = userbase / 'lib' / py_version / 'site-packages'
        user_packages.mkdir(parents=True)
        for label, location, variable in (('PYTHONPATH sitecustomize', injection, 'PYTHONPATH'),
                                         ('PYTHONPATH usercustomize', injection, 'PYTHONPATH'),
                                         ('user-site usercustomize', user_packages, 'PYTHONUSERBASE'),
                                         ('cwd sitecustomize', cwd, 'PYTHONPATH')):
            module = 'usercustomize' if 'usercustomize' in label else 'sitecustomize'
            hook = location / (module + '.py')
            hook.write_text(HOOK)
            env[variable] = str(userbase if variable == 'PYTHONUSERBASE' else location)
            gates(label)
            check(not audit.exists(), f'{label}: zero hook executions and zero DNS/socket attempts')
            # Use the base interpreter for usercustomize: isolated venvs disable user-site hooks.
            run([sys.executable, '-c', 'pass'])
            events = [json.loads(line)['event'] for line in audit.read_text().splitlines()]
            check('hook' in events and 'socket.getaddrinfo' in events, f'{label}: active negative control')
            audit.unlink()
            hook.unlink()
            env.pop(variable)

        env['PYTHONHOME'] = str(t / 'nonexistent-home')
        env['PYTHONPATH'] = str(cwd)
        (cwd / 'email.py').write_text('raise RuntimeError("foreign stdlib module")\n')
        gates('PYTHONHOME and foreign stdlib paths')
        env.pop('PYTHONHOME')
        env.pop('PYTHONPATH')
        marker = t / 'launcher-ran'
        payload = f"open({str(marker)!r},'w').write('ran'); print({PIN!r})"
        (cwd / 'launcher.py').write_text(payload)
        for label, header in (
                ('direct -c', f'#!{hf_python} -c{payload}'),
                ('direct -m', f'#!{hf_python} -mlauncher'),
                ('env -S -c', f'#!/usr/bin/env -S {hf_python} -c {shlex.quote(payload)}')):
            hf.write_text(header + '\n')
            gates(label, 'unknown')
            check(not marker.exists(), f'{label}: zero payload executions')
        for header in (f'#!{hf_python} -S', f'#!{hf_python} -W ignore',
                       f'#!/usr/bin/env -S {hf_python} launcher.py',
                       f'#!/usr/bin/env -S PYTHONPATH={cwd} {hf_python}',
                       f'#!/usr/bin/env "{hf_python}"', '#!python3',
                       '#!/usr/bin/env', '#!/usr/bin/env -S'):
            hf.write_text(header + '\n')
            gates('unsupported launcher arguments', 'unknown')
        hf.write_text(original)

        spoof = cwd / f'huggingface_hub-{PIN}.dist-info'
        spoof.mkdir()
        (spoof / 'METADATA').write_text(f'Name: huggingface-hub\nVersion: {PIN}\n')
        env['PYTHONPATH'] = str(cwd)
        metadata.write_text(original_metadata.replace(PIN, '9.8.7'))
        gates('conflicting cwd and PYTHONPATH dist-info', '9.8.7')
        metadata.write_text(original_metadata)
        env.pop('PYTHONPATH')

        # Explicit supported/refused layouts, with no guess across unrelated package roots.
        cfg = bindir / 'hf-env/pyvenv.cfg'
        cfg_text = cfg.read_text()
        cfg.write_text(cfg_text.replace('include-system-site-packages = false', 'include-system-site-packages = true'))
        gates('system-site-enabled venv', 'unknown')
        cfg.write_text(cfg_text)
        cfg.rename(cfg.with_suffix('.saved'))
        gates('system or user installation without pyvenv.cfg', 'unknown')
        cfg.with_suffix('.saved').rename(cfg)
        duplicate = packages / f'huggingface-hub-{PIN}.dist-info'
        duplicate.mkdir()
        (duplicate / 'METADATA').write_text(original_metadata)
        gates('two stale dist-info directories', 'unknown')
        shutil.rmtree(duplicate)
        metadata.parent.rename(duplicate)
        gates('normalized hyphen distribution directory')
        duplicate.rename(metadata.parent)
        metadata.write_text(original_metadata.replace('huggingface-hub', 'huggingface_hub'))
        gates('normalized underscore metadata name')
        metadata.write_text(original_metadata)
        direct_url = metadata.parent / 'direct_url.json'
        direct_url.write_text('{"dir_info": {"editable": true}}')
        gates('editable installation', 'unknown')
        direct_url.unlink()
        metadata.rename(metadata.with_suffix('.saved'))
        gates('missing metadata', 'unknown')
        metadata.with_suffix('.saved').rename(metadata)
        linked = bindir / 'pipx-hf'
        hf.rename(linked)
        hf.symlink_to(linked.name)
        gates('pipx-style symlink to a venv entry point')

        # Enforce the data-only implementation: even a new child-process path fails this test.
        code = ('import runpy, sys; '
                'sys.addaudithook(lambda event, args: (_ for _ in ()).throw(RuntimeError("child forbidden")) '
                'if event in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn") else None); '
                f'sys.argv=[{str(helper)!r}]; runpy.run_path({str(helper)!r}, run_name="__main__")')
        p = run([str(outer_python), '-I', '-S', '-c', code])
        check(p.returncode == 0 and p.stdout.strip() == PIN, 'metadata read starts zero child processes')
        for flags in ([], ['-I'], ['-S']):
            p = run([str(outer_python), *flags, str(helper)])
            check(p.returncode == 1 and p.stdout.strip() == 'unknown', f'{flags}: both startup flags are required')
    print(f'check-hf-version-isolation: {len(failures)} failed')
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
