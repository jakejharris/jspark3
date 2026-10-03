"""Install a fake hf in a real, pip-free venv with local distribution metadata."""
import sys
import venv


def install_hf(entry, body, version):
    home = entry.parent / 'hf-env'
    venv.EnvBuilder(with_pip=False, symlinks=True).create(home)
    packages = home / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
    metadata = packages / 'huggingface_hub-0.0.0.dist-info'
    metadata.mkdir(exist_ok=True)
    (metadata / 'METADATA').write_text(f'Metadata-Version: 2.1\nName: huggingface-hub\nVersion: {version}\n')
    if 'bash' in body.splitlines()[0]:
        body = 'import os, sys\nos.execv("/bin/bash", ["bash", "-c", ' + repr(body) + ', "hf", *sys.argv[1:]])\n'
    else:
        body = body.partition('\n')[2]
    python = home / 'bin/python'
    if ' ' in str(python) or len(str(python)) > 100:
        header = f'#!/bin/sh\n\'\'\'exec\' "{python}" "$0" "$@"\n\' \'\'\'\n'
    else:
        header = f'#!{python}\n'
    entry.write_text(header + body)
    entry.chmod(0o755)
    return packages
