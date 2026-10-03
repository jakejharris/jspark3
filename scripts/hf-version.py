#!/usr/bin/env -S python3 -I -S
"""Read hf's isolated venv metadata as data, without starting hf or its interpreter.

Run: python3 -I -S scripts/hf-version.py
Both flags are required before Python starts: ignore Python environment/cwd paths
and disable site processing (.pth, sitecustomize and usercustomize). A plain
python3 invocation cannot prevent startup hooks that run before this file.

Supports standard Linux venvs (including pipx), direct/env Python shebangs with
only an optional -s, and pip's quoted shell trampoline. Reads one local dist-info
METADATA file; never imports the Hub or runs installation code. System/user,
system-site-enabled, editable, ambiguous or unreadable installs print unknown
and exit 1. A readable version exits 0; callers compare the whole pinned version.
"""
import sys

import argparse
import json
import os
import re
import shlex
import shutil
from email.parser import Parser
from pathlib import Path


def interpreter(entry):
    def launch_line(stream):
        # Match bytes Linux/shell see: no universal-newline or whitespace folding.
        # Reject controls except horizontal tab; preserve valid UTF-8 path bytes.
        raw = stream.readline().removesuffix(b'\n')
        if any(byte < 32 and byte != 9 or byte == 127 for byte in raw):
            raise ValueError('unsupported launcher bytes')
        return raw.decode('utf-8')

    with open(entry, 'rb') as stream:
        first = launch_line(stream)
        # Linux reads a bounded shebang buffer; never parse a truncated command.
        if len(first.encode('utf-8')) > 255:
            raise ValueError('overlong shebang')
        command = re.split(r'[ \t]+', first[2:].strip(' \t'), maxsplit=1)
        if command == ['/bin/sh']:
            second, third = launch_line(stream), launch_line(stream)
    if not first.startswith('#!'):
        raise ValueError('no shebang')
    if command == ['/bin/sh']:
        # Recognize distlib's trampoline, not general shell syntax or expansions.
        match = re.fullmatch(r"'''exec' (.+) \"\$0\" \"\$@\"", second)
        if not match or third != "' '''":
            raise ValueError('unsupported shell launcher')
        word = match[1]
        # Shell syntax is ASCII; non-ASCII path characters stay literal.
        if not (re.fullmatch(r'[/\w.+\-\u0080-\U0010ffff]+', word) or
                re.fullmatch(r'"[^"$`\\\n]+"', word)):
            raise ValueError('unsupported shell interpreter')
        command = shlex.split(word)
    elif command and command[0] in ('/usr/bin/env', '/bin/env'):
        if '$' in command[1] or '\\' in command[1]:
            raise ValueError('unsupported env expansion')
        if re.split(r'[ \t]+', command[1], maxsplit=1)[:1] == ['-S']:
            command = shlex.split(command[1])[1:]
        else:
            # Without -S, env receives one literal kernel argument, including quotes.
            command = [command[1]]
    elif command and not Path(command[0]).is_absolute():
        raise ValueError('relative shebang interpreter')
    # A direct shebang has at most one kernel argument. Do not shlex-split it.
    if not command or command[1:] not in ([], ['-s']):
        raise ValueError('unsupported interpreter options')
    python = command[0]
    if not re.fullmatch(r'python(?:[0-9]+(?:\.[0-9]+)*)?', Path(python).name):
        raise ValueError('not a Python entry point')
    python = shutil.which(python)
    if python is None:
        raise ValueError('missing interpreter')
    # Never resolve this symlink: the lexical path selects the venv, even with -S
    # on Python versions where site would otherwise establish the venv prefix.
    return Path(os.path.abspath(python))


def optional_exists(path):
    # Path.exists() hides ELOOP on supported Pythons. Only absence is optional;
    # unreadable paths (including loops in any ancestor) must refuse the layout.
    try:
        path.stat()
    except FileNotFoundError:
        return False
    return True


def version(entry):
    python = interpreter(entry)
    if python.parent.name != 'bin':
        raise ValueError('unsupported installation layout')
    home = python.parent.parent
    cfg = {}
    for line in (home / 'pyvenv.cfg').read_text().splitlines():
        key, sep, value = line.partition('=')
        if sep:
            key = key.strip().lower()
            if key in cfg:
                raise ValueError('ambiguous venv configuration')
            cfg[key] = value.strip()
    if cfg.get('include-system-site-packages', '').lower() != 'false':
        raise ValueError('not an isolated venv')
    match = re.fullmatch(r'(3\.[0-9]+)\.[0-9]+', cfg.get('version', ''))
    if not match:
        raise ValueError('unknown venv Python version')
    packages = home / 'lib' / ('python' + match[1]) / 'site-packages'
    alternate = home / 'lib64' / ('python' + match[1]) / 'site-packages'
    if optional_exists(alternate) and alternate.resolve() != packages.resolve():
        raise ValueError('ambiguous package locations')
    candidates = []
    for path in packages.iterdir():
        if re.fullmatch(r'huggingface[-_.]+hub(?:-.+)?\.(?:dist-info|egg-info|egg-link)', path.name, re.I):
            candidates.append(path)
    if len(candidates) != 1 or not candidates[0].name.endswith('.dist-info'):
        raise ValueError('missing or ambiguous distribution metadata')
    dist = candidates[0]
    direct_url = dist / 'direct_url.json'
    if optional_exists(direct_url) and json.loads(direct_url.read_text()).get('dir_info', {}).get('editable'):
        raise ValueError('editable installation')
    metadata = Parser().parsestr((dist / 'METADATA').read_text(), headersonly=True)
    if (len(metadata.get_all('Name', [])) != 1 or
            re.sub(r'[-_.]+', '-', metadata['Name']).lower() != 'huggingface-hub' or
            len(metadata.get_all('Version', [])) != 1 or metadata.defects):
        raise ValueError('invalid distribution metadata')
    value = metadata['Version']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.!+_-]*', value):
        raise ValueError('invalid version')
    return value


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    try:
        if not (sys.flags.isolated and sys.flags.no_site):
            raise ValueError('launch with -I -S')
        entry = shutil.which('hf')
        if entry is not None:
            print(version(entry))
            return 0
    except (OSError, RuntimeError, ValueError, IndexError, TypeError, AttributeError):
        pass
    print('unknown')
    return 1


if __name__ == '__main__':
    sys.exit(main())
