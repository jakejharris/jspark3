# SPDX-License-Identifier: Apache-2.0
"""Share structure, never arbitrary remote text. Raw diagnostics stay private."""
from __future__ import annotations

import builtins
from functools import lru_cache
import os
from pathlib import Path
import re
import tempfile

TAIL_LIMIT = 16 * 1024  # Characters; bound before parsing, including long lines.
PRIVATE_SUFFIX = '.may-contain-secrets-do-not-share.log'
EXCEPTIONS = frozenset(name for name, cls in vars(builtins).items()
                       if isinstance(cls, type) and issubclass(cls, BaseException)) | {'Refusal'}


@lru_cache(maxsize=1)
def source_files() -> frozenset[str]:
    # Names come from controller source, never the remote traceback's path.
    return frozenset(p.name for p in Path(__file__).resolve().parent.rglob('*.py')) | {'<string>', '<stdin>'}


def command_context(argv: list[str]) -> tuple[str, frozenset[str]]:
    """Only the locally constructed executable/script; no args, paths or code."""
    files = source_files()
    command = 'unrecognized command'
    if argv and argv[0] in ('python3', 'python', 'docker', 'mkdir'):
        command = argv[0]
    if command == 'docker' and len(argv) > 1 and argv[1] in ('exec', 'logs', 'container', 'start', 'stop', 'rm', 'create'):
        command += ' ' + argv[1]
    if command in ('python3', 'python'):
        for arg in argv[1:]:
            if arg == '-c':
                command += ' -c'
                break
            if arg in ('-B', '-S'):
                continue
            name = Path(arg).name
            # This script name is supplied by controller code, not stderr.
            if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,100}\.py', name):
                files = files | {name}
                command += ' ' + name
            break
    return command, files


def structure(text: str, files: frozenset[str] | None = None) -> dict:
    files = source_files() if files is None else files
    frames, exceptions = [], []
    # No JSON/URL decoding, credential regex, or free-form message copying.
    for raw in text[-TAIL_LIMIT:].splitlines():
        if len(raw) > 1024:
            continue
        line = raw.strip()
        if line.startswith('File "'):
            path, sep, rest = line[6:].partition('", line ')
            number = rest.partition(',')[0]
            if sep and re.fullmatch(r'[0-9]{1,9}', number):
                name = path.rsplit('/', 1)[-1]
                frames.append({'file': name if name in files else '<unrecognized file>', 'line': int(number)})
        else:
            name, _, number = line.partition(':')
            if (name in files or name == '<unrecognized file>') and re.fullmatch(r'[0-9]{1,9}', number):
                frames.append({'file': name, 'line': int(number)})
        name = line.partition(':')[0]
        if name in EXCEPTIONS and name not in exceptions:
            exceptions.append(name)
    return {'frames': frames[-32:], 'exception_types': exceptions,
            'stderr_truncated': len(text) > TAIL_LIMIT}


def render(detail: dict) -> str:
    lines = [f"{frame['file']}:{frame['line']}" for frame in detail['frames']]
    lines += detail['exception_types']
    return '\n'.join(lines) or 'No recognized traceback; inspect the private diagnostic file locally.'


def private_tail(output: Path, text: str) -> str:
    """Unique, exclusive 0600 sidecar next to output; never follow an old link."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='diagnostic-', suffix=PRIVATE_SUFFIX, dir=output.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(text[-TAIL_LIMIT:])
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        os.unlink(name)
        raise
    return Path(name).name


def failure(exc: Exception) -> dict:
    if hasattr(exc, 'safe_diagnostics'):
        detail = dict(exc.safe_diagnostics)
    else:
        detail = structure(str(exc))
        name = type(exc).__name__
        if name in EXCEPTIONS and name not in detail['exception_types']:
            detail['exception_types'].append(name)
    frames = []
    tb = exc.__traceback__
    while tb is not None:
        name = Path(tb.tb_frame.f_code.co_filename).name
        frames.append({'file': name if name in source_files() else '<unrecognized file>', 'line': tb.tb_lineno})
        tb = tb.tb_next
    detail['controller_frames'] = frames[-32:]
    return detail
