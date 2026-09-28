# SPDX-License-Identifier: Apache-2.0
"""Share structure, never arbitrary remote text. Raw diagnostics stay private."""
from __future__ import annotations

import builtins
from contextlib import contextmanager
import hashlib
import json
from functools import lru_cache
import os
from pathlib import Path
import re
import sys
import subprocess
import tempfile
import traceback
import urllib.error

TAIL_LIMIT = 16 * 1024  # Characters; bound before parsing, including long lines.
PRIVATE_SUFFIX = '.may-contain-secrets-do-not-share.log'
EXCEPTIONS = frozenset(name for name, cls in vars(builtins).items()
                       if isinstance(cls, type) and issubclass(cls, BaseException)) | {'Refusal', 'QAError', 'ImageRefusal', 'Cancelled'}


def retain(text: str, output: Path | None = None) -> str | None:
    """Retain complete raw protocol/log data privately, never as evidence inputs."""
    try:
        if output is None:
            output = private_directory() / 'diagnostic.json'
        return str(output.parent / private_text(output, text))
    except Exception:
        return None


def private_read(response):
    value = response.read()
    retain(value.decode('utf-8', 'replace') if isinstance(value, bytes) else value)
    return value


def private_lines(response):
    chunks = []
    try:
        for chunk in response:
            chunks.append(chunk)
            yield chunk
    finally:
        retain(b''.join(chunks).decode('utf-8', 'replace'))


@lru_cache(maxsize=1)
def private_directory() -> Path:
    return Path(tempfile.mkdtemp(prefix='jspark3-private-'))


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def numeric_usage(value: dict) -> dict:
    """Project only token counts; vendor extensions and messages stay private."""
    if not isinstance(value, dict):
        raise ValueError('usage must be an object')
    result = {}
    for name in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'cached_tokens', 'reasoning_tokens'):
        if name in value:
            if type(value[name]) is not int or value[name] < 0:
                raise ValueError('invalid token count: ' + repr(value))
            result[name] = value[name]
    for name in ('prompt_tokens_details', 'completion_tokens_details'):
        if value.get(name) is not None:
            result[name] = numeric_usage(value[name])
    return result


@contextmanager
def capture_log(output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='diagnostic-', suffix=PRIVATE_SUFFIX, dir=output.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.private_diagnostic_path = name
            yield handle
    finally:
        # The log is private; only the bounded structural summary is shared.
        with open(name, 'rb') as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - TAIL_LIMIT))
            tail = handle.read().decode('utf-8', 'replace')
        output.write_text(render(structure(tail)) + '\nPrivate diagnostics (do not share): ' + Path(name).name + '\n')


def run_private(argv, **kwargs):
    """Capture inherited child output instead of leaking it to the terminal."""
    with capture_log(private_directory() / 'command.log') as log:
        try:
            result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, **kwargs)
        except BaseException as exc:
            exc.private_child_output = log.private_diagnostic_path
            raise
    return result


@lru_cache(maxsize=1)
def source_files() -> frozenset[str]:
    # Names come from controller source, never the remote traceback's path.
    root = Path(__file__).resolve().parents[2]
    return frozenset(p.name for folder in (root / 'recipe/scripts', root / 'tools')
                     for p in folder.rglob('*.py')) | {'<string>', '<stdin>'}


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
    return private_text(output, text[-TAIL_LIMIT:])


def private_text(output: Path, text: str) -> str:
    """Unique, exclusive 0600 sidecar next to output; never follow an old link."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='diagnostic-', suffix=PRIVATE_SUFFIX, dir=output.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        os.unlink(name)
        raise
    return Path(name).name


def failure(exc: Exception) -> dict:
    if 'safe_diagnostics' in vars(exc):
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


def record_failure(exc: BaseException, output: Path | None = None, *, command: str = 'operation', preserve: bool = True) -> dict:
    """One failure boundary for receipts, CLIs and nested cleanup errors."""
    detail = failure(exc)
    detail.setdefault('command', command)
    detail.setdefault('rank', None)
    detail.setdefault('exit_code', None)
    raw = vars(exc).get('raw_output')
    if raw is None:
        raw = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    if isinstance(exc, (subprocess.CalledProcessError, subprocess.TimeoutExpired)):
        for value in (exc.stdout, exc.stderr):
            if value:
                raw += '\n' + (value.decode('utf-8', 'replace') if isinstance(value, bytes) else value)
    if isinstance(exc, urllib.error.HTTPError):
        try:
            raw += '\n' + exc.read().decode('utf-8', 'replace')
        except Exception:
            pass
    if 'private_child_output' in vars(exc):
        raw += '\nComplete private child output: ' + exc.private_child_output
    if vars(exc).get('raw_output') is None and private_directory.cache_info().currsize:
        raw += '\nOther captured private protocol data: ' + str(private_directory())
    private = full = None
    location = output
    try:
        if not preserve:
            raise OSError('diagnostic files disabled for dry run')
        if location is None:
            location = Path(tempfile.mkdtemp(prefix='jspark3-private-')) / 'failure.json'
        private = private_tail(location, raw)
        full = private_text(location, raw) if len(raw) > TAIL_LIMIT else private
        if output is None:
            private, full = str(location.parent / private), str(location.parent / full)
    except Exception:
        # A failing diagnostic writer must never leak its own exception.
        pass
    prefix = (f"rank{detail['rank']} remote command failed (exit {detail['exit_code']})"
              if detail['rank'] is not None else 'operation refused')
    reason = prefix + ': ' + detail['command'] + '\n' + render(detail)
    reason += '\nController frames:\n' + render({'frames': detail['controller_frames'], 'exception_types': []})
    reason += ('\nPrivate diagnostics (do not share): ' + full if full else '\nPrivate diagnostic write failed.')
    if private and private != full:
        reason += '\nBounded private tail (do not share): ' + private
    return {'reason': reason, 'diagnostics': detail, 'private_stderr_tail': private,
            'private_diagnostic': full,
            'private_protocol_directory': str(private_directory()) if private_directory.cache_info().currsize else None}


def report_failure(exc: BaseException, output: Path | None = None, *, command: str = 'operation') -> dict:
    report = record_failure(exc, output, command=command)
    print('REFUSE: ' + report['reason'], file=sys.stderr)
    return report


def install_exception_hook() -> None:
    """Uncaught CLI failures get the same private/public boundary."""
    sys.excepthook = lambda kind, value, tb: report_failure(value.with_traceback(tb))


if __name__ == '__main__':
    install_exception_hook()
    print('Build output is private; diagnostics directory: ' + str(private_directory()))
    result = run_private(['bash', *sys.argv[1:]], env=dict(os.environ, JSPARK_PRIVATE_BUILD_LOG='1'))
    print('PASS private build command' if result.returncode == 0 else 'REFUSE: private build command failed')
    raise SystemExit(result.returncode)
