#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Private container worker: strict sanitizer execution plus saved launch coverage."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import os
import subprocess
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'recipe/scripts'))
import _diagnostics as diagnostics
from _coop_qualification import need
from coop_evidence import KERNEL_FILTER, sanitizer_summary, sanitizer_launches

EXECUTABLE = '/sanitizer/compute-sanitizer'


def command(tool, records, log, application):
    need(tool in ('memcheck', 'racecheck') and application, 'invalid sanitizer operation')
    # Info-level launch records are counted as errors by the pinned frontend.
    # Save them for coverage, while the live error gate keeps warn/error/fatal.
    return [EXECUTABLE, '--tool', tool, '--error-exitcode', '9', '--print-limit', '0',
            '--dump-kernel-launches', '--print-level', 'warn', '--kernel-name', KERNEL_FILTER,
            '--save', str(records), '--log-file', str(log), *application]


def run(tool, application):
    # coop_environment routes TMPDIR to retained private protocol storage. The
    # standalone detector control does the same. Never reuse a saved run.
    directory = Path(tempfile.mkdtemp(prefix='sanitizer-'))
    records, strict = directory / 'records', directory / 'strict.log'
    old_umask = os.umask(0o077)
    try:
        result = subprocess.run(command(tool, records, strict, application))
        # Preserve the actual failing status; never turn error-exitcode 9 into 0.
        if result.returncode:
            return result.returncode if result.returncode > 0 else 9
        sanitizer_summary(strict.read_text(), tool)
        need(records.is_file() and records.stat().st_size > 0, 'missing saved sanitizer records')
        # This is a CPU readback of this exact execution, not a second GPU run.
        # The default readback error-exitcode is 0; the live gate above is strict.
        with diagnostics.capture_log(directory / 'readback.log') as log:
            replay = subprocess.run([EXECUTABLE, '--tool', tool, '--read', str(records), '--print-level', 'info',
                                     '--print-limit', '0'], stdout=log, stderr=subprocess.STDOUT)
        need(replay.returncode == 0, 'sanitizer record readback failed')
        count = sanitizer_launches(Path(log.private_diagnostic_path).read_text())
        # Inherited stdout is the campaign's private capture. These closed facts
        # are also safe for the shared gate projection; no raw vendor text leaves.
        for index in range(1, count + 1):
            print(f'========= Launch #{index}\n=========   Kernel: exl3_moe_coop_instrumented')
        print('========= ' + ('ERROR SUMMARY: 0 errors' if tool == 'memcheck' else
                            'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'))
        return 0
    finally:
        os.umask(old_umask)


def main():
    try:
        need(len(sys.argv) >= 3, 'tool and application required')
        return run(sys.argv[1], sys.argv[2:])
    except (OSError, ValueError) as exc:
        diagnostics.report_failure(exc)
        return 9


if __name__ == '__main__':
    raise SystemExit(main())
