#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Owner-only clean/OOB detector controls before reserving a full campaign."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import json
import os
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'recipe/scripts'), str(ROOT / 'tools')]
import _diagnostics as diagnostics
from _coop_qualification import need, read, regular, sha
import build_native as native
from experiment_coop_build import run_container, Cancelled
from coop_evidence import KERNEL_FILTER, sanitizer
import coop_sanitizer


def inside():
    work = Path('/work')
    private = work / 'private'
    private.mkdir(mode=0o700)
    os.environ['TMPDIR'] = str(private)
    tempfile.tempdir = str(private)
    with diagnostics.capture_log(private / 'driver.log') as log:
        subprocess.run(['nvidia-smi', '--query-gpu=driver_version,uuid', '--format=csv,noheader'],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    driver = Path(log.private_diagnostic_path).read_text().strip()
    need(driver, 'missing driver identity')
    results = {}
    # Reproduce the counting artifact with two real, safe kernel launches.
    with diagnostics.capture_log(private / 'legacy-clean.log') as log:
        legacy = subprocess.run([coop_sanitizer.EXECUTABLE, '--tool', 'memcheck', '--error-exitcode', '9',
            '--print-limit', '0', '--dump-kernel-launches', '--kernel-name', KERNEL_FILTER,
            '/work/control', 'clean'], stdout=log, stderr=subprocess.STDOUT)
    text = Path(log.private_diagnostic_path).read_text()
    need(legacy.returncode == 9 and text.count('========= Launch #') == 2
         and re.findall(r'(?m)^========= ERROR SUMMARY: .*$', text) == ['========= ERROR SUMMARY: 2 errors']
         and 'CONTROL_COMPLETE clean sync=0' in text
         and not re.search(r'Invalid |Program hit |Internal Sanitizer Error|========= Error:', text),
         'legacy launch-record artifact did not reproduce; review private output')
    results['legacy-clean'] = {'exit_code': 9, 'launch_records': 2, 'reported_errors': 2}
    for tool, mode in [('memcheck', 'clean'), ('memcheck', 'oob'), ('racecheck', 'clean')]:
        name = tool + '-' + mode
        before = set(private.iterdir())
        with diagnostics.capture_log(private / (name + '.log')) as log:
            result = subprocess.run(['python3', '-B', str(ROOT / 'tools/v16/coop_sanitizer.py'),
                                     tool, '/work/control', mode], stdout=log, stderr=subprocess.STDOUT)
        application = Path(log.private_diagnostic_path).read_text()
        strict_logs = [p / 'strict.log' for p in set(private.iterdir()) - before if p.is_dir()]
        need(len(strict_logs) == 1, 'control missing live sanitizer log')
        strict = strict_logs[0].read_text()
        need(len(re.findall(r'(?m)^CONTROL_COMPLETE ' + mode + r' sync=\d+$', application)) == 1,
             'control application did not finish')
        if mode == 'clean':
            need(result.returncode == 0, 'clean control failed')
            launches = sanitizer(application, tool)
            need(launches == 1, 'control candidate launch count differs')
            results[name] = {'exit_code': 0, 'instrumented_launches': launches}
        else:
            counts = re.findall(r'(?m)^========= ERROR SUMMARY: ([0-9]+) errors?$', strict)
            need(result.returncode == 9 and len(counts) == 1 and int(counts[0]) > 0,
                 'OOB failed to trigger the sanitizer exit gate')
            need(re.search(r'Invalid __global__ write of size 4(?: bytes)?', strict)
                 and 'exl3_moe_coop_detector_control' in strict and 'out of bounds' in strict,
                 'negative failed without the intended OOB diagnostic')
            results[name] = {'exit_code': 9, 'errors': int(counts[0]), 'invalid_global_write': True}
    report = {'status': 'PASS', 'driver': diagnostics.fingerprint(driver), 'controls': results,
              'source_sha256': sha(ROOT / 'tools/v16/sanitizer_control.cu'),
              'runner_sha256': sha(ROOT / 'tools/v16/coop_sanitizer.py'), 'binary_sha256': sha(work / 'control')}
    (work / 'control.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')


def host(args):
    output = args.output
    need(output.is_absolute() and not output.exists() and not output.is_symlink()
         and not output.resolve().is_relative_to(ROOT), 'new output outside source required')
    image = native.read_operator_record(args.image_receipt)
    native.verify_local_image(image)
    pin = read(ROOT / 'recipe/config/coop-sanitizer.json')
    for name, digest in pin['files'].items():
        need(sha(regular(args.sanitizer_root, name)) == digest, 'sanitizer package drift')
    need({p.relative_to(args.sanitizer_root).as_posix() for p in args.sanitizer_root.rglob('*')
          if p.is_file() and 'docs' not in p.relative_to(args.sanitizer_root).parts} == set(pin['files']),
         'sanitizer inventory differs')
    output.mkdir(parents=True, mode=0o700)
    common = ['docker', 'create', '--platform', 'linux/arm64', '--network', 'none',
              '--cpus', '2', '--memory', '2g', '--memory-swap', '2g', '--pids-limit', '256']
    for source, target, mode in [(ROOT, '/src', 'ro'), (output, '/work', 'rw'), (args.sanitizer_root, '/sanitizer', 'ro')]:
        need(source.is_absolute() and ':' not in str(source) and ',' not in str(source), 'unsafe mount')
        common += ['-v', f'{source}:{target}:{mode}']
    compile_command = common + ['-e', 'NVIDIA_VISIBLE_DEVICES=void', '-e', 'CUDA_VISIBLE_DEVICES=',
        '--entrypoint', '/usr/local/cuda/bin/nvcc', image['config_digest'], '-std=c++17', '-lineinfo',
        '-gencode', 'arch=compute_121a,code=sm_121a', '/src/tools/v16/sanitizer_control.cu', '-o', '/work/control']
    gpu_command = common + ['--gpus', 'device=0', '--entrypoint', 'python3', image['config_digest'],
                           '-B', '/src/tools/v16/check_sanitizer.py', '--inside']
    for name, command in [('compile', compile_command), ('controls', gpu_command)]:
        stage = output / name
        stage.mkdir()
        with diagnostics.capture_log(output / (name + '.log')) as log:
            result = run_container(command, stage, log)
        need(result.returncode == 0, 'sanitizer control stage failed; private evidence retained')
    report = read(output / 'control.json')
    need(report['status'] == 'PASS', 'control did not pass')
    report.update(image_config=image['config_digest'], sanitizer_package_sha256=pin['package_sha256'])
    (output / 'RESULT.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print('PASS sanitizer clean/OOB controls; campaign qualification still required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inside', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--image-receipt', type=Path)
    parser.add_argument('--sanitizer-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        if args.inside:
            inside()
        else:
            need(all((args.image_receipt, args.sanitizer_root, args.output)), 'all three paths required')
            host(args)
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError, Cancelled) as exc:
        diagnostics.report_failure(exc)
        return 9


if __name__ == '__main__':
    raise SystemExit(main())
