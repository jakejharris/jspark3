#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Three fresh GPU-hidden builds of every native artifact; no qualification."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2] / 'recipe/scripts')]
import argparse
import itertools
import json
import re
import subprocess

import build_native as native
import _diagnostics as diagnostics
from _coop_qualification import need, verify_builder_host, independent_builders
from diff_native_elf import compare, public_comparison
from experiment_coop_build import run_container, Cancelled

ARTIFACTS = {relative: (kind, name) for kind, names in native.OUTPUTS.items() for name, relative in names.items()}
CHURN = (0, 7, 19)


def write(path, record):
    record = {k: v for k, v in record.items() if k != 'payload_sha256'}
    record['payload_sha256'] = diagnostics.fingerprint(record)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')


def checked_report(path):
    record = json.loads(path.read_text())
    need(record.get('payload_sha256') == diagnostics.fingerprint({k: v for k, v in record.items() if k != 'payload_sha256'}),
         'experiment report hash drift')
    need(record.get('schema_version') == 1 and record.get('status') == 'PASS'
         and record.get('hardware_qualified') is False and len(record['runs']) == 3,
         'complete passing experiment required')
    verify_builder_host(record['builder_host'])
    for i, row in enumerate(record['runs']):
        need(row['run'] == i + 1 and row['process_churn'] == CHURN[i]
             and set(row['artifacts']) == set(ARTIFACTS)
             and all(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v) for v in row['artifacts'].values()),
             'experiment artifact inventory malformed')
    need(all(row['artifacts'] == record['runs'][0]['artifacts'] for row in record['runs']), 'inconsistent build hashes')
    need(record['source_recipe_sha256'] == native.sha(native.ROOT / 'recipe/SHA256SUMS')
         and record['build_inputs'] == native.build_inputs()
         and record['experiment_sha256'] == native.sha(Path(__file__)), 'experiment source differs from this checkout')
    pairs = {(name, a, b) for name in ARTIFACTS for a, b in itertools.combinations(range(1, 4), 2)}
    rows = record['comparisons']
    need(len(rows) == len(pairs)
         and {(r['artifact'], r['first_run'], r['second_run']) for r in rows} == pairs
         and all(r['identical'] is True and isinstance(r['report_sha256'], str)
                 and re.fullmatch('[0-9a-f]{64}', r['report_sha256']) for r in rows),
         'complete passing ELF comparisons required')
    return record


def compare_hosts(paths, *, allow_emulated=False):
    left, right = map(checked_report, paths)
    if not allow_emulated:
        independent_builders(left['builder_host'], right['builder_host'])
    rows = {name: {'first_sha256': left['runs'][0]['artifacts'][name],
                   'second_sha256': right['runs'][0]['artifacts'][name],
                   'identical': left['runs'][0]['artifacts'][name] == right['runs'][0]['artifacts'][name]}
            for name in ARTIFACTS}
    return {'status': 'PASS' if all(r['identical'] for r in rows.values()) else 'FAIL',
            'hardware_qualified': False, 'independent_native_builders': not allow_emulated, 'artifacts': rows}


def run(args):
    output = args.output.resolve()
    need(not output.exists() and not args.output.is_symlink() and not output.is_relative_to(native.ROOT),
         'new output outside source required')
    from validate_release import verify
    need(not verify(native.ROOT)['failed'], 'source validation failed')
    image = native.read_operator_record(args.image_receipt)
    need(image['source_recipe_sha256'] == native.sha(native.ROOT / 'recipe/SHA256SUMS'), 'image source differs')
    native.verify_local_image(image)
    endpoint, host, inputs = native.local_docker(), native.builder_host(), native.build_inputs()
    verify_builder_host(host)
    need(host['architecture'] == 'aarch64' or args.allow_emulated, 'native ARM64 required; emulation is diagnostic only')
    output.mkdir(parents=True, mode=0o700)
    report = {'schema_version': 1, 'status': 'INCOMPLETE', 'hardware_qualified': False,
              'source_recipe_sha256': image['source_recipe_sha256'], 'image_receipt_sha256': image['payload_sha256'],
              'experiment_sha256': native.sha(Path(__file__)), 'build_inputs': inputs, 'builder_host': host,
              'runs': [], 'comparisons': []}
    write(output / 'report.json', report)
    try:
        for i, churn in enumerate(CHURN, 1):
            artifacts = {}
            for kind in native.OUTPUTS:
                stage = output / f'{kind}-{i}'
                def execute(command, **kwargs):
                    # Use the real builder command and the proven exact-ID
                    # create/start/cleanup path, including signal handling.
                    need(command[:3] == ['docker', 'run', '--rm'], 'unexpected native container command')
                    position = command.index(image['config_digest'])
                    script = 'for ((i=0;i<$1;i++)); do /bin/true; done\nshift\nexec bash "$@"'
                    create = ['docker', 'create', *command[3:position + 1], '-c', script,
                              'native-experiment', str(churn), *command[position + 1:]]
                    with diagnostics.capture_log(stage / 'console.log') as log:
                        result = run_container(create, stage, log)
                    result.check_returncode()
                    return result
                artifacts.update(native.build(kind, stage, image['config_digest'], execute=execute))
            report['runs'].append({'run': i, 'process_churn': churn, 'artifacts': artifacts})
            write(output / 'report.json', report)
        for name, (kind, relative) in ARTIFACTS.items():
            for a, b in itertools.combinations(range(1, 4), 2):
                left, right = (output / f'{kind}-{i}' / relative for i in (a, b))
                need(native.sha(left) == report['runs'][a - 1]['artifacts'][name]
                     and native.sha(right) == report['runs'][b - 1]['artifacts'][name], 'built artifact changed')
                comparison = compare(left, right)
                path = output / f'{Path(name).name}-{a}-{b}.json'
                path.write_text(json.dumps(public_comparison(comparison, path), indent=2) + '\n')
                report['comparisons'].append({'artifact': name, 'first_run': a, 'second_run': b,
                    'identical': comparison['identical'], 'report_sha256': native.sha(path)})
        need(native.build_inputs() == inputs and native.local_docker() == endpoint and native.builder_host() == host,
             'builder identity or source changed')
        native.verify_local_image(image)
        report['status'] = 'PASS' if all(row['identical'] for row in report['comparisons']) else 'FAIL'
        write(output / 'report.json', report)
        print(json.dumps({'status': report['status'], 'hardware_qualified': False,
                          'runs': report['runs'], 'comparisons': report['comparisons']}, sort_keys=True))
        return 0 if report['status'] == 'PASS' else 9
    except BaseException as exc:
        report['status'] = 'REFUSED'
        report['failure'] = diagnostics.record_failure(exc, output / 'report.json', command='native experiment')
        write(output / 'report.json', report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-receipt', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-emulated', action='store_true', help='local CPU diagnostic only, never native qualification')
    parser.add_argument('--compare', type=Path, nargs=2, metavar=('FIRST_REPORT', 'SECOND_REPORT'))
    args = parser.parse_args()
    try:
        if args.compare:
            need(not args.image_receipt and not args.output, 'choose build or comparison')
            result = compare_hosts(args.compare, allow_emulated=args.allow_emulated)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result['status'] == 'PASS' else 9
        need(args.image_receipt is not None and args.output is not None, 'image receipt and new output required')
        return run(args)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, Cancelled) as exc:
        diagnostics.report_failure(exc)
        return 9


if __name__ == '__main__':
    diagnostics.install_exception_hook()
    raise SystemExit(main())
