#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Local-only, GPU-free coop reproducibility experiment; preserves every build."""
import argparse
import difflib
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import build_native as native
from diff_native_elf import Elf, compare

# The deliberately different process counts test independence from PID allocation.
# No GPU is exposed and no kernel is launched; CDLL tests host dependency loading.
COMMAND = '''set -euo pipefail
for ((i=0; i<$1; i++)); do /bin/true; done
bash /w/build_repro.sh /w/out
readelf -W -S -s -n -d /w/out/cooperative_moe.so > /w/out/elf.txt
/usr/local/cuda/bin/cuobjdump --dump-elf /w/out/cooperative_moe.so > /w/out/cuda-elf.txt 2> /w/out/cuda-elf.stderr
ldd /w/out/cooperative_moe.so > /w/out/ldd.txt
python3 -c 'import ctypes; ctypes.CDLL("/w/out/cooperative_moe.so"); print("PASS host library load; no kernel execution")' > /w/out/load.txt
'''


def compare_runs(stages, output, label):
    reports = []
    for index, stage in enumerate(stages[1:], 2):
        first = stages[0] / 'out/cooperative_moe.so'
        other = stage / 'out/cooperative_moe.so'
        result = compare(first, other)
        name = f'{label}-1-vs-{index}'
        (output / (name + '.json')).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
        # Decode CUDA line tables as well as reporting raw .nv_fatbin offsets.
        a, b = (s / 'out/cuda-elf.txt' for s in (stages[0], stage))
        if a.exists() and b.exists():
            diff = difflib.unified_diff(a.read_text().splitlines(True), b.read_text().splitlines(True),
                                       fromfile=str(a), tofile=str(b))
            (output / (name + '.cuda-elf.diff')).write_text(''.join(diff))
        reports.append({'report': name + '.json', 'identical': result['identical']})
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new directory outside the source export')
    parser.add_argument('--runs', type=int, default=3, help='fresh containers per builder; minimum 3')
    parser.add_argument('--baseline-builder', type=Path,
                        help='optional saved older build_repro.sh, run separately as a diagnostic control')
    args = parser.parse_args()
    if not 3 <= args.runs <= 10:
        parser.error('--runs must be between 3 and 10')
    try:
        output = args.output.resolve()
        if args.output.is_symlink() or output.exists() or output.is_relative_to(ROOT):
            raise ValueError('output must be new and outside the source export')
        from validate_release import verify
        if verify(ROOT)['failed']:
            raise ValueError('source export failed validation')
        image = native.read_operator_record(args.image_receipt)
        if image['source_recipe_sha256'] != native.sha(ROOT / 'recipe/SHA256SUMS'):
            raise ValueError('build an image receipt for this source revision first')
        native.verify_local_image(image)
        builders = {'candidate': ROOT / native.COOP / 'build_repro.sh'}
        if args.baseline_builder:
            if args.baseline_builder.is_symlink() or not args.baseline_builder.is_file():
                raise ValueError('baseline builder must be a regular file')
            builders = {'baseline': args.baseline_builder, **builders}
        inputs = native.build_inputs()
        report = {'schema_version': 1, 'scope': 'compilation and host load only; no GPU qualification',
                  'hardware_qualified': False, 'host_machine': platform.machine(),
                  'docker_machine': subprocess.check_output(
                      ['docker', 'info', '--format', '{{.Architecture}}'], text=True).strip(),
                  'image_receipt_sha256': native.sha(args.image_receipt),
                  'image_config': image['config_digest'], 'build_inputs': inputs, 'builders': {}}
        output.mkdir(parents=True)
        for label, builder in builders.items():
            group = {'builder_sha256': native.sha(builder), 'runs': []}
            report['builders'][label] = group
            stages = []
            for run in range(args.runs):
                stage = output / f'{label}-{run + 1}'
                stage.mkdir()
                stages.append(stage)
                shutil.copy2(builder, stage / 'build_repro.sh')
                shutil.copy2(ROOT / native.COOP / 'SOURCE_MANIFEST.json', stage / 'SOURCE_MANIFEST.json')
                shutil.copytree(ROOT / native.COOP / 'source', stage / 'source')
                burns = run * 37
                command = ['docker', 'run', '--rm', '--platform', 'linux/arm64', '--network', 'none',
                           '--cpus', '4', '--memory', '8g', '--memory-swap', '8g', '--pids-limit', '512',
                           '--user', f'{os.getuid()}:{os.getgid()}', '-v', f'{stage}:/w', '-w', '/w',
                           '--entrypoint', 'bash', image['config_digest'], '-c', COMMAND, 'bash', str(burns)]
                print(f'BUILD {label} {run + 1}/{args.runs}; retained at {stage}', flush=True)
                with (stage / 'console.log').open('w') as log:
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                row = {'directory': stage.name, 'exit_code': process.returncode, 'pid_burns': burns}
                group['runs'].append(row)
                binary = stage / 'out/cooperative_moe.so'
                if binary.is_file():
                    row['sha256'] = native.sha(binary)
                    try:
                        symbols = Elf(binary).symbols()
                        row['elf_valid'] = True
                        row['tmpxft_symbols'] = [s[1] for s in symbols if 'tmpxft_' in s[1]]
                        row['defined_cudart_symbols'] = [s[1] for s in symbols
                            if s[4] != 0 and s[1].split('@')[0] in {'cudaLaunchKernel', 'cudaRuntimeGetVersion',
                                                                  '__cudaRegisterFatBinary', '__cudaRegisterFunction'}]
                    except (ValueError, IndexError, struct.error) as exc:
                        row['elf_error'] = str(exc)
                if process.returncode == 0:
                    row['shared_cudart'] = 'Shared library: [libcudart.so.13]' in (stage / 'out/elf.txt').read_text()
                    row['dependencies_found'] = 'not found' not in (stage / 'out/ldd.txt').read_text()
                print(json.dumps(row, sort_keys=True), flush=True)
                (output / 'experiment.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
            if all(r.get('elf_valid') for r in group['runs']):
                group['comparisons'] = compare_runs(stages, output, label)
            else:
                group['comparisons'] = []
        candidate = report['builders']['candidate']
        passed = (native.build_inputs() == inputs
                  and len(candidate['comparisons']) == args.runs - 1
                  and all(c['identical'] for c in candidate['comparisons'])
                  and all(r['exit_code'] == 0 and r.get('shared_cudart') and r.get('dependencies_found')
                          and not r.get('defined_cudart_symbols') and not r.get('tmpxft_symbols')
                          for r in candidate['runs']))
        report['status'] = 'PASS' if passed else 'FAIL'
        (output / 'experiment.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        print(f'{report["status"]} candidate compilation experiment; evidence={output}; GPU qualification still required')
        return 0 if passed else 9
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'REFUSE: {exc}', file=sys.stderr)
        return 9


if __name__ == '__main__':
    raise SystemExit(main())
