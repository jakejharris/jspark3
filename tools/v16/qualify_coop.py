#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Owner-only component campaign; check-only has no GPU or fleet lifecycle access."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'recipe/scripts')]
import build_native as native
from _coop_qualification import (TARGET_NATIVE, canonical, compiled_inputs, digest_value,
                                gate_names, independent_builders, need, read, regular, sha, verify_record)
from _image_identity import build_policy
from _coop_checkpoint import authenticate as fixture
from _coop_bundle import identity as bundle_identity, verify_bundle
from experiment_coop_build import run_container, Cancelled
from coop_evidence import KERNEL_FILTER, validate_campaign, validate_gate, validate_environment
import coop_projection as projection
import _diagnostics as diagnostics
import coop_h1_control as h1

COOP = ROOT / native.COOP
SRC = '/src/recipe/overlays/v16/coop/source'
SNAPSHOT = '/root/.cache/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw/snapshots/'


def write(path, value):
    path.write_bytes(canonical(value))


def runner_inputs():
    paths = ['tools/v16/' + name for name in ('qualify_coop.py', 'coop_environment.py', 'coop_evidence.py', 'coop_projection.py')]
    paths += ['recipe/scripts/' + name for name in ('_coop_checkpoint.py', '_coop_bundle.py',
                                                   '_coop_qualification.py', 'validate_checkpoint.py', '_diagnostics.py')]
    return {name: sha(regular(ROOT, name)) for name in paths}


def matrix():
    commands = []
    for rank in range(3):
        for geometry in range(3):
            for mode in ('baseline', 'perturb'):
                name = f'h1-r{rank}-g{geometry}-{mode}'
                command = ['python3', '-B', '/src/tools/v16/coop_h1_control.py', '--mode', mode,
                           '--rank', str(rank), '--geometry', str(geometry), '--test-source', SRC + '/test_cuda_integration.py',
                           '--source-manifest', '/src/recipe/overlays/v16/coop/SOURCE_MANIFEST.json',
                           '--bundle', '/campaign/raw-bundle', '--receipt', '/campaign/' + name + '-proof.json']
                if mode == 'perturb':
                    command += ['--baseline-receipt', f'/campaign/h1-r{rank}-g{geometry}-baseline-proof.json']
                commands.append((name, command))
            env = ['env', 'GLM53_COOP_QUALIFICATION=1', f'GLM53_COOP_GEOMETRY={geometry}', f'GLM53_COOP_EP_RANK={rank}']
            commands.append((f'profile-r{rank}-g{geometry}', env + ['python3', '-B', SRC + '/profile_shapes.py']))
            for tool in ('memcheck', 'racecheck'):
                sanitizer = ['/sanitizer/compute-sanitizer', '--tool', tool, '--error-exitcode', '9', '--print-limit', '0',
                             '--dump-kernel-launches', '--kernel-name', KERNEL_FILTER]
                commands.append((f'smoke-r{rank}-g{geometry}-{tool}', env + sanitizer + ['python3', '-B', SRC + '/sanitizer_smoke.py']))
                if geometry == 2:
                    commands.append((f'geometry2-r{rank}-g{geometry}-{tool}', env + sanitizer + ['python3', '-B', SRC + '/test_geometry2_sanitizer.py']))
    commands.append(('select-policy', ['python3', '-B', SRC + '/select_policy.py', '--bundle', '/campaign/selected-bundle',
                    *[f'/campaign/profiles/rank{r}-geo{g}.jsonl' for r in range(3) for g in range(3)]]))
    for rank in range(3):
        commands.append((f'policy-r{rank}', ['env', f'GLM53_COOP_EP_RANK={rank}',
                        'GLM53_COOP_BUNDLE=/campaign/selected-bundle', 'python3', '-B', SRC + '/test_policy_gpu.py']))
    return commands


def inputs(args):
    import apply_coop_moe as coop
    import apply_base_pipeline as pipeline
    coop.verify_sources()
    pipeline.verify_sources(args.fly_root, pipeline.contract(ROOT / 'recipe/config/patch-contract.json'))
    helper = read(ROOT / 'recipe/config/coop-helper.json')
    need({p.name for p in args.helpers_root.iterdir()} == {'test_exl3_overlay.py', 'LICENSE'},
         'helper directory must contain only the two pinned files; no import shadows or bytecode')
    need(sha(regular(args.helpers_root, 'test_exl3_overlay.py')) == helper['sha256'], 'helper hash')
    need(sha(regular(args.helpers_root, 'LICENSE')) == helper['license_sha256'], 'helper license hash')
    sanitizer = read(ROOT / 'recipe/config/coop-sanitizer.json')
    need({p.relative_to(args.sanitizer_root).as_posix() for p in args.sanitizer_root.rglob('*')
          if p.is_file() and 'docs' not in p.relative_to(args.sanitizer_root).parts} == set(sanitizer['files']),
         'sanitizer executable/support inventory differs')
    for name, expected in sanitizer['files'].items():
        need(sha(regular(args.sanitizer_root, name)) == expected, 'sanitizer package drift')
    image = native.read_operator_record(args.image_receipt)
    build = native.read_native_record(regular(args.build_root, 'native-build-receipt.json'), image)
    for name, expected in build['binary_sha256'].items():
        need(sha(regular(args.build_root, name)) == expected, 'local native output drift')
    for run in ('a', 'b'):
        stage = args.build_root / ('coop-' + run)
        need(sha(regular(stage, 'out/cooperative_moe.so')) == TARGET_NATIVE, 'raw build pin')
        need(sha(regular(stage, 'SOURCE_MANIFEST.json')) == sha(COOP / 'SOURCE_MANIFEST.json'), 'raw source manifest')
        need(sha(regular(stage, 'build_repro.sh')) == sha(COOP / 'build_repro.sh'), 'raw builder')
        need(not any(p.name == '__pycache__' or p.suffix in ('.pyc', '.pyo') or p.is_symlink()
                     for p in stage.rglob('*')), 'unsafe raw build tree')
        for name, expected in read(COOP / 'SOURCE_MANIFEST.json')['files'].items():
            need(sha(regular(stage / 'source', name)) == expected, 'raw source drift')
        verify_bundle(stage / 'out', COOP, source_policy=True)
    return image, build, helper, fixture(args.model_root)


def independent_build(args, first):
    need(args.independent_build_root and args.independent_image_receipt,
         'check/campaign/seal require second-machine build and image receipts')
    image = native.read_operator_record(args.independent_image_receipt)
    second = native.read_native_record(regular(args.independent_build_root, 'native-build-receipt.json'), image)
    independent_builders(first['builder_host'], second['builder_host'])
    need(sha(regular(args.independent_build_root, native.COOP + '/bundle/cooperative_moe.so')) == TARGET_NATIVE,
         'second-machine native pin')
    return {'native_receipt': second, 'image_receipt': image}


def plan_estimates(plan):
    # Planning allocations from the owner's 1.5–4 h estimate, not measured GPU
    # runtimes or timeouts. The only measured step is the CPU environment (~17 s).
    ranges = {'environment': [17, 60], 'h1': [80, 180], 'profile': [120, 300],
              'smoke': [90, 240], 'geometry2': [180, 600], 'select': [17, 30], 'policy': [60, 180]}
    steps = {name: {'seconds': ranges[name.split('-')[0]],
                   'basis': 'one CPU rehearsal; upper allowance' if name == 'environment' else 'unmeasured planning allowance'}
             for name in plan}
    return {'schema_version': 1, 'confidence': 'low', 'steps': steps,
            'gpu_window_seconds': [sum(row['seconds'][i] for name, row in steps.items() if name != 'environment') for i in (0, 1)],
            'excludes': ['staging', 'drain', 'seal/review', 'fleet restore', 'serving qualification']}


def container(args, campaign, stage, image, command, gpu):
    memory = '16g' if gpu else '4g'
    options = ['docker', 'create', '--platform', 'linux/arm64', '--network', 'none',
               '--cpus', '4', '--memory', memory, '--memory-swap', memory, '--pids-limit', '1024']
    if gpu:
        options += ['--gpus', 'device=0']
    else:
        options += ['-e', 'NVIDIA_VISIBLE_DEVICES=void', '-e', 'CUDA_VISIBLE_DEVICES=']
    for source, target, mode in [(ROOT, '/src', 'ro'), (campaign / 'recipe', '/recipe', 'ro'),
                                (args.fly_root, '/sources/fly', 'ro'), (args.helpers_root, '/helpers', 'ro'),
                                (args.model_root, SNAPSHOT + args.model_root.name, 'ro'), (args.sanitizer_root, '/sanitizer', 'ro'),
                                (campaign, '/campaign', 'rw'), (campaign / 'raw-bundle', '/campaign/raw-bundle', 'ro'), (stage, '/work', 'rw')]:
        need(':' not in str(source) and ',' not in str(source), 'unsafe mount path')
        options += ['-v', f'{source}:{target}:{mode}']
    if SRC + '/select_policy.py' not in command:  # only selection writes this copy
        options += ['-v', f'{campaign / "selected-bundle"}:/campaign/selected-bundle:ro']
    # Remove image defaults as well as host overrides. Never use serving entrypoint.
    clean = ['env']
    for key in (*h1.SERVING_ENV_KEYS, 'GLM53_COOP_QUALIFICATION', 'GLM53_COOP_GEOMETRY',
                'GLM53_COOP_SANITIZER', 'JSPARK3_V16_COOP_MAINTENANCE'):
        clean += ['-u', key]
    clean += ['PYTHONDONTWRITEBYTECODE=1', 'HF_HUB_OFFLINE=1',
              'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
              'GLM53_COOP_MAINTENANCE_TEST=1',
              'GLM53_COOP_TEST_HELPERS=/helpers', 'GLM53_COOP_BUNDLE=/campaign/raw-bundle']
    return options + ['--entrypoint', '/usr/bin/env', image['config_digest'], *clean[1:],
                      'python3', '-B', '/src/tools/v16/coop_environment.py', *command]


def campaign(args):
    output = Path(str(args.output) + '.check') if args.check_only else args.output
    need(not output.exists() and not output.is_symlink() and not output.resolve().is_relative_to(ROOT), 'output must be new outside source')
    image, build, helper, checkpoint = inputs(args)
    second = independent_build(args, build)
    native.verify_local_image(image)
    output.mkdir(parents=True)
    shutil.copytree(ROOT / 'recipe', output / 'recipe')
    write(output / 'recipe/config/operator-image.json', image)
    for name in native.OUTPUTS['display'].values():
        shutil.copyfile(args.build_root / name, output / name)
    projection.copy_bundle(args.build_root / 'coop-a/out', output / 'raw-bundle')
    projection.copy_bundle(output / 'raw-bundle', output / 'selected-bundle')
    (output / 'profiles').mkdir()
    identity = {'native_sha256': TARGET_NATIVE, 'source_manifest_sha256': sha(COOP / 'SOURCE_MANIFEST.json'),
                'raw_bundle': bundle_identity(output / 'raw-bundle'),
                'raw_bundle_manifest': read(output / 'raw-bundle/manifest.json'),
                'build_artifacts_sha256': {f'coop-{run}/out/{name}': sha(regular(args.build_root, f'coop-{run}/out/{name}'))
                    for run in ('a', 'b') for name in ('cooperative_moe.so', 'manifest.json', 'toolchain.txt', 'build32.log', 'build64.log', 'link.log')},
                'image_receipt_sha256': image['payload_sha256'], 'helper': helper, 'checkpoint': checkpoint,
                'sanitizer': read(ROOT / 'recipe/config/coop-sanitizer.json'),
                'runner_sha256': runner_inputs()}
    commands = [('environment', []), *matrix()]
    plan = {name: container(args, output, output / ('container-' + name), image, command, name != 'environment')
            for name, command in commands}
    write(output / 'plan.json', {name: projection.command(command) for name, command in plan.items()})
    write(output / 'plan-estimates.json', plan_estimates(plan))
    for name in plan:
        print('Planned ' + name, flush=True)
    write(output / 'image.json', image)
    write(output / 'native-build.json', build)
    write(output / 'independent-build.json', second)
    write(output / 'campaign.json', {'schema_version': 1, 'status': 'INCOMPLETE', 'identity': identity})
    for name, _ in commands:
        if args.check_only and name != 'environment':
            continue
        stage = output / ('container-' + name)
        stage.mkdir()
        started = datetime.now(timezone.utc).isoformat()
        with diagnostics.capture_log(output / (name + '.log')) as log:
            result = run_container(plan[name], stage, log)
        write(stage / 'execution.json', {'started_at': started, 'completed_at': datetime.now(timezone.utc).isoformat(),
                                       'exit_code': result.returncode, 'command': projection.command(plan[name])})
        need(result.returncode == 0, name + ' failed; evidence retained')
        validate_environment(regular(stage, 'environment.json'), gpu=name != 'environment')
        if name in gate_names():
            # Full output remains in its exclusive 0600 file. Only closed facts
            # enter the log bound by gate receipts; private files are optional.
            raw_text = Path(log.private_diagnostic_path).read_text()
            (output / (name + '.log')).write_text(projection.gate_log(name, raw_text))
            active_bundle = output / ('selected-bundle' if name.startswith('policy-') else 'raw-bundle')
            write(output / (name + '.json'), {'schema_version': 1, 'name': name, 'kind': gate_names()[name],
                    'status': 'PASS', 'exit_code': 0, 'identity': identity, 'bundle': bundle_identity(active_bundle),
                    'log_sha256': sha(output / (name + '.log'))})
            if name.startswith('profile-'):
                r, g = name[-4], name[-1]
                full = output / (name + '.log')
                shutil.copyfile(full, output / f'profiles/rank{r}-geo{g}.jsonl')
            validate_gate(name, output, identity, output / 'raw-bundle')
    need(inputs(args) == (image, build, helper, checkpoint), 'qualification inputs changed during run')
    need(independent_build(args, build) == second, 'independent build changed during run')
    if args.check_only:
        write(output / 'check.json', {'status': 'PASS', 'scope': 'GPU-free environment and complete command plan', 'identity': identity})
    else:
        write(output / 'campaign.json', {'schema_version': 1, 'status': 'COMPLETE', 'identity': identity})
        write(output / 'QUALIFICATION.json', validate_campaign(output))
    print('PASS component campaign')


def seal(args):
    index = validate_campaign(args.seal)
    image = native.read_operator_record(regular(args.seal, 'image.json'))
    build = native.read_native_record(regular(args.seal, 'native-build.json'), image)
    need(index['source_manifest_sha256'] == sha(COOP / 'SOURCE_MANIFEST.json'), 'campaign source changed; review before rebinding')
    need(index['runner_sha256'] == runner_inputs(), 'runner changed')
    second = independent_build(args, build)
    need(read(args.seal / 'independent-build.json') == second, 'independent build differs from campaign preflight')
    index['qualification_build'] = build
    index['independent_build'] = second
    output = args.output
    need(not output.exists() and not output.is_symlink() and not output.resolve().is_relative_to(ROOT), 'seal output must be new outside source')
    output.mkdir(parents=True)
    # Preserve raw campaign/build trees; selection and sealing never mutate them.
    shutil.copytree(COOP / 'source', output / 'source')
    shutil.copyfile(COOP / 'SOURCE_MANIFEST.json', output / 'SOURCE_MANIFEST.json')
    shutil.copyfile(COOP / 'build_repro.sh', output / 'build_repro.sh')
    projection.copy_bundle(args.seal / 'selected-bundle', output / 'bundle')
    write(output / 'QUALIFICATION.json', index)
    manifest = read(output / 'bundle/manifest.json')
    record = {'schema_version': 2, 'source_manifest_sha256': sha(COOP / 'SOURCE_MANIFEST.json'),
              'qualification_source_manifest_sha256': index['source_manifest_sha256'],
              'compiled_inputs': compiled_inputs(COOP), 'builder_sha256': sha(COOP / 'build_repro.sh'),
              'build_policy_sha256': digest_value(build_policy()), 'qualification_image': image,
              'helper': index['helper'], 'sanitizer': index['sanitizer'], 'checkpoint': index['checkpoint'],
              'reproducibility': {'runs': 2, 'comparison': 'bit-identical', 'binary_sha256': TARGET_NATIVE,
                                  'evidence_sha256': sha(args.seal / 'native-build.json')},
              'bundle': {'manifest_sha256': sha(output / 'bundle/manifest.json'), 'native_sha256': TARGET_NATIVE,
                         'runtime_sha256': manifest['files']['runtime.py'], 'dispatch_policy_sha256': manifest['files']['dispatch_policy.json']},
              'gate_index_sha256': sha(output / 'QUALIFICATION.json')}
    verify_record(record, output / 'bundle', output, release=False)
    write(output / 'BUILD.json', record)
    print('PASS component seal; release integration and final-source rebuilds remain required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-receipt', type=Path)
    parser.add_argument('--build-root', type=Path)
    parser.add_argument('--model-root', type=Path)
    parser.add_argument('--fly-root', type=Path)
    parser.add_argument('--helpers-root', type=Path)
    parser.add_argument('--sanitizer-root', type=Path, help='pinned NVIDIA package compute-sanitizer directory')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--seal', type=Path, help='completed local campaign to validate and seal into a new directory')
    parser.add_argument('--independent-build-root', type=Path)
    parser.add_argument('--independent-image-receipt', type=Path)
    parser.add_argument('--check-seal', type=Path, help='validate an existing component seal, without enabling it for release')
    args = parser.parse_args()
    try:
        if args.check_seal:
            need(not args.seal and not args.check_only, 'choose one operation')
            record = read(args.check_seal / 'BUILD.json')
            verify_record(record, args.check_seal / 'bundle', args.check_seal, release=False)
            for name, expected in read(args.check_seal / 'bundle/manifest.json')['files'].items():
                need(sha(regular(args.check_seal / 'bundle', name)) == expected, 'sealed artifact changed')
            print('PASS component seal; this is not release or fleet admission')
            return 0
        need(args.output is not None, '--output required')
        if args.seal:
            need(not args.check_only, '--check-only applies to the environment phase')
            seal(args)
        else:
            need(all(getattr(args, n) is not None for n in ('image_receipt','build_root','model_root','fly_root','helpers_root','sanitizer_root',
                 'independent_build_root', 'independent_image_receipt')), 'all fixture and independent-builder paths required')
            for name in ('output', 'image_receipt','build_root','model_root','fly_root','helpers_root','sanitizer_root',
                         'independent_build_root', 'independent_image_receipt'):
                path = getattr(args, name)
                need(path.is_absolute() and not path.is_symlink(), 'absolute non-symlink path required: ' + name)
            campaign(args)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError, Cancelled) as exc:
        diagnostics.report_failure(exc)
        return 9
    return 0


if __name__ == '__main__':
    diagnostics.install_exception_hook()
    raise SystemExit(main())
