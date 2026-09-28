# SPDX-License-Identifier: Apache-2.0
"""Validate retained component gate output before creating a seal."""
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'recipe/scripts'))
sys.path.insert(0, str(ROOT / 'recipe/overlays/v16/coop/source'))
from _coop_qualification import TARGET_NATIVE, need, read, regular, sha, gate_names
import coop_h1_control as h1
import select_policy

KERNEL_FILTER = 'kns=exl3_moe_coop_'


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]


def sanitizer(text, tool):
    summary = ('ERROR SUMMARY: 0 errors' if tool == 'memcheck' else
               'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)')
    summaries = re.findall(r'(?m)^========= (?:ERROR SUMMARY:.*|RACECHECK SUMMARY:.*)$', text)
    need(summaries == ['========= ' + summary], 'sanitizer missing or contradictory summary')
    need(not re.search(r'(?i)internal sanitizer error|didn.t track|no attachable process|no kernels|not supported|========= Error:', text),
         'sanitizer incomplete')
    # --dump-kernel-launches includes filtered and unfiltered launches. Require
    # candidate launch records explicitly marked not filtered; a zero summary
    # alone (or host-side launch counter) cannot prove instrumentation.
    launches = re.findall(r'(?ms)^========= Launch #[^\n]*\n.*?(?=^========= Launch #|\Z)', text)
    checked = [block for block in launches if re.search(r'^=========   Kernel: .*exl3_moe_coop_', block, re.M)
               and '(filtered:' not in block.splitlines()[0]]
    need(checked, 'no explicitly instrumented candidate launches; retain output and review tool format')
    return len(checked)


def validate_environment(path, *, gpu):
    environment = read(path)
    need(environment.get('status') == 'PASS'
         and environment.get('exl3_sha256') == '71e7118bd5af385821d7cb23e96fb154a3f31e1e835599a1082c72abb3aeb174'
         and environment.get('fatpath_sha256') == '69309df5f236502ec8cf55648f72369c20052c646ebd8850cd88272be881b48e'
         and 'libcudart.so.13 =>' in environment.get('ldd', '') and 'not found' not in environment['ldd']
         and all(environment.get(k) for k in ('nvcc', 'gcc', 'sanitizer', 'torch', 'cuda')), 'stage-8 environment evidence')
    need(('gpu' in environment) == gpu, 'environment GPU scope differs')
    if gpu:
        need(environment['gpu'].get('name') == 'NVIDIA GB10'
             and environment['gpu'].get('capability') == [12, 1] and environment['gpu'].get('driver'), 'GB10 identity')
    return environment


def validate_gate(name, root, identity, raw_bundle):
    kind = gate_names()[name]
    receipt = read(regular(root, name + '.json'))
    log = regular(root, name + '.log')
    need(receipt == {'schema_version': 1, 'name': name, 'kind': kind, 'status': 'PASS',
                    'exit_code': 0, 'identity': identity, 'log_sha256': sha(log)}, 'gate receipt drift: ' + name)
    stage_dir = root / ('container-' + name)
    execution = read(regular(stage_dir, 'execution.json'))
    need(execution.get('exit_code') == 0 and execution.get('command') == read(root / 'plan.json')[name]
         and execution.get('started_at') < execution.get('completed_at'), 'execution binding: ' + name)
    from qualify_coop import matrix
    command = execution['command']
    entry = '/src/tools/v16/coop_environment.py'
    need(command.count(entry) == 1 and command[command.index(entry) + 1:] == dict(matrix())[name],
         'gate command does not match reviewed matrix: ' + name)
    need(command.count('--gpus') == 1 and command[command.index('--gpus') + 1] == 'device=0'
         and command[command.index('--network') + 1] == 'none', 'gate container scope')
    validate_environment(regular(stage_dir, 'environment.json'), gpu=True)
    rows = events(log)
    if kind == 'h1':
        match = re.fullmatch(r'h1-r([0-2])-g([0-2])-(baseline|perturb)', name)
        rank, geometry, mode = match.groups()
        proof = read(regular(root, name + '-proof.json'))
        h1.verify_payload_hash(proof)
        expected = {'schema_version': 1, 'control': 'coop-h1-raw-output', 'mode': mode, 'status': 'PASS',
                    'ep_rank': int(rank), 'qualification_geometry': int(geometry),
                    'test_source_sha256': sha(ROOT / 'recipe/overlays/v16/coop/source/test_cuda_integration.py'),
                    'bundle_manifest_sha256': sha(raw_bundle / 'manifest.json'),
                    'policy_sha256': sha(raw_bundle / 'dispatch_policy.json'), 'native_sha256': TARGET_NATIVE,
                    'control_sha256': sha(ROOT / 'tools/v16/coop_h1_control.py'),
                    'exl3_sha256': '71e7118bd5af385821d7cb23e96fb154a3f31e1e835599a1082c72abb3aeb174',
                    'fatpath_sha256': '69309df5f236502ec8cf55648f72369c20052c646ebd8850cd88272be881b48e',
                    'thresholds': {'peak': .003, 'row_peak': .05, 'rel_l2': .05},
                    'real_weight_comparisons': 156, 'expected_real_weight_comparisons': 156,
                    'detector_rejections': 0 if mode == 'baseline' else 156,
                    'detector_outcome': 'unmodified-pass' if mode == 'baseline' else 'expected-peak-tolerance-failure-observed',
                    'serving_path_reachable': False}
        need(all(proof.get(k) == v for k, v in expected.items()), 'H1 evidence incomplete: ' + name)
        need(proof in rows, 'H1 proof not in retained command output')
        if mode == 'perturb':
            need(proof.get('minimum_observed_peak_rel', 0) > .003, 'H1 detector sensitivity')
    elif kind == 'sanitizer':
        tool = name.rsplit('-', 1)[1]
        sanitizer(log.read_text(), tool)
        stage = 'geometry2_sanitizer_complete' if name.startswith('geometry2') else 'race_smoke_complete'
        done = [r for r in rows if r.get('stage') == stage]
        rank = int(re.search(r'-r([0-2])-', name)[1])
        need(len(done) == 1 and done[0].get('pass') is True and done[0].get('rank') == rank,
             'sanitizer application completion')
        if stage.startswith('geometry2'):
            need(done[0].get('rows') == [2,3,4,5,8,10,12,16,18,20,24,28,32,64]
                 and done[0].get('graph_replays') == 87 and done[0].get('host_launches', 0) > 0,
                 'geometry-2 coverage')
    elif kind == 'policy':
        done = [r for r in rows if r.get('stage') == 'production_policy_complete']
        policy = read(root / 'selected-bundle/dispatch_policy.json')
        choices = {str(n): policy['rows'].get(str(n), 'stock') for n in (*select_policy.ROWS, 33, 65)}
        need(len(done) == 1 and done[0] == {'stage': 'production_policy_complete', 'pass': True, 'choices': choices},
             'production policy completion/choices')
    return {'kind': kind, 'status': 'PASS', 'receipt_sha256': sha(root / (name + '.json')), 'log_sha256': sha(log),
            'environment_sha256': sha(stage_dir / 'environment.json'), 'execution_sha256': sha(stage_dir / 'execution.json')}


def validate_campaign(root):
    campaign = read(root / 'campaign.json')
    need(campaign.get('schema_version') == 1 and campaign.get('status') == 'COMPLETE', 'campaign incomplete')
    identity = campaign['identity']
    need(identity['native_sha256'] == TARGET_NATIVE, 'candidate native drift')
    raw = root / 'raw-bundle'
    for bundle in (raw, root / 'selected-bundle'):
        manifest = read(bundle / 'manifest.json')
        for name, expected in manifest['files'].items():
            need(sha(regular(bundle, name)) == expected, 'bundle changed: ' + name)
    logs = [regular(root, f'profiles/rank{r}-geo{g}.jsonl') for r in range(3) for g in range(3)]
    for path in logs:
        rank, geometry = map(int, re.fullmatch(r'rank([0-2])-geo([0-2]).jsonl', path.name).groups())
        for row in events(path):
            if row.get('stage') in ('profile', 'profile_complete'):
                need((row['rank'], row['geometry']) == (rank, geometry), 'profile filename/range binding')
            if row.get('stage') == 'profile':
                import math
                for impl in ('stock', 'candidate'):
                    times = row[impl]
                    need(set(times) == {'min_ms', 'median_ms', 'max_ms'}
                         and all(type(v) in (int, float) and math.isfinite(v) for v in times.values())
                         and 0 < times['min_ms'] <= times['median_ms'] <= times['max_ms'], 'profile timing statistics')
    policy = select_policy.select(logs, TARGET_NATIVE)
    need(policy == read(root / 'selected-bundle/dispatch_policy.json'), 'policy differs from complete measured profiles')
    gates = {name: validate_gate(name, root, identity, raw) for name in gate_names()}
    for r in range(3):
        for g in range(3):
            need((root / f'profile-r{r}-g{g}.log').read_bytes() == (root / f'profiles/rank{r}-geo{g}.jsonl').read_bytes(),
                 'profile log copy differs')
    return {**identity, 'schema_version': 1, 'status': 'PASS', 'gates': gates,
            'profile_log_sha256': {path.name: sha(path) for path in logs}}
