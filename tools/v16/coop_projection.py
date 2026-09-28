# SPDX-License-Identifier: Apache-2.0
"""Closed shared facts from complete, privately retained component output."""
import json
import math
import re
import shutil

from _coop_qualification import need, read, regular, sha
import _diagnostics as diagnostics

VERSIONS = ('torch', 'cuda', 'sanitizer', 'nvcc', 'gcc')
LINKAGE = 'libcudart.so.13 => resolved'


def environment(raw):
    need('libcudart.so.13 =>' in raw['ldd'] and 'not found' not in raw['ldd'], 'shared cudart linkage')
    need(all(isinstance(raw.get(k), str) and raw[k].strip() for k in VERSIONS), 'missing toolchain identity')
    result = {k: raw[k] for k in ('status', 'checkpoint', 'bundle', 'exl3_sha256', 'fatpath_sha256')}
    result.update(ldd=LINKAGE, **{k: diagnostics.fingerprint(raw[k]) for k in VERSIONS})
    if 'gpu' in raw:
        gpu = raw['gpu']
        need(gpu['name'] == 'NVIDIA GB10' and gpu['capability'] == [12, 1]
             and isinstance(gpu['driver'], str) and gpu['driver'].strip(), 'GB10 identity')
        result['gpu'] = dict(name='NVIDIA GB10', capability=[12, 1], driver=diagnostics.fingerprint(gpu['driver']))
    return result


def command(argv):
    """Keep executable scope/arguments, hash only operator-local mount sources."""
    result = list(argv)
    for i, arg in enumerate(argv):
        if arg == '-v':
            source, target, mode = argv[i + 1].rsplit(':', 2)
            result[i + 1] = '/host-sha256/' + diagnostics.fingerprint(source) + ':' + target + ':' + mode
    return result


def number(value):
    need(type(value) in (int, float) and math.isfinite(value), 'nonfinite or nonnumeric gate fact')
    return value


def event(row):
    """Reject free text in consumed fields; discard unconsumed diagnostics."""
    stage = row.get('stage')
    fields = {
        'profile_identity': ('stage', 'native_sha256'),
        'compare': ('stage', 'label', 'pass', 'peak_rel', 'row_rel', 'rel_l2'),
        'profile': ('stage', 'rank', 'geometry', 'rows', 'pattern', 'stock', 'candidate'),
        'profile_complete': ('stage', 'rank', 'geometry'),
        'race_smoke_complete': ('stage', 'rank', 'pass'),
        'geometry2_sanitizer_complete': ('stage', 'rank', 'pass', 'rows', 'graph_replays', 'host_launches'),
        'production_policy_complete': ('stage', 'pass', 'choices'),
    }
    if row.get('control') == 'coop-h1-raw-output':
        fields[None] = ('schema_version', 'control', 'mode', 'status', 'ep_rank', 'qualification_geometry',
            'test_source_sha256', 'bundle_manifest_sha256', 'policy_sha256', 'native_sha256', 'control_sha256',
            'exl3_sha256', 'fatpath_sha256', 'thresholds', 'real_weight_comparisons',
            'expected_real_weight_comparisons', 'detector_rejections', 'detector_outcome', 'serving_path_reachable',
            'payload_sha256', *[k for k in ('minimum_observed_peak_rel', 'maximum_observed_peak_rel', 'injection') if k in row])
    if stage not in fields:
        return None
    result = {k: row[k] for k in fields[stage]}
    for key in ('peak_rel', 'row_rel', 'rel_l2', 'minimum_observed_peak_rel', 'maximum_observed_peak_rel'):
        if key in result:
            number(result[key])
    words = set(fields) | {'coop-h1-raw-output', 'baseline', 'perturb', 'PASS', 'stock',
                          'unmodified-pass', 'expected-peak-tolerance-failure-observed', 'ep_uniform', 'concentrated'}
    def checked(value):
        if isinstance(value, str):
            need(value in words or re.fullmatch(r'[0-9a-f]{64}', value)
                 or re.fullmatch(r'profile/rank=[0-2]/geometry=[0-2]/rows=[0-9]{1,2}/(?:ep_uniform|concentrated)', value),
                 'unexpected text in gate fact')
        elif isinstance(value, dict):
            need(all(k in ('min_ms', 'median_ms', 'max_ms', 'peak', 'row_peak', 'rel_l2', 'element', 'tolerance_multiplier')
                     or re.fullmatch(r'[0-9]{1,2}', k) for k in value), 'unexpected gate fact key')
            for v in value.values(): checked(v)
        elif isinstance(value, list):
            for v in value: number(v)
        elif type(value) is not bool:
            number(value)
    for value in result.values(): checked(value)
    return result


def gate_log(name, text):
    from coop_evidence import sanitizer
    stages = ({'profile_identity', 'compare', 'profile', 'profile_complete'} if name.startswith('profile-') else
              {'geometry2_sanitizer_complete'} if name.startswith('geometry2-') else
              {'race_smoke_complete'} if name.startswith('smoke-') else {'production_policy_complete'})
    raw = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
    rows = [event(row) for row in raw if (row.get('control') == 'coop-h1-raw-output'
            if name.startswith('h1-') else row.get('stage') in stages)]
    result = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows if row is not None)
    if name.startswith(('smoke-', 'geometry2-')):
        tool = name.rsplit('-', 1)[1]
        # Inspect the complete output before discarding arbitrary launch names,
        # banners and paths. Preserve instrumented launch count and clean summary.
        count = sanitizer(text, tool)
        result += ''.join(f'========= Launch #{i}\n=========   Kernel: exl3_moe_coop_instrumented\n' for i in range(1, count + 1))
        summary = 'ERROR SUMMARY: 0 errors' if tool == 'memcheck' else 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
        result += '========= ' + summary + '\n'
    return result


def copy_bundle(source, target):
    """Copy authenticated assets, never build logs, intermediates or sidecars."""
    files = read(source / 'manifest.json')['files']
    # These are the reviewed historical and reproducible candidate reports.
    # A new compiler report requires review, even if it yields identical code.
    need(sha(regular(source, 'toolchain.txt')) in {
        '09f5ba5ca19ef6d9e064667fe5059810cc1feb1e05b71b38d0d5c05e58076ce2',
        '1815f9aaad395bd45a02937234fe15afd2f808c0ea2afcc6417c410580cd6e28'}, 'unreviewed toolchain report')
    target.mkdir()
    for name in ('manifest.json', *files):
        src = regular(source, name)
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
