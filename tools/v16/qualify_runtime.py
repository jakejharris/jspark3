#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Qualify one prepared production-stock boot and produce runnable admission receipts.

Keep ordinary traffic blocked throughout. Uses only shipped tools and the operator's
existing Docker/SSH access. Output must be a new directory outside source and recipe.
"""
import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import admission_gate
import apc_gate
import prefill_gate
from v16_common import ERROR_RE, QAError, load_json, sha256_json, write_json, diagnostics

HYGIENE_ROOTS = ['/models', '/recipe', '/sources/fly', '/evidence',
                 '/root/.cache/vllm', '/root/.triton/cache', '/root/.tilelang/cache']
EPOCH_READ = '''import hashlib,json,pathlib,stat
out={}
for name in ('triar-epoch.json','adaptive-k-epoch.json','sched-epoch.json','fatpath-epoch.json'):
 p=pathlib.Path('/evidence')/name
 try: info=p.lstat()
 except FileNotFoundError: out[name]=None;continue
 if not stat.S_ISREG(info.st_mode) or info.st_size>16384:raise SystemExit('unsafe runtime epoch')
 out[name]=hashlib.sha256(p.read_bytes()).hexdigest()
print(json.dumps(out,sort_keys=True))
'''


def need(condition, reason):
    if not condition:
        raise QAError(reason)


def snapshot(fleet, values, manifest):
    rows = []
    for binding in manifest['containers']:
        rank, cid = binding['rank'], binding['container_id']
        item = fleet.inspect_identity(values, binding)
        state = item['State']
        need(state['Running'] and not state['OOMKilled'] and item['RestartCount'] == 0,
             f'rank{rank} is not a healthy unrestarted boot')
        proc = fleet.remote(values, rank, ['docker', 'exec', cid, 'python3', '-B', '-S', '-c', EPOCH_READ])
        epochs = json.loads(proc.stdout)
        need(set(epochs) == {'triar-epoch.json', 'adaptive-k-epoch.json', 'sched-epoch.json', 'fatpath-epoch.json'}
             and all(value is None for value in epochs.values()),
             'operator qualification requires boot defaults with no runtime epoch overrides')
        rows.append({'rank': rank, 'container_id': cid, 'started_at': state['StartedAt'], 'epochs': epochs})
    need([row['rank'] for row in rows] == [0, 1, 2], 'three rank bindings required')
    return rows


def triar_off(logs, enabled):
    if enabled:
        need(not any(marker in logs for marker in ('dual-captured', 'dual-replay', 'TRIAR ON', 'TRIAR OFF')),
             'TRIAR graph activity contradicts the inactive Cadence path')
    return {'resident': enabled, 'mode': 'INACTIVE_NCCL_B45' if enabled else 'off'}


def verified(path, manifest_sha, fleet):
    doc = load_json(path)
    need(doc.get('payload_sha256') == fleet.sha_bytes(fleet.canonical({k: v for k, v in doc.items() if k != 'payload_sha256'})),
         'verify receipt hash drift')
    need(doc.get('status') == 'VERIFY_PASS' and doc.get('manifest_sha256') == manifest_sha
         and doc.get('production_stock', {}).get('status') == 'PASS', 'stock verification not PASS for this boot')
    return doc


def prefill_pass(doc, blocking):
    need(doc['receipt_sha256'] == sha256_json({k: v for k, v in doc.items() if k != 'receipt_sha256'}),
         'prefill receipt hash drift')
    need(doc['floor_tok_s'] == 1100, 'prefill floor differs')
    turns = [row for row in doc['gate'] if row['tag'].startswith('pi-turn')]
    need(len(turns) == 8 and all(row['requests'] == 1 for row in doc['gate']), 'prefill workload/traffic differs')
    if blocking:
        need(doc['verdict'] == 'PASS' and not prefill_gate.evaluate(doc['gate'], doc['compile_lines_after_warmup'], 1100),
             'post-hygiene prefill did not pass')
        need(any(row['new_tokens'] >= 2500 for row in turns), 'no qualifying long prefill canary')


def run(args, fleet):
    values = fleet.load_env(args.env_file)
    fleet.validate_env(values)
    need(values['JSPARK3_V16_PROFILE'] == 'production-stock' and values['ABLIT'] == '0'
         and values['JSPARK3_V16_COOP'] == '0' and values['JSPARK3_V16_APC_LRU'] == '1',
         'operator admission supports production-stock, ABLIT=0, coop=0, APC=1 only')
    manifest = fleet.bound_manifest(args.manifest, values, require_all=True, require_started=True)
    manifest_sha = fleet.sha_file(args.manifest)
    env_sha = fleet.sha_file(args.env_file)
    for binding in manifest['containers']:
        fleet.verify_remote_recipe(values, binding['rank'], manifest['recipe_manifest_sha256'])
    initial = snapshot(fleet, values, manifest)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    base = f"http://{values['JSPARK_MASTER_ADDR']}:{values['JSPARK_API_PORT']}"
    controller = args.recipe / 'scripts/fleetctl.py'
    identity = {'coop': 'off', 'adaptive-k': values['GLM53_ADAPTIVE_K'], 'dense-fp8': values['JSPARK3_V16_DENSE_FP8']}

    def command(name, script, arguments, allowed=(0,)):
        proc = subprocess.run([sys.executable, '-B', str(script), *map(str, arguments)],
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        diagnostics.retain(proc.stdout + proc.stderr)
        fleet.save_diagnostics(out / (name + '.log'), proc.stdout + proc.stderr)
        if proc.returncode not in allowed:
            error = QAError('qualification command refused')
            error.raw_output = proc.stdout + proc.stderr
            error.safe_diagnostics = diagnostics.structure(error.raw_output)
            error.safe_diagnostics.update(command=name, rank=None, exit_code=proc.returncode)
            raise error
        return proc.returncode

    def verify(name):
        command(name, controller, ['verify', '--env-file', args.env_file, '--manifest', args.manifest,
            '--output', out / (name + '.json'), '--log-output', out / (name + '-rank0.log')])
        return verified(out / (name + '.json'), manifest_sha, fleet)

    def inputs():
        # Only named products of this producer are shared evidence. Diagnostics
        # and unrelated files must never become admission dependencies.
        names = ['verify-first.json', 'verify-first.log', 'verify-final.json', 'verify-final.log',
                 'first-prompt.json', 'prefill-first.json', 'prefill-first.log',
                 'prefill-post.json', 'prefill-post.log', 'apc.json', 'apc.log', 'apc-fixtures.json']
        names += [f'{kind}-rank{rank}.{suffix}' for rank in range(3)
                  for kind, suffix in (('hygiene', 'json'), ('triar-inactive', 'json'), ('post-warmup', 'log'))]
        return {name: fleet.sha_file(out / name) for name in sorted(names) if (out / name).is_file()}

    def public_boot():
        return [{"rank": row['rank'], "container_id": row['container_id'],
                 "started_at_sha256": diagnostics.fingerprint(row['started_at']), "epochs": row['epochs']}
                for row in initial]

    def same_boot():
        need(fleet.sha_file(args.manifest) == manifest_sha and fleet.sha_file(args.env_file) == env_sha
             and snapshot(fleet, values, manifest) == initial, 'boot, environment or runtime epoch changed during qualification')

    verify('verify-first')
    same_boot()
    first = {'schema': admission_gate.FIRST_SCHEMA, 'verdict': 'PASS', 'identity_config': identity,
             'producer': 'qualify_runtime.py', 'manifest_sha256': manifest_sha, 'boot': public_boot(),
             'correctness': 'fleetctl arithmetic, focused and long-context witnesses', 'evidence_sha256': inputs()}
    write_json(out / 'first-prompt.json', first)
    gate_args = ['--base-url', base, '--env-file', args.env_file]
    command('prefill-first', Path(prefill_gate.__file__), [*gate_args, '--out', out / 'prefill-first.json'], allowed=(0, 1))
    prefill_first = load_json(out / 'prefill-first.json')
    prefill_pass(prefill_first, blocking=False)
    same_boot()
    for row in initial:
        rank = row['rank']
        proc = fleet.remote(values, rank, ['docker', 'exec', '--user', '0', row['container_id'],
            'python3', '-B', '-S', '/recipe/scripts/page_cache_hygiene.py', *HYGIENE_ROOTS], check=False)
        # Runs with read access to the exact bound container's mounted files, avoiding
        # the root-owned cache/evidence EACCES seen with host-user-only walks.
        doc = json.loads(proc.stdout)
        private = diagnostics.retain(proc.stdout + proc.stderr, out / f'hygiene-rank{rank}.json')
        valid = (doc.get('verdict') == 'PASS' and not doc.get('errors')
                 and type(doc.get('files')) is int and doc['files'] > 0 and type(doc.get('bytes')) is int)
        write_json(out / f'hygiene-rank{rank}.json', {'rank': rank, 'container_id': row['container_id'],
            'verdict': 'PASS' if valid else 'FAIL', 'files': doc.get('files') if valid else 0,
            'bytes': doc.get('bytes') if valid else 0, 'errors': [] if valid else ['hygiene refused'],
            'response_sha256': diagnostics.fingerprint(doc), 'private_diagnostic': Path(private).name if private else None})
        need(proc.returncode == 0 and doc['verdict'] == 'PASS' and not doc['errors'] and doc['files'] > 0,
             f'rank{rank} hygiene incomplete')
    same_boot()
    command('prefill-post', Path(prefill_gate.__file__), [*gate_args, '--out', out / 'prefill-post.json'])
    prefill_pass(load_json(out / 'prefill-post.json'), blocking=True)
    command('apc', Path(apc_gate.__file__), [*gate_args, '--expect', 'finehit', '--fixtures', out / 'apc-fixtures.json',
                                          '--out', out / 'apc.json'])
    apc = load_json(out / 'apc.json')
    need(apc['analysis']['verdict'] == 'PASS' and apc['expect'] == 'finehit'
         and all(v == 'FAIL' for v in apc['checker_controls'].values()), 'fine-hit cache gate or negative controls failed')
    verify('verify-final')
    all_logs = ''
    for row in initial:
        proc = fleet.remote(values, row['rank'], ['docker', 'logs', row['container_id']])
        all_logs += proc.stdout + proc.stderr
        proc = fleet.remote(values, row['rank'], ['docker', 'logs', '--since', prefill_first['gate_start'], row['container_id']])
        text = proc.stdout + proc.stderr
        fleet.save_diagnostics(out / f"post-warmup-rank{row['rank']}.log", text)
        need(not prefill_gate.compile_lines(text) and not ERROR_RE.search(text),
             f"rank{row['rank']} compiled or reported an engine error after warmup")
    inactive = triar_off(all_logs, values.get('JSPARK3_TRIAR', '0') == '1')
    if values.get('JSPARK3_TRIAR', '0') == '1':
        for row in initial:
            proc = fleet.remote(values, row['rank'], ['docker', 'exec', row['container_id'], 'python3', '-B', '-S',
                '/recipe/scripts/triar_inactive.py', '--started-at', row['started_at']])
            proof = json.loads(proc.stdout)
            private = diagnostics.retain(proc.stdout + proc.stderr, out / 'triar-inactive.json')
            write_json(out / f"triar-inactive-rank{row['rank']}.json", {
                'verdict': 'PASS' if proof.get('verdict') == 'PASS' else 'FAIL',
                'attestation': {'rank': row['rank']}, 'response_sha256': diagnostics.fingerprint(proof),
                'private_diagnostic': Path(private).name if private else None})
            need(proof['verdict'] == 'PASS' and proof['attestation']['rank'] == row['rank'],
                 'TRIAR inactive attestation failed')
    same_boot()
    final = {'schema': admission_gate.FINAL_SCHEMA, 'verdict': 'PASS', 'identity_config': identity,
             'producer': 'qualify_runtime.py', 'manifest_sha256': manifest_sha, 'environment_sha256': env_sha,
             'qualification_tools_sha256': {name: fleet.sha_file(Path(__file__).with_name(name)) for name in
                 ('qualify_runtime.py', 'prefill_gate.py', 'apc_gate.py', 'admission_gate.py', 'v16_common.py')},
             'boot': public_boot(), 'triar': inactive, 'completed_at': datetime.now(timezone.utc).isoformat(),
             'prefill_floor_tok_s': 1100, 'evidence_sha256': inputs(),
             'input_hashes': {'client_evidence': [{'schema': admission_gate.FIRST_SCHEMA,
                                                 'sha256': fleet.sha_file(out / 'first-prompt.json')}]}}
    final['payload_sha256'] = sha256_json(final)
    write_json(out / 'finalize.json', final)
    command('admission', Path(admission_gate.__file__), ['--first-prompt', out / 'first-prompt.json',
             '--finalize', out / 'finalize.json', '--out', out / 'admission.json'])
    print('PASS operator admission: ' + str(out / 'admission.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True, help='prepared runtime recipe directory')
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new evidence directory outside source/recipe')
    args = parser.parse_args()
    args.recipe, args.output = args.recipe.resolve(), args.output.resolve()
    source = Path(__file__).resolve().parents[2]
    if args.output.is_relative_to(source) or args.output.is_relative_to(args.recipe):
        parser.error('evidence must be outside the source export and prepared recipe')
    sys.path.insert(0, str(args.recipe / 'scripts'))
    fleet = importlib.import_module('fleetctl')
    try:
        run(args, fleet)
    except Exception as exc:
        diagnostics.report_failure(exc, args.output / 'qualification-failure.json', command='qualify runtime')
        return 1
    return 0


if __name__ == '__main__':
    diagnostics.install_exception_hook()
    raise SystemExit(main())
