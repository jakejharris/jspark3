#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Healthy three-rank verify: real producers/gates, synthetic external IO only."""
import argparse
import copy
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import io
import itertools
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'recipe/scripts'), str(ROOT / 'tools/v16')]
import _fleetctl as fleet
import _contracts
import focused_witness
import long_context_witness
import qualify_runtime
import resource_scope
import verify_stock

SECRET = 'opaqueLoaderFilenameCredentialValue'


class Response(io.BytesIO):
    status = 200

    def __init__(self, value):
        super().__init__(value if isinstance(value, bytes) else json.dumps(value).encode())


class HealthyFleet:
    def __init__(self, root, fault='', historical=False):
        self.root, self.fault, self.historical = root, fault, historical
        self.witnesses, self.native_records, self.calls = {}, {}, []
        self.counter = itertools.count(1)
        self.values = fleet.load_env(ROOT / 'recipe/.env.example')
        self.values['JSPARK3_V16_COOP'] = '0'
        for rank in range(3):
            self.values[f'JSPARK_RANK{rank}_HOST'] = f'rank{rank}.test'
            self.values[f'JSPARK_RANK{rank}_ADDR'] = f'127.0.0.{rank + 10}'
            self.values[f'JSPARK_FABRIC_ADDRS_{rank}'] = self.values[f'JSPARK_FABRIC_ADDRS_{rank}'].replace('192.0.2.', '127.1.0.')
        self.values['JSPARK_MASTER_ADDR'] = self.values['JSPARK_RANK0_ADDR']
        self.env = root / 'operator.env'
        self.env.write_text(''.join(f'{k}={v}\n' for k, v in self.values.items()))
        self.recipe_sha = fleet.recipe_manifest_sha256()
        self.manifest_path = root / 'service.json'
        self.manifest = dict(schema_version=1, candidate='jspark3', grade='ENGINEERING-EVIDENCE',
            configuration_sha256=fleet.configuration_digest(self.values), preflight_sha256='b'*64,
            recipe_manifest_sha256=self.recipe_sha, image_manifest=fleet.IMAGE_MANIFEST,
            image_config=fleet.IMAGE_CONFIG, start_order=[2, 1, 0], containers=[], status='STARTED')
        self.configs, self.items, self.logs = {}, {}, {}
        for rank in range(3):
            cid = str(rank + 1) * 64
            receipt = dict(schema_version=2, manifest_digest=fleet.IMAGE_MANIFEST, config_digest=fleet.IMAGE_CONFIG,
                verification='host-observed-inspect-bound-create', container_id=cid, rank=rank,
                preflight_sha256=self.manifest['preflight_sha256'], recipe_manifest_sha256=self.recipe_sha)
            receipt['payload_sha256'] = fleet.sha_bytes(fleet.canonical(receipt))
            binding = dict(rank=rank, container_id=cid, name=fleet.container_name(rank), image_config=fleet.IMAGE_CONFIG,
                           image_receipt_sha256=fleet.sha_bytes(fleet.canonical(receipt)))
            self.manifest['containers'].append(binding)
            self.configs[rank] = dict(image_receipt=receipt, image_receipt_sha256=binding['image_receipt_sha256'],
                target_runtime_config='55201c73ed092c5a77f9b87ce40298edb450790ad864c1256cb6ca3a182683bd',
                draft_runtime_config='c9f0c3a6c41f8a226fb31a1fb7817cea274d1f4b7b0d2e4d787d38c0f508283f')
            self.items[rank] = self.inspect_item(rank, cid)
            self.logs[rank] = self.startup_log(rank)
        self.manifest['payload_sha256'] = fleet.sha_bytes(fleet.canonical(self.manifest))
        self.manifest_path.write_bytes(fleet.canonical(self.manifest))

    def inspect_item(self, rank, cid):
        work = f"{self.values['JSPARK_WORK_ROOT']}/rank{rank}"
        mounts = {'/recipe': (self.values['JSPARK_RECIPE_ROOT'], False),
                  '/sources/fly': (self.values['JSPARK_FLY_ROOT'], False),
                  '/models': (self.values['JSPARK_MODEL_ROOT'], False), '/evidence': (work + '/evidence', True)}
        mounts.update({dest: (work + '/cache/' + key, True) for key, dest in
                       [('vllm', '/root/.cache/vllm'), ('triton', '/root/.triton/cache'), ('tilelang', '/root/.tilelang/cache')]})
        return dict(Id=cid, Name='/' + fleet.container_name(rank), Image=fleet.IMAGE_CONFIG, RestartCount=0,
            State=dict(Running=True, OOMKilled=False, ExitCode=0, Pid=100 + rank, StartedAt='2026-09-28T00:00:00Z'),
            Config=dict(Image=fleet.IMAGE, Hostname=fleet.container_name(rank), WorkingDir='/vllm-workspace',
                Entrypoint=['/usr/bin/python3'], Cmd=['-S', '/recipe/scripts/triar_entry.py', *fleet.server_argv(self.values, rank)],
                Env=fleet.rank_env(self.values, rank, self.manifest['preflight_sha256'], self.recipe_sha),
                Labels={'org.opencontainers.image.title': 'jspark3-recipe', 'jspark3.release': fleet.RELEASE_LABEL,
                        'jspark3.rank': str(rank), 'jspark3.grade': 'engineering-evidence', 'jspark3.b45': 'cadence-reference'}),
            HostConfig=dict(NetworkMode='host', IpcMode='host', ShmSize=fleet.SHM, Memory=fleet.MEMORY, MemorySwap=fleet.MEMORY,
                RestartPolicy={'Name': 'no'}, CgroupnsMode='private', CapAdd=['CAP_IPC_LOCK'], SecurityOpt=['label=disable'],
                Annotations={resource_scope.ANNOTATION: resource_scope.VALUE},
                Devices=[dict(PathOnHost='/dev/infiniband', PathInContainer='/dev/infiniband', CgroupPermissions='rwm'),
                         dict(PathOnHost='/dev/dri/card0', PathInContainer='/dev/dri/card0', CgroupPermissions='rwm')],
                DeviceRequests=[dict(Driver='', Count=-1, DeviceIDs=None, Capabilities=[['gpu']], Options={})],
                Ulimits=[dict(Name='memlock', Soft=-1, Hard=-1), dict(Name='nofile', Soft=65536, Hard=65536)]),
            Mounts=[dict(Destination=dst, Source=src, RW=rw, Type='bind') for dst, (src, rw) in mounts.items()])

    def startup_log(self, rank):
        cap, credit, pool = fleet.V14_ORDINARY_CAP_BYTES[rank], 1006632960, 1879048192
        identity = dict(rank=str(rank), profile='full', draft_block=640, mla_block=2560,
            pool_open_bytes=pool, credited_bytes=credit, carved_bytes=credit,
            ordinary_profiled_bytes=cap + 1024, ordinary_cap_bytes=cap, ordinary_budget_bytes=cap, budget_calls=1,
            budget=dict(rank=rank, profiled_bytes=cap + 1024, cap_bytes=cap, ordinary_bytes=cap,
                        clamped_bytes=1024, credited_bytes=credit, total_bytes=cap + credit),
            num_blocks=1, plan=dict(from_pool=[0], pool_sizes=[credit]),
            placement=[dict(index=0, bytes=credit, owners=[dict(group=0, layer='layer')])])
        dense = dict(status='JSPARK3_V16_DENSE_FP8_FINALIZE_PASS', rank=rank, mode='trunk',
            scheme='e4m3fn_per_output_channel', activation_dtype='bfloat16', w8a8=False,
            runtime_modules=169, logical_tensors=225,
            category_counts=dict(kda_o=34, mla_fused_qkv_a=11, mla_q_b=11, mla_kv_b=11, mla_o=11,
                                 shared_gate_up=42, shared_down=42, dense_gate_up=3, dense_down=3, lm_head=1),
            b45_kda_composite_owner='B45_UNCHANGED', b45_kda_composite_converted=False,
            b45_kda_input_modules=34, b45_kda_fg_modules=34, b45_kda_ownership_audited=True)
        if rank == 2 and self.fault == 'v14': identity['budget_calls'] = 2
        if rank == 2 and self.fault == 'v16': dense['runtime_modules'] -= 1
        return '\n'.join(['Capturing dflash2 CUDA graphs (FULL): 100%|done| 5/5', 'Application startup complete.',
            fleet.B5_RECEIPT_MARKER, f'JSPARK3_V14_PROFILE rank={rank} profile=full ',
            f'JSPARK3_V14_IDENTITY rank={rank} ok=1 ' + json.dumps(identity),
            '[cadence-sched:v6] mode=cap cap=2560 ', '[cadence-sched:v6] identity ok=1 ', '[cadence-sched:v6] kv identity ok=1 ',
            f'[jspark3-v16:coop] rank={rank} state=off', f'[jspark3-v16:adaptive-k] rank={rank} state=ema',
            f'[jspark3-v16:dense-fp8] rank={rank} state=trunk', 'JSPARK3_V16_DENSE_FP8_RECEIPT=' + json.dumps(dense),
            '[glm53-apc-per-group] retention_by_group=fixture (global=fixture swa_env=0 eagle_min_exempt=[6] low_priority=[] ',
            'private loader filename: ' + SECRET])

    def stock_response(self, rank):
        """Execute verify_stock.main; substitute only container files/hash reads."""
        package = '/usr/local/lib/python3.12/dist-packages/'
        config = lambda name: json.loads((ROOT / 'recipe/config' / name).read_text())
        swa, loader, profile = config('swa-contract.json'), config('loader-audit.json'), config('profile.json')
        files, hashes = {}, {}
        native_loaders = {}
        for kind, shards in [('target', 120), ('draft', 1)]:
            loaded = dict(state='COMPLETE', loader='instanttensor', rank=rank, kind=kind, shards=shards,
                tensors=100, copy_ownership=True, bytes=175622979576 if kind == 'target' else 1024,
                first_tensor_descriptors=dict(nofile=[65536, 65536], open_fd_count=100),
                ordered_headers_sha256=('c' if kind == 'target' else 'd') * 64, filename=SECRET)
            if rank == 2 and kind == 'target':
                if self.fault == 'incomplete': loaded['state'] = 'LOADING'
                if self.fault == 'missing-census': loaded.pop('ordered_headers_sha256')
                if self.fault == 'malformed-census': loaded['ordered_headers_sha256'] = SECRET
                if self.fault == 'census-drift': loaded['ordered_headers_sha256'] = 'e' * 64
            native_loaders[kind] = loaded
            files['/evidence/loader-' + kind + '.json'] = loaded
            files['/evidence/loader-memory-' + kind + '.json'] = dict(rank=rank, kind=kind, **{'pass': True},
                backend='URING', completed_target_files_eligible=120 if kind == 'draft' else 0,
                io_buffer_bound=1, tensor_buffer_bound=2, kernel_high_watermark_bytes=3, required_cuda_free=7,
                after=dict(cuda_free=8), after_empty_cache=dict(host={'MemAvailable': 8 * 1024**3 + 7}))
            files['/evidence/loader-memory-' + kind + '-agreement.json'] = dict(rank=rank, kind=kind, all_ranks_pass=True, local_error=None)
        ablation = dict(schema_version=1, ablit=0, rank=rank, state='DISABLED', applied_layers=[])
        files['/evidence/ablit-receipt.json'] = ablation
        files['/evidence/swa-receipt.json'] = dict(contract_sha256=fleet.sha_file(ROOT/'recipe/config/swa-contract.json'),
            rank=str(rank), ablit=0, state='APPLIED', targets={name: row['after'] for name, row in swa['targets'].items()})
        instant = config('image-build-policy.json')['instanttensor_files']
        files['/opt/jspark3-v13/instanttensor-files.json'] = instant
        hashes.update({package + name: digest for name, digest in instant.items()})
        hashes.update({package + name: row['after'] for name, row in swa['targets'].items()})
        hashes[package + 'vllm/model_executor/model_loader/default_loader.py'] = loader['after']
        hashes[package + 'vllm/model_executor/model_loader/base_loader.py'] = profile['w8a16_overlay']['base_loader_after_sha256']
        for row in _contracts.V16_DENSE_FP8['targets']: hashes[package + row['path']] = row['after_sha256']
        for source, installed in [('scripts/validate_ablit_artifacts.py', 'ablit_artifacts.py'),
                                  ('overlays/trunk_w8a16.py', 'trunk_w8a16.py'), ('overlays/instanttensor_audit.py', 'instanttensor_audit.py')]:
            hashes[package + 'vllm/model_executor/layers/quantization/' + installed] = fleet.sha_file(ROOT/'recipe'/source)
        hashes[package + 'exllamav3_ext.cpython-312-aarch64-linux-gnu.so'] = '0eeb983b09dfe33451b8bc7625174320529b986a73f60b58bf4ebfa9f43ad9c0'
        original_read, original_exists = Path.read_text, Path.exists
        def read(path, *a, **kw):
            if str(path) in files: return json.dumps(files[str(path)])
            if str(path).startswith('/recipe/'):
                return original_read(ROOT / str(path).lstrip('/'), *a, **kw)
            return original_read(path, *a, **kw)
        def sha(path):
            if str(path).startswith('/recipe/'): return fleet.sha_file(ROOT / str(path).lstrip('/'))
            return hashes[str(path)]
        def exists(path):
            if str(path).startswith(('/opt/', package)): return False
            return original_exists(path)
        env = dict(self.values, NODE_RANK=str(rank), VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA='0',
                   GLM53_DENSE_FP8='off', EXL3_FAT_GROUPED='0')
        console = io.StringIO()
        with patch.dict(os.environ, env, clear=True), patch.object(Path, 'read_text', read), \
             patch.object(Path, 'exists', exists), patch.object(verify_stock, 'sha', side_effect=sha), redirect_stdout(console):
            verify_stock.main()
        projected = json.loads(console.getvalue())
        self.native_records[rank] = projected
        if self.historical:
            return dict(status='PASS', rank=rank, ablit=0, ablation=ablation, loaders=native_loaders, filename=SECRET)
        if self.fault == 'missing-loaders' and rank == 2: projected.pop('loaders', None)
        return projected

    def remote(self, values, rank, argv, **kwargs):
        self.calls.append((rank, tuple(argv[:3])))
        if argv[:3] == ['docker', 'container', 'inspect']:
            doc = [self.items[rank]]
        elif argv[:2] == ['docker', 'logs']:
            return subprocess.CompletedProcess(argv, 0, self.logs[rank], '')
        elif '/recipe/scripts/verify_stock.py' in argv:
            doc = self.stock_response(rank)
        elif '/recipe/scripts/apply_base_pipeline.py' in argv:
            doc = dict(state='ALREADY_APPLIED', target_set_sha256='e675e27d5d28fa8d864b689a8b2faa04bf728b5ab6ef99ee4034f379ef6821ec')
        elif argv == fleet.cgroup_argv(100 + rank):
            doc = dict(memory_max=str(fleet.MEMORY), swap_max='1' if rank == 2 and self.fault == 'swap' else '0',
                swap_current=0, events={k: '0' for k in ['low', 'high', 'max', 'oom', 'oom_kill', 'oom_group_kill']})
        elif argv == fleet.b45_identity_argv(self.manifest['containers'][rank]['container_id']):
            doc = dict(modules=fleet.B45_MODULES, pth_sha256=fleet.B45_PTH_SHA256,
                kda_original_sha256=fleet.B45_KDA_ORIGINAL_SHA256, b45_out_entries=['graphs/activation-123.jsonl'],
                capture_receipts=1 if rank == 2 and self.fault == 'capture' else 16,
                capture_dots_intact=True, serving_graph_dumps=8)
        elif argv == fleet.runtime_identity_argv(self.manifest['containers'][rank]['container_id'])[0]:
            doc = self.configs[rank]
        else:
            raise AssertionError(('unimplemented external IO', rank, argv))
        text = json.dumps(doc)
        fleet.diagnostics.retain(text)
        return subprocess.CompletedProcess(argv, 0, text, '')

    def http(self, request, **kwargs):
        url = request if isinstance(request, str) else request.full_url
        if url.endswith('/health'): return Response({})
        if url.endswith('/v1/models'): return Response({'data': [{'id': 'glm-5.3-flash'}]})
        payload = json.loads(request.data)
        if payload.get('stream'):
            choices = [{'choices': [{'delta': {'content': text}}]} for text in ('first', 'last')]
            choices += [dict(choices=[], usage=dict(completion_tokens=2 if self.fault == 'focused' else 400))]
            return Response((''.join('data: ' + json.dumps(row) + '\n' for row in choices) + 'data: [DONE]\n').encode())
        if payload['max_tokens'] == 8:
            return Response({'choices': [{'message': {'content': 'wrong' if self.fault == 'arithmetic' else '323'}}]})
        return Response(dict(model='glm-5.3-flash', usage=dict(prompt_tokens=32768 if self.fault == 'long-context' else 40000,
            completion_tokens=40), choices=[dict(finish_reason='stop', message=dict(content=long_context_witness.CODE_WORD))]))

    def process(self, argv, **kwargs):
        if argv[0] == 'ssh':
            command = shlex.split(argv[-1]); count = next(self.counter)
            counters = {hca: {name: count for name in focused_witness.COUNTERS} for hca in command[-2:]}
            return subprocess.CompletedProcess(argv, 0, json.dumps(counters), '')
        name = Path(argv[1]).name
        module = {'focused_witness.py': focused_witness, 'long_context_witness.py': long_context_witness}[name]
        out, err = io.StringIO(), io.StringIO()
        with patch.object(sys, 'argv', argv[1:]), redirect_stdout(out), redirect_stderr(err):
            code = module.main()
        if code == 0: self.witnesses[name] = json.loads(out.getvalue())
        return subprocess.CompletedProcess(argv, code, out.getvalue(), err.getvalue())

    def run(self):
        cli = ['fleetctl', 'verify', '--env-file', str(self.env), '--manifest', str(self.manifest_path),
               '--output', str(self.root/'verify.json'), '--log-output', str(self.root/'rank0.log')]
        output, errors = io.StringIO(), io.StringIO()
        with ExitStack() as stack:
            for name in ('remote', 'validate_container_contract', 'container_argv'):
                stack.enter_context(patch.object(fleet, name, getattr(fleet, name)))
            fleet.remote = self.remote
            resource_scope.install(fleet)
            stack.enter_context(patch.object(sys, 'argv', cli))
            stack.enter_context(patch.object(fleet.subprocess, 'run', side_effect=self.process))
            stack.enter_context(patch.object(fleet.urllib.request, 'urlopen', side_effect=self.http))
            stack.enter_context(patch.object(focused_witness.time, 'monotonic_ns', side_effect=itertools.count(0, 10**9)))
            stack.enter_context(redirect_stdout(output)); stack.enter_context(redirect_stderr(errors))
            code = fleet.main()
        self.console = output.getvalue() + errors.getvalue()
        self.receipt = json.loads((self.root/'verify.json').read_text())
        return code


class HealthyVerifyTests(unittest.TestCase):
    def test_three_rank_verify_through_real_stock_and_every_gate(self):
        for historical in (False, True):
            with self.subTest(historical=historical), tempfile.TemporaryDirectory() as directory:
                fixture = HealthyFleet(Path(directory), historical=historical)
                self.assertEqual(fixture.run(), 0, fixture.console)
                doc = fixture.receipt
                self.assertEqual(doc['status'], 'VERIFY_PASS')
                self.assertEqual(len(doc['load']), 7)
                for key in ('load', 'v14_identity', 'v16_identity'):
                    self.assertTrue(all(doc[key].values()), doc[key])
                self.assertEqual(len(doc['runtime_identity']), 3)
                self.assertEqual(len(doc['image_and_safety']), 3)
                self.assertEqual(doc['production_stock']['status'], 'PASS')
                self.assertTrue(doc['focused_witness']['pass'] and doc['long_context_witness']['pass'])
                self.assertEqual(fixture.witnesses['focused_witness.py']['all_three_ranks_observed'],
                                 {'rank0': True, 'rank1': True, 'rank2': True})
                self.assertEqual(len(fixture.native_records), 3)
                self.assertNotIn(SECRET, json.dumps(doc) + fixture.console + (Path(directory)/'rank0.log').read_text())
                # Real downstream qualification reader, including the receipt hash.
                qualify_runtime.verified(Path(directory)/'verify.json', fleet.sha_file(fixture.manifest_path), fleet)

    def test_verify_refuses_lost_facts_and_real_gate_failures(self):
        faults = {'incomplete': 'AssertionError', 'missing-loaders': 'missing loader evidence',
                  'missing-census': 'header census missing or malformed', 'malformed-census': 'header census missing or malformed',
                  'census-drift': 'load/graph/cadence receipt gate failed', 'capture': 'cadence capture evidence missing',
                  'swap': 'safety/image gate failed', 'v14': 'v1.4 identity gate failed', 'v16': 'v1.6 option identity gate failed',
                  'arithmetic': 'arithmetic gate failed', 'focused': 'focused witness failed', 'long-context': 'long-context witness failed'}
        for fault, reason in faults.items():
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                fixture = HealthyFleet(Path(directory), fault=fault)
                self.assertEqual(fixture.run(), 9, fixture.console)
                self.assertEqual(fixture.receipt['status'], 'VERIFY_REFUSED')
                self.assertNotIn(SECRET, json.dumps(fixture.receipt) + fixture.console)
                self.assertIn(reason, (Path(directory)/fixture.receipt['private_diagnostic']).read_text())


if __name__ == '__main__':
    unittest.main()
