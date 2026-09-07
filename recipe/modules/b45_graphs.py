"""Request-scoped original B4; one BF16 and one INT8 bank, each at physical M4 and M8.

The B4 quantization module is copied byte-for-byte. Only its phase predicate is
additionally gated here. FULL graph execution selects an actual captured object;
the Python selection flag is used during capture and eager execution only.
"""
import contextlib
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import time

SITE = Path('/usr/local/lib/python3.12/dist-packages')
OUT = Path('/tmp/b45/graphs')
EXPECTED = {
 'vllm/v1/worker/gpu/model_runner.py': 'f84255d75435e84f44972d3fd25e53447f9d4d2edd8bff4f8c19dfb793448415',
 'vllm/v1/worker/gpu/cudagraph_utils.py': 'c183937e6eb5b9c28c79d98fb4c64f562e7649d5f6d65743e6640b2f378ecf9f',
 'vllm/model_executor/layers/quantization/kda_mixed_output_blocks.py': 'db6d60f0ac99d3cc23d5d0b6194a557779b82f2b34132c4dba11f28098fda61f',
}
PATH = 'bf16'
CURRENT = None
CAPTURE = None
REQUESTS = {}
STEP = 0
ACTIVE = False
_GATED = False
_DONE = set()
BANKS = ('bf16_0', 'int8_0')
BASE = 'bf16_0'


def capture_order():
    order = json.loads(os.environ['B4_CAPTURE_ORDER'])
    if len(order) != len(BANKS) or set(order) != set(BANKS):
        raise RuntimeError('B4 invalid presealed capture order')
    return order


def normalize(flag):
    aliases = {'bf16': BASE, 'int8': 'int8_0'}
    flag = aliases.get(flag, flag)
    if flag not in BANKS:
        raise RuntimeError('Invalid B4 request selection: ' + repr(flag))
    return flag


def write(kind, row):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / (kind + '-' + str(os.getpid()) + '.jsonl'), 'a') as f:
        f.write(json.dumps({'rank': os.getenv('NODE_RANK'), 'time': time.time(), **row}) + '\n')


@contextlib.contextmanager
def select(path):
    global PATH
    old = PATH
    PATH = path
    try:
        yield
    finally:
        PATH = old


def patch_kda(mod):
    pure = mod._pure_decode
    mod._pure_decode = lambda layer: PATH == 'int8' and pure(layer)
    apply = mod.KdaMixedOutputBlocksMethod.apply

    def observed(self, layer, x, bias=None):
        if CAPTURE is not None:
            m = x.numel() // mod.EXPECTED_INPUT_SIZE
            use = m <= mod.EXACT_M_MAX and getattr(layer, 'kda_qkv_shadow_enabled', False) and mod._pure_decode(layer)
            CAPTURE.append({'prefix': layer.prefix, 'm': m, 'path': 'int8' if use else ('bf16' if m <= 8 else 'parent'),
                            'weight_ptr': layer.kda_shape_static_exact_weight.data_ptr(),
                            'control_ptr': layer.kda_qkv_shadow_control_weight.data_ptr()})
        return apply(self, layer, x, bias)
    mod.KdaMixedOutputBlocksMethod.apply = observed


def descriptor(desc):
    return {'mode': desc.cg_mode.name, 'tokens': desc.num_tokens, 'requests': desc.num_reqs,
            'uniform': desc.uniform_token_count, 'max_query': desc.max_query_len}


def patch_graphs(mod):
    import torch
    Base, Model = mod.CudaGraphManager, mod.ModelCudaGraphManager
    original_capture, original_profile = Base.capture, Base.profile_memory

    @torch.inference_mode()
    def capture(self, create_forward_fn, progress_bar_desc='Capturing CUDA graphs'):
        global CAPTURE
        if not isinstance(self, Model):
            return original_capture(self, create_forward_fn, progress_bar_desc)
        if set(self._capture_descs) != {mod.CUDAGraphMode.FULL}:
            raise RuntimeError('B4 requires the frozen FULL_DECODE_ONLY graph family')
        descs = self._capture_descs[mod.CUDAGraphMode.FULL]
        small = [d for d in descs if d.num_tokens <= 8]
        if len(small) != 2 or {(d.num_tokens, d.uniform_token_count, d.num_reqs) for d in small} != {(4, 4, 1), (8, 8, 1)}:
            raise RuntimeError('B4 graph descriptor drift')
        self._b4_banks = {bank: (self.graphs if bank == BASE else {}) for bank in BANKS}
        self._b4_receipts = {}
        phase = 'profile' if progress_bar_desc.startswith('Profiling') else 'serving'
        with mod.graph_capture(device=self.device):
            families = [(BASE, [d for d in descs if d.num_tokens > 8])]
            families += [(bank, small) for bank in capture_order()]
            for slot, (bank, family) in enumerate(families):
                with select('int8' if bank.startswith('int8_') else 'bf16'):
                    for desc in family:
                        forward = create_forward_fn(desc, warmup=True)
                        forward(mod.CUDAGraphMode.NONE)
                        forward = create_forward_fn(desc, warmup=False)
                        graph = torch.cuda.CUDAGraph(keep_graph=True)
                        mod.get_offloader().sync_prev_onload()
                        mod.set_graph_pool_id(self.pool if self.pool is not None else mod.current_platform.graph_pool_handle())
                        CAPTURE = []
                        try:
                            with torch.cuda.graph(graph, self.pool):
                                forward(mod.CUDAGraphMode.NONE)
                                mod.get_offloader().join_after_forward()
                            graph.instantiate()
                            calls = CAPTURE
                        finally:
                            CAPTURE = None
                        want = 'int8' if bank.startswith('int8_') else ('bf16' if desc.num_tokens <= 8 else 'parent')
                        if len(calls) != 34 or len({x['prefix'] for x in calls}) != 34 or any(x['path'] != want for x in calls):
                            raise RuntimeError('B4 capture did not execute all 34 intended projection paths: ' + repr(calls))
                        self._b4_banks[bank][desc] = graph
                        mod.compilation_counter.num_cudagraph_captured += 1
                        OUT.mkdir(parents=True, exist_ok=True)
                        dot = OUT / f'graph-{phase}-{os.getpid()}-{bank}-{desc.num_tokens}.dot'
                        graph.debug_dump(str(dot))
                        if not dot.exists() or dot.stat().st_size == 0:
                            raise RuntimeError('B4 missing actual CUDA graph node dump')
                        receipt = {'phase': phase, 'bank': bank, 'capture_slot': slot, 'capture_order': capture_order(), 'desc': descriptor(desc), 'object': id(graph),
                                   'dot': dot.name, 'dot_sha256': hashlib.sha256(dot.read_bytes()).hexdigest(), 'calls': calls}
                        self._b4_receipts[(bank, desc)] = receipt
                        write('captures', receipt)
        for desc in small:
            refs = [self._b4_receipts[(b, desc)] for b in BANKS]
            if len({r['object'] for r in refs}) != len(BANKS):
                raise RuntimeError('B4 graph alias')
            if any([(c['prefix'], c['weight_ptr'], c['control_ptr']) for c in r['calls']] !=
                   [(c['prefix'], c['weight_ptr'], c['control_ptr']) for c in refs[0]['calls']] for r in refs[1:]):
                raise RuntimeError('B4 retained BF16 weight pointers changed across capture')
        self._graphs_captured = True

    def profile(self, *args, **kwargs):
        try:
            return original_profile(self, *args, **kwargs)
        finally:
            if isinstance(self, Model):
                # Original profile drops the reference graphs; drop all additional
                # banks too before the throwaway graph pool and KV cache disappear.
                for bank in getattr(self, '_b4_banks', {}).values():
                    bank.clear()
                self._b4_banks = {}
                self._b4_receipts = {}
    Base.capture, Base.profile_memory = capture, profile
    run = Model.run_fullgraph

    def run_fullgraph(self, desc):
        bank = CURRENT['selection'] if CURRENT is not None else BASE
        selected = bank if desc in self._b4_banks.get(bank, {}) else BASE
        original = self.graphs
        self.graphs = self._b4_banks[selected]
        try:
            if CURRENT is not None:
                rec = self._b4_receipts[(selected, desc)]
                CURRENT['graph'] = {'bank': selected, 'object': id(self.graphs[desc]), 'desc': descriptor(desc),
                                    'capture_dot_sha256': rec['dot_sha256'], 'projection': rec['calls'][0]['path']}
            return run(self, desc)
        finally:
            self.graphs = original
    Model.run_fullgraph = run_fullgraph


def request_selection(output):
    for rid in output.finished_req_ids or ():
        REQUESTS.pop(rid, None)
    for req in output.scheduled_new_reqs:
        extra = getattr(req.sampling_params, 'extra_args', None) or {}
        flag = normalize(extra.get('b4', 'int8_0'))
        REQUESTS[req.req_id] = flag
    ids = list(output.num_scheduled_tokens)
    choices = {REQUESTS.get(r, BASE) for r in ids}
    return (next(iter(choices)) if len(choices) == 1 else BASE), {r: REQUESTS.get(r, BASE) for r in ids}


def patch_runner(mod):
    R = mod.GPUModelRunner
    execute, sample, capture = R.execute_model, R.sample_tokens, R.capture_model

    def execute_model(self, output, *args, **kwargs):
        global CURRENT, STEP
        dummy = kwargs.get('dummy_run', False) or (len(args) >= 2 and bool(args[1]))
        if dummy or not ACTIVE:
            return execute(self, output, *args, **kwargs)
        choice, choices = request_selection(output)
        STEP += 1
        CURRENT = {'step': STEP, 'selection': choice, 'requests': choices, 'scheduled': dict(output.num_scheduled_tokens), 'start': time.time()}
        with select('int8' if choice.startswith('int8_') else 'bf16'):
            try:
                result = execute(self, output, *args, **kwargs)
                if self.execute_model_state is not None:
                    ib = self.execute_model_state.input_batch
                    CURRENT['tokens'] = int(ib.num_tokens)
                    CURRENT['padded_tokens'] = int(ib.num_tokens_after_padding)
                return result
            except BaseException:
                write('errors', CURRENT)
                raise

    def sample_tokens(self, *args, **kwargs):
        global CURRENT
        rec = CURRENT
        try:
            return sample(self, *args, **kwargs)
        finally:
            if rec is not None:
                rec['end'] = time.time()
                write('steps', rec)
                CURRENT = None

    def capture_model(self):
        global ACTIVE
        out = capture(self)
        ACTIVE = True
        write('activation', {'active': True, 'hash_gate': _GATED, 'banks': {k: len(v) for k,v in self.cudagraph_manager._b4_banks.items()}})
        return out
    R.execute_model, R.sample_tokens, R.capture_model = execute_model, sample_tokens, capture_model


TARGETS = {'vllm.v1.worker.gpu.cudagraph_utils': patch_graphs,
           'vllm.v1.worker.gpu.model_runner': patch_runner,
           'vllm.model_executor.layers.quantization.kda_mixed_output_blocks': patch_kda}

