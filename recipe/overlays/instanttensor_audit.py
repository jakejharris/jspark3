"""Read-only tensor census around the pinned InstantTensor iterator."""
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import stat
import struct
import time

_completed_target_files = ()


def descriptor_state():
    value = {'nofile': list(resource.getrlimit(resource.RLIMIT_NOFILE))}
    try:
        value['open_fd_count'] = len(os.listdir('/proc/self/fd'))
    except OSError as error:
        value['count_error'] = {'errno': error.errno, 'message': str(error)}
    return value


def _prepare_loader_memory(files, tensor_sizes, kind):
    """Make the pinned loader's real free-memory predicate usable on GB10 UMA."""
    import torch
    from instanttensor import _impl
    from vllm.distributed import get_world_group

    if any(key.startswith('INSTANTTENSOR_') for key in os.environ):
        raise RuntimeError('undeclared InstantTensor override')
    group = get_world_group()
    if descriptor_state()['nofile'] != [65536, 65536]:
        raise RuntimeError('worker nofile contract drift')
    if group.world_size != 3 or group.rank != int(os.environ['NODE_RANK']):
        raise RuntimeError('loader reclamation requires the declared TP3 world')
    backend = _impl.select_backend(_impl.default_backend)
    if backend not in (_impl.Backend.URING, _impl.Backend.AIO):
        raise RuntimeError('loader reclamation requires pinned direct I/O backend')
    # These are the pinned 0.2.0 defaults for both direct-I/O backends. Compute
    # the unshrunk I/O bound; do not reproduce its low-free-memory downscaling.
    io_buffer = _impl.required_buffer_size_for_io(8 * 1024**2, max(512 // 3, 3), 3)
    tensor_buffer = _impl.recommended_buffer_size_for_tensors(tensor_sizes)
    buffer_bytes = max(io_buffer, tensor_buffer)
    zone_high = sum(int(n) for n in re.findall(
        r'^\s+high\s+(\d+)\s*$', Path('/proc/zoneinfo').read_text(), re.M)) * os.sysconf('SC_PAGE_SIZE')
    if zone_high <= 0:
        raise RuntimeError('missing kernel high-watermark reserve')
    required_free = 2 * buffer_bytes + zone_high

    def sample():
        memory = {line.split(':')[0]: int(line.split()[1]) * 1024
                  for line in Path('/proc/meminfo').read_text().splitlines()
                  if line.endswith(' kB')}
        swap = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
        free, total = torch.cuda.mem_get_info()
        return {'cuda_free': free, 'cuda_total': total,
                'cuda_allocated': torch.cuda.memory_allocated(),
                'cuda_reserved': torch.cuda.memory_reserved(),
                'descriptors': descriptor_state(),
                'host': {k: memory[k] for k in ('MemFree', 'MemAvailable', 'Cached')},
                'pswpout': int(swap['pswpout'])}

    before = sample()
    torch.cuda.empty_cache()
    after_empty = sample()
    print('JSPARK3_LOADER_MEMORY_BEGIN=' + json.dumps({
        'rank': group.rank, 'kind': kind, 'backend': backend.name,
        'io_buffer_bound': io_buffer, 'tensor_buffer_bound': tensor_buffer,
        'required_cuda_free': required_free, 'before': before,
        'after_empty_cache': after_empty}, sort_keys=True), flush=True)
    if after_empty['host']['MemAvailable'] < required_free + 8 * 1024**3:
        raise RuntimeError('insufficient available memory plus host reserve for loader')
    parents = {
        'target': Path('/models') / 'Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw-25a44fdb',
        'draft': Path('/models') / 'incoai--GLM-5.3-Flash-DFlash2-dc77ff1c-native'}
    advice_files = [(filename, kind) for filename in sorted(files)]
    if kind == 'draft':
        # Only a successfully completed, owned target iterator can register
        # these files. GPU tensors no longer depend on their clean file cache.
        advice_files.extend((filename, 'target') for filename in _completed_target_files)
    advised = []
    current = after_empty
    started = time.monotonic()
    for filename, checkpoint in advice_files:
        if current['cuda_free'] >= required_free:
            break
        path = Path(filename).resolve(strict=True)
        if path.parent != parents[checkpoint] or path.suffix != '.safetensors':
            raise RuntimeError('loader cache advice escaped immutable checkpoint')
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            original = os.fstat(fd)
            if not stat.S_ISREG(original.st_mode):
                raise RuntimeError('loader cache advice requires regular checkpoint')
            for offset in range(0, original.st_size, 256 * 1024**2):
                if current['cuda_free'] >= required_free:
                    break
                if time.monotonic() - started > 30:
                    raise RuntimeError('bounded loader cache advice deadline exceeded')
                length = min(256 * 1024**2, original.st_size - offset)
                os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
                current = sample()
                advised.append({'file': path.name, 'checkpoint': checkpoint,
                                'offset': offset, 'length': length,
                                'cuda_free_after': current['cuda_free']})
            after = os.fstat(fd)
            if (after.st_ino, after.st_size, after.st_mtime_ns) != (
                    original.st_ino, original.st_size, original.st_mtime_ns):
                raise RuntimeError('checkpoint changed during loader cache advice')
        finally:
            os.close(fd)
    receipt = {'rank': group.rank, 'kind': kind, 'backend': backend.name,
               'io_buffer_bound': io_buffer, 'tensor_buffer_bound': tensor_buffer,
               'kernel_high_watermark_bytes': zone_high, 'required_cuda_free': required_free,
               'before': before, 'after_empty_cache': after_empty, 'after': current,
               'advised_ranges': advised, 'seconds': time.monotonic() - started,
               'completed_target_files_eligible': len(_completed_target_files) if kind == 'draft' else 0,
               'pass': current['cuda_free'] >= required_free}
    with (Path('/evidence') / ('loader-memory-' + kind + '.json')).open('x') as handle:
        json.dump(receipt, handle, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    print('JSPARK3_LOADER_MEMORY=' + json.dumps(receipt, sort_keys=True), flush=True)
    if not receipt['pass']:
        raise RuntimeError(f'exact checkpoint advice did not establish loader headroom: free={current["cuda_free"]}, required={required_free}')
    # The original library independently samples free memory and takes MIN over
    # its process group. Its allocation guard and copy=True remain unchanged.


def prepare_loader_memory(files, tensor_sizes, kind):
    """All ranks agree on preparation success before the loader's collectives."""
    import torch
    from vllm.distributed import get_world_group
    group = get_world_group()
    failure = None
    try:
        _prepare_loader_memory(files, tensor_sizes, kind)
    except Exception as error:
        failure = type(error).__name__ + ': ' + str(error)
    flag = torch.tensor([int(failure is None)], dtype=torch.int32, device='cpu')
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group)
    agreement = {'rank': group.rank, 'kind': kind, 'local_error': failure,
                 'all_ranks_pass': flag.item() == 1}
    with (Path('/evidence') / ('loader-memory-' + kind + '-agreement.json')).open('x') as handle:
        json.dump(agreement, handle, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    if not agreement['all_ranks_pass']:
        raise RuntimeError('TP3 loader memory preparation failed: ' + (failure or 'another rank failed'))


def audited_iterator(files, use_tqdm_on_load):
    global _completed_target_files
    from vllm.model_executor.model_loader.weight_utils import instanttensor_weights_iterator
    expected = {}
    headers = []
    tensor_sizes = []
    for filename in sorted(files):
        path = Path(filename)
        with path.open('rb') as handle:
            length = struct.unpack('<Q', handle.read(8))[0]
            if not 2 <= length <= 32 * 1024 * 1024:
                raise RuntimeError('invalid local safetensors header length')
            raw = handle.read(length)
        headers.append({'file': path.name, 'sha256': hashlib.sha256(raw).hexdigest()})
        rows = {k: v for k, v in json.loads(raw).items() if k != '__metadata__'}
        for name, row in sorted(rows.items(), key=lambda kv: kv[1]['data_offsets'][0]):
            if name in expected:
                raise RuntimeError('duplicate tensor in loading census')
            expected[name] = row['data_offsets'][1] - row['data_offsets'][0]
            tensor_sizes.append(expected[name])
    kind = 'target' if len(files) == 120 else 'draft' if len(files) == 1 else None
    if kind is None:
        raise RuntimeError('unexpected checkpoint shard census')
    if kind == 'target' and _completed_target_files:
        raise RuntimeError('target iterator already completed in this worker')
    started = time.monotonic()
    prepare_loader_memory(files, tensor_sizes, kind)
    observed = {}
    first_tensor_descriptors = None
    try:
        for name, tensor in instanttensor_weights_iterator(files, use_tqdm_on_load):
            if first_tensor_descriptors is None:
                first_tensor_descriptors = descriptor_state()
                print('JSPARK3_LOADER_FIRST_TENSOR=' + json.dumps({
                    'rank': int(os.environ['NODE_RANK']), 'kind': kind,
                    'descriptors': first_tensor_descriptors}, sort_keys=True), flush=True)
            size = tensor.numel() * tensor.element_size()
            if name in observed or expected.get(name) != size:
                raise RuntimeError('InstantTensor duplicate/unexpected/short tensor')
            observed[name] = size
            yield name, tensor
    except RuntimeError as error:
        print('JSPARK3_LOADER_FAILURE=' + json.dumps({
            'rank': int(os.environ['NODE_RANK']), 'kind': kind,
            'error': str(error), 'descriptors': descriptor_state()}, sort_keys=True), flush=True)
        raise
    if observed != expected:
        raise RuntimeError('InstantTensor incomplete checkpoint iteration')
    value = {'schema_version': 1, 'state': 'COMPLETE', 'loader': 'instanttensor',
             'rank': int(os.environ['NODE_RANK']), 'kind': kind,
             'shards': len(files), 'tensors': len(observed), 'bytes': sum(observed.values()),
             'ordered_headers_sha256': hashlib.sha256(json.dumps(headers, sort_keys=True).encode()).hexdigest(),
             'iterator_and_consumer_seconds': time.monotonic() - started,
             'copy_ownership': True}
    value['first_tensor_descriptors'] = first_tensor_descriptors
    path = Path('/evidence') / ('loader-' + kind + '.json')
    # Exclusive creation catches accidental second loading in the same rank.
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    if kind == 'target':
        _completed_target_files = tuple(sorted(files))
    print('JSPARK3_INSTANTTENSOR_COMPLETE=' + json.dumps(value, sort_keys=True), flush=True)
