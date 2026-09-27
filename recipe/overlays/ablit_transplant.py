"""Verified TP3 o_proj transplant immediately before Cadence W8A16 packing."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import torch

LAYER = re.compile(r'(?:^|\.)layers\.(\d+)\.self_attn\.o_proj$')


def tensor_sha(tensor):
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def partition(full, rank, world_size):
    if type(rank) is not int or type(world_size) is not int or world_size != 3 or rank not in range(3):
        raise RuntimeError('transplant requires initialized TP3 rank 0, 1 or 2')
    if full.dtype != torch.bfloat16 or full.ndim != 2 or list(full.shape) not in ([4096, 8192], [4096, 16384]):
        raise RuntimeError('donor full tensor geometry/dtype drift')
    width = full.shape[1]
    padded_width = width // 64 * 66
    padded = torch.nn.functional.pad(full, (0, padded_width - width))
    local = padded_width // 3
    return padded[:, rank * local:(rank + 1) * local].contiguous()


def load_slice(root, row, rank):
    full = torch.from_file(str(root / row['file']), shared=False,
                           size=row['bytes'] // 2, dtype=torch.bfloat16).reshape(row['shape'])
    return partition(full, rank, 3)


def receipt_preflight():
    root = Path('/evidence')
    path = root / 'ablit-receipt.json'
    if not root.is_dir() or path.exists() or path.is_symlink():
        raise RuntimeError('missing evidence directory or stale ablation receipt')
    fd, temporary = tempfile.mkstemp(prefix='.ablit-write-probe-', dir=root)
    os.close(fd)
    Path(temporary).unlink()


def save_receipt(value):
    path = Path('/evidence/ablit-receipt.json')
    if path.exists():
        raise RuntimeError('duplicate ablation finalization receipt')
    raw = (json.dumps(value, sort_keys=True) + '\n').encode()
    fd, temporary = tempfile.mkstemp(prefix='.ablit-receipt-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    print('JSPARK3_ABLIT_RECEIPT=' + raw.decode().strip(), flush=True)


def finalize_with_ablit(model):
    from . import trunk_w8a16 as trunk
    mode = os.environ.get('ABLIT')
    if mode not in ('0', '1'):
        raise RuntimeError('ABLIT must be explicitly 0 or 1')
    if getattr(model, '_jspark3_ablit_finalized', False):
        raise RuntimeError('duplicate ablation application/finalization')
    configured_rank = os.environ.get('NODE_RANK')
    if configured_rank not in ('0', '1', '2'):
        raise RuntimeError('missing or invalid declared NODE_RANK')
    receipt_preflight()
    if mode == '0':
        trunk.finalize_trunk_w8a16(model)
        model._jspark3_ablit_finalized = True
        save_receipt({'schema_version': 1, 'ablit': 0, 'state': 'DISABLED',
                      'rank': int(os.environ['NODE_RANK']), 'applied_layers': []})
        return
    if (os.environ.get('ABLIT_METHOD') != 'transplant' or
            os.environ.get('ABLIT_LAYERS') != '15-45' or os.environ.get('ABLIT_INCLUDE_MTP') != '1'):
        raise RuntimeError('only the pinned transplant method/range is supported')
    from .ablit_artifacts import validate
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    try:
        rank = get_tensor_model_parallel_rank()
        world = get_tensor_model_parallel_world_size()
    except Exception as exc:
        raise RuntimeError('distributed state unavailable for transplant') from exc
    if world != 3 or rank not in range(3) or str(rank) != os.environ['NODE_RANK']:
        raise RuntimeError('distributed and configured TP3 ranks disagree')
    root = Path('/ablit')
    manifest_sha = os.environ['JSPARK_ABLIT_MANIFEST_SHA256']
    manifest = validate(root, manifest_sha)
    rows = {row['layer']: row for row in manifest['tensors']}
    modules = {}
    for name, module in model.named_modules():
        match = LAYER.search(name)
        if match:
            layer = int(match.group(1))
            if layer in modules:
                raise RuntimeError('duplicate target layer')
            modules[layer] = (name, module)
    if set(modules) != set(range(45)):
        raise RuntimeError('target layer census differs from 0..44; MTP needs a separately qualified hook')
    # DFlash2 is the separate drafter. This target construction has no MTP L45.
    torch.cuda.synchronize()
    snapshots = {name: (parameter.data_ptr(), parameter._version, tuple(parameter.shape), parameter.dtype)
                 for name, parameter in model.named_parameters()}
    selected_names = {modules[layer][0] + '.weight' for layer in range(15, 45)}
    # Validate all destinations before the first copy; early packing is an error.
    for layer in range(15, 45):
        name, module = modules[layer]
        weight = getattr(module, 'weight', None)
        expected = (4096, rows[layer]['shape'][1] // 64 * 22)
        if (weight is None or tuple(weight.shape) != expected or weight.dtype != torch.bfloat16 or
                weight.device.type != 'cuda' or
                not weight.is_contiguous() or weight.storage_offset() != 0 or
                weight.untyped_storage().nbytes() != weight.numel() * weight.element_size() or
                type(getattr(module, 'quant_method', None)).__name__ != 'UnquantizedLinearMethod' or
                getattr(module, '_jspark3_ablit_applied', False)):
            raise RuntimeError(f'premature packing, duplicate application, or destination drift: {name}')
        if rank == 2 and torch.count_nonzero(weight[:, -(rows[layer]['shape'][1] // 32):]).item() != 0:
            raise RuntimeError(f'stock TP3 end-padding geometry mismatch: {name}')
    anchors = {layer: tensor_sha(modules[layer][1].weight) for layer in (0, 14)}
    records = {}
    with torch.no_grad():
        for layer in range(15, 45):
            name, module = modules[layer]
            original = tensor_sha(module.weight)
            reference = load_slice(root, rows[layer], rank)
            expected_hash = tensor_sha(reference)
            module.weight.copy_(reference.to(device=module.weight.device))
            actual_hash = tensor_sha(module.weight)
            if actual_hash != expected_hash:
                raise RuntimeError(f'loaded donor slice mismatch: L{layer}')
            module._jspark3_ablit_applied = True
            records[id(module)] = {'layer': layer, 'name': name, 'original_sha256': original,
                                   'donor_sha256': rows[layer]['sha256'], 'slice_sha256': actual_hash,
                                   'original_equals_slice': original == actual_hash, 'packed_verified': False}
    changed = {name for name, parameter in model.named_parameters()
               if snapshots[name] != (parameter.data_ptr(), parameter._version, tuple(parameter.shape), parameter.dtype)}
    if changed != selected_names:
        raise RuntimeError('transplant changed an unexpected parameter or missed a target')
    if anchors != {layer: tensor_sha(modules[layer][1].weight) for layer in anchors}:
        raise RuntimeError('untouched anchor bytes changed')
    original_pack = trunk.pack_weight

    def checked_pack(module, weight, suffix=''):
        record = records.get(id(module))
        if record is None:
            return original_pack(module, weight, suffix)
        if record['packed_verified'] or tensor_sha(weight) != record['slice_sha256']:
            raise RuntimeError('duplicate pack or transplanted tensor overwritten before packing')
        reference = load_slice(root, rows[record['layer']], rank).to(weight.device)
        reference_module = torch.nn.Module()
        reference_receipt = original_pack(reference_module, reference, suffix)
        actual_receipt = original_pack(module, weight, suffix)
        names = trunk._names(suffix)
        for field in ('qweight', 'scales', 'empty'):
            if not torch.equal(getattr(module, names[field]), getattr(reference_module, names[field])):
                raise RuntimeError(f'packed donor differs from reference packer: {field}')
        generator = torch.Generator(device=weight.device).manual_seed(20260920 + record['layer'] * 3 + rank)
        x = torch.randn((2, weight.shape[1]), dtype=torch.bfloat16, device=weight.device, generator=generator)
        method = trunk.TrunkW8A16Method(actual_receipt['k'], actual_receipt['n'])
        actual_output = method.apply(module, x)
        reference_output = method.apply(reference_module, x)
        bf16_reference = x.float() @ reference.float().T
        additional_error = float((actual_output.float() - reference_output.float()).abs().max().item())
        relative_l2 = float(((actual_output.float() - bf16_reference).norm() /
                             bf16_reference.norm().clamp_min(1e-12)).item())
        if not torch.isfinite(actual_output).all() or additional_error != 0.0 or relative_l2 > 0.05:
            raise RuntimeError('packed donor kernel differs from its control or exceeds frozen 5% relative L2 bound')
        residual = reference_receipt['dequantized'].float() - reference.T.float()
        record.update(packed_verified=True, packed_sha256=tensor_sha(getattr(module, names['qweight'])),
                      scales_sha256=tensor_sha(getattr(module, names['scales'])),
                      reference_max_abs_error=float(residual.abs().max().item()),
                      reference_rmse=float(residual.square().mean().sqrt().item()),
                      additional_packing_error=additional_error,
                      kernel_reference_relative_l2=relative_l2,
                      kernel_reference_rows=2, kernel_reference_limit=0.05)
        return actual_receipt

    trunk.pack_weight = checked_pack
    try:
        trunk.finalize_trunk_w8a16(model)
    finally:
        trunk.pack_weight = original_pack
    if len(records) != 30 or not all(row['packed_verified'] for row in records.values()):
        raise RuntimeError('incomplete packed-donor verification')
    model._jspark3_ablit_finalized = True
    save_receipt({'schema_version': 1, 'ablit': 1, 'state': 'APPLIED_AND_PACKED', 'rank': rank,
                  'manifest_sha256': manifest_sha, 'donor_revision': manifest['revision'],
                  'applied_layers': list(range(15, 45)), 'anchor_hashes': anchors,
                  'untouched_parameter_count': len(snapshots) - len(selected_names),
                  'untouched_parameter_versions_and_storage': True,
                  'mtp': {'layer': 45, 'artifact_verified': True, 'active': False,
                          'reason': 'DFlash2 profile; target instantiates only layers 0..44'},
                  'layers': sorted(records.values(), key=lambda row: row['layer'])})
