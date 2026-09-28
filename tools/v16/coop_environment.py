#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GPU-free import, fixture-shape and stage-8 checks inside a disposable image."""
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True


def main():
    sys.path[:0] = ['/recipe/scripts', '/helpers', '/src/recipe/overlays/v16/coop/source']
    from _coop_qualification import need, read, sha
    from _atomic import canonical
    image = read(Path('/recipe/config/operator-image.json'))
    cid = Path('/work/container.cid').read_text().strip()
    receipt = {'schema_version': 2, 'manifest_digest': image['manifest_digest'],
               'config_digest': image['config_digest'], 'verification': 'host-observed-inspect-bound-create',
               'container_id': cid, 'rank': 0, 'preflight_sha256': sha(Path('/campaign/plan.json')),
               'recipe_manifest_sha256': sha(Path('/recipe/SHA256SUMS'))}
    receipt['payload_sha256'] = hashlib.sha256(canonical(receipt)).hexdigest()
    Path('/work/image-receipt.json').write_bytes(canonical(receipt))
    vllm = Path(importlib.util.find_spec('vllm').origin).parent
    subprocess.run(['python3', '-B', '-S', '/recipe/scripts/apply_base_pipeline.py', '--vllm-root', str(vllm),
                    '--source-root', '/sources/fly', '--asset-root', '/opt/glm53',
                    '--contract', '/recipe/config/patch-contract.json', '--image-receipt', '/work/image-receipt.json',
                    '--apply'], check=True)
    os.environ.update(GLM53_COOP_QUALIFICATION='1', GLM53_COOP_GEOMETRY='0',
                      GLM53_COOP_BUNDLE='/campaign/raw-bundle')
    import test_cuda_integration as gate
    import test_exl3_overlay
    import torch
    import exllamav3_ext
    from safetensors import safe_open
    ctypes.CDLL('/campaign/raw-bundle/cooperative_moe.so')
    ldd = subprocess.check_output(['ldd', '/campaign/raw-bundle/cooperative_moe.so'], text=True)
    need('libcudart.so.13 =>' in ldd and 'not found' not in ldd, 'shared cudart linkage')
    target = read(Path('/recipe/config/checkpoint-contract.json'))['target']
    snapshot = Path('/root/.cache/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw/snapshots') / target['revision']
    index = read(snapshot / 'model.safetensors.index.json')['weight_map']
    # Check every real96 EP range without allocating a CUDA tensor.
    grouped = {}
    for expert in range(288):
        for proj in ('gate_proj', 'up_proj', 'down_proj'):
            for suffix in ('trellis', 'suh', 'svh', 'mcg'):
                name = f'model.language_model.layers.3.mlp.experts.{expert}.{proj}.{suffix}'
                grouped.setdefault(index[name], []).append((name, proj, suffix))
    for name, tensors in grouped.items():
        with safe_open(snapshot / name, framework='pt', device='cpu') as handle:
            for tensor, proj, suffix in tensors:
                shape = handle.get_slice(tensor).get_shape()
                need(shape and all(n > 0 for n in shape), 'empty checkpoint tensor')
                if suffix == 'trellis':
                    need(shape[:2] == ([128, 256] if proj == 'down_proj' else [256, 128]), 'wrong real-weight shape')
    help_text = subprocess.check_output(['/sanitizer/compute-sanitizer', '--help'], text=True)
    need('--dump-kernel-launches' in help_text and '--kernel-name' in help_text, 'sanitizer lacks launch evidence support')
    report = {'status': 'PASS', 'scope': 'CPU stage-8 imports and fixture shapes; no GPU correctness',
              'exl3_sha256': gate.EXL3_SHA256, 'fatpath_sha256': gate.FATPATH_SHA256,
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'ldd': ldd,
              'sanitizer': subprocess.check_output(['/sanitizer/compute-sanitizer', '--version'], text=True),
              'nvcc': subprocess.check_output(['/usr/local/cuda/bin/nvcc', '--version'], text=True),
              'gcc': subprocess.check_output(['gcc', '--version'], text=True)}
    if len(sys.argv) > 1:
        gpu = torch.cuda.get_device_properties(0)
        report['gpu'] = {'name': gpu.name, 'capability': [gpu.major, gpu.minor],
                         'driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version,uuid', '--format=csv,noheader'], text=True).strip()}
        need(gpu.name == 'NVIDIA GB10' and (gpu.major, gpu.minor) == (12, 1), 'qualification requires GB10')
    Path('/work/environment.json').write_text(json.dumps(report, indent=2) + '\n')
    for key in ('GLM53_COOP_QUALIFICATION', 'GLM53_COOP_GEOMETRY'):
        os.environ.pop(key)
    if len(sys.argv) > 1:
        # exec a clean Python process: no imported candidate adapter survives
        # into a different geometry or the measured-policy production check.
        os.execvp(sys.argv[1], sys.argv[1:])
    print(json.dumps(report))


if __name__ == '__main__':
    main()
