#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GPU-free import, fixture-shape and stage-8 checks inside a disposable image."""
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'recipe/scripts'))
import _diagnostics as diagnostics


def output(command):
    proc = subprocess.run(command, text=True, capture_output=True)
    diagnostics.retain(proc.stdout + proc.stderr)
    proc.check_returncode()
    return proc.stdout


def main():
    # Protected child CLIs also create standalone diagnostic directories. Keep
    # those in the retained bind mount so container removal cannot erase them.
    private = Path('/work/private-protocol')
    private.mkdir(mode=0o700, exist_ok=True)
    os.environ['TMPDIR'] = str(private)
    sys.path[:0] = ['/recipe/scripts', '/helpers', '/src/recipe/overlays/v16/coop/source']
    from _coop_qualification import need, read, sha
    from _coop_checkpoint import authenticate, verify_shapes
    from _coop_bundle import verify_bundle, verify_selection
    from _atomic import canonical
    from _image_identity import verify_operator_record
    image = read(Path('/recipe/config/operator-image.json'))
    verify_operator_record(image)
    cid = Path('/work/container.cid').read_text().strip()
    need(re.fullmatch('[0-9a-f]{64}', cid), 'container identity malformed')
    receipt = {'schema_version': 2, 'manifest_digest': image['manifest_digest'],
               'config_digest': image['config_digest'], 'verification': 'host-observed-inspect-bound-create',
               'container_id': cid, 'rank': 0, 'preflight_sha256': sha(Path('/campaign/plan.json')),
               'recipe_manifest_sha256': sha(Path('/recipe/SHA256SUMS'))}
    receipt['payload_sha256'] = hashlib.sha256(canonical(receipt)).hexdigest()
    Path('/work/image-receipt.json').write_bytes(canonical(receipt))
    vllm = Path(importlib.util.find_spec('vllm').origin).parent
    diagnostics.run_private(['python3', '-B', '-S', '/recipe/scripts/apply_base_pipeline.py', '--vllm-root', str(vllm),
                    '--source-root', '/sources/fly', '--asset-root', '/opt/glm53',
                    '--contract', '/recipe/config/patch-contract.json', '--image-receipt', '/work/image-receipt.json',
                    '--apply'], check=True)
    coop = Path('/recipe/overlays/v16/coop')
    raw = Path('/campaign/raw-bundle')
    active = Path('/campaign/selected-bundle') if 'GLM53_COOP_BUNDLE=/campaign/selected-bundle' in sys.argv else raw
    verify_bundle(raw, coop, source_policy=True)
    bundle = verify_bundle(active, coop)
    verify_selection(read(raw / 'manifest.json'), read(active / 'manifest.json'))
    target = read(Path('/recipe/config/checkpoint-contract.json'))['target']
    snapshot = Path('/root/.cache/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw/snapshots') / target['revision']
    checkpoint = authenticate(snapshot)
    verify_shapes(snapshot)
    os.environ.update(GLM53_COOP_QUALIFICATION='1', GLM53_COOP_GEOMETRY='0',
                      GLM53_COOP_BUNDLE='/campaign/raw-bundle')
    import test_cuda_integration as gate
    import test_exl3_overlay
    import torch
    import exllamav3_ext
    from safetensors import safe_open
    ctypes.CDLL('/campaign/raw-bundle/cooperative_moe.so')
    ldd = output(['ldd', '/campaign/raw-bundle/cooperative_moe.so'])
    need('libcudart.so.13 =>' in ldd and 'not found' not in ldd, 'shared cudart linkage')
    help_text = output(['/sanitizer/compute-sanitizer', '--help'])
    need(all(flag in help_text for flag in ('--dump-kernel-launches', '--kernel-name', '--print-level', '--save', '--read')),
         'sanitizer lacks saved launch evidence support')
    report = {'status': 'PASS', 'scope': 'CPU stage-8 imports and fixture shapes; no GPU correctness',
              'checkpoint': checkpoint, 'bundle': bundle,
              'exl3_sha256': gate.EXL3_SHA256, 'fatpath_sha256': gate.FATPATH_SHA256,
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'ldd': ldd,
              'sanitizer': output(['/sanitizer/compute-sanitizer', '--version']),
              'nvcc': output(['/usr/local/cuda/bin/nvcc', '--version']),
              'gcc': output(['gcc', '--version'])}
    if len(sys.argv) > 1:
        gpu = torch.cuda.get_device_properties(0)
        report['gpu'] = {'name': gpu.name, 'capability': [gpu.major, gpu.minor],
                         'driver': output(['nvidia-smi', '--query-gpu=driver_version,uuid', '--format=csv,noheader']).strip()}
        need(gpu.name == 'NVIDIA GB10' and (gpu.major, gpu.minor) == (12, 1), 'qualification requires GB10')
    from coop_projection import environment
    diagnostics.retain(json.dumps(report), Path('/work/environment.json'))
    report = environment(report)
    Path('/work/environment.json').write_text(json.dumps(report, indent=2) + '\n')
    for key in ('GLM53_COOP_QUALIFICATION', 'GLM53_COOP_GEOMETRY'):
        os.environ.pop(key)
    if len(sys.argv) > 1:
        # exec a clean Python process: no imported candidate adapter survives
        # into a different geometry or the measured-policy production check.
        os.execvp(sys.argv[1], sys.argv[1:])
    print(json.dumps(report))


if __name__ == '__main__':
    diagnostics.install_exception_hook()
    main()
