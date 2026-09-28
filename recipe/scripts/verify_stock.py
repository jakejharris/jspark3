#!/usr/bin/env python3
"""Verify the common v1.3-v1.5 runtime plus the selected sealed v1.6 overlays."""
import sys
sys.dont_write_bytecode = True
import _diagnostics as diagnostics
import hashlib
import json
import os
import re
from pathlib import Path

import production_stock


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_overlays(recipe, package):
    v16 = 'JSPARK3_V16_PROFILE' in os.environ
    adaptive = os.environ['GLM53_ADAPTIVE_K']
    dense = os.environ.get('JSPARK3_V16_DENSE_FP8', 'off')
    assert adaptive in (('off', 'ema') if v16 else ('off',)), 'GLM53_ADAPTIVE_K'
    assert dense in (('off', 'trunk', 'negative-coarse') if v16 else ('off',)), 'JSPARK3_V16_DENSE_FP8'
    if v16:
        assert os.environ['JSPARK3_V16_PROFILE'] in ('production', 'production-stock', 'qa'), 'JSPARK3_V16_PROFILE'
        if os.environ['JSPARK3_V16_PROFILE'] == production_stock.PROFILE:
            production_stock.environment(os.environ)
        if os.environ['JSPARK3_V16_PROFILE'] == 'production':
            assert os.environ['ABLIT'] == '1', 'production promotion requires ABLIT=1'
    if dense == 'negative-coarse':
        assert (os.environ['JSPARK3_V16_PROFILE'] == 'qa' and
                os.environ.get('JSPARK3_V16_DENSE_FP8_NEGATIVE_CONTROL') ==
                'I_UNDERSTAND_TEST_ONLY'), 'dense-FP8 negative control is QA-only'
    else:
        assert 'JSPARK3_V16_DENSE_FP8_NEGATIVE_CONTROL' not in os.environ, 'unexpected dense-FP8 test acknowledgement'

    quant = package / 'vllm/model_executor/layers/quantization'
    if dense != 'off':
        from _contracts import V16_DENSE_FP8
        contract = json.loads((recipe / 'config/v16-dense-fp8-contract.json').read_text())
        assert contract['transforms']['apply_dense_fp8.py'] == V16_DENSE_FP8, 'dense-FP8 contract drift'
        # Compare installed bytes to the sealed transform output, never to an
        # arbitrary patched file or just a marker. Include both runtime modules.
        for row in V16_DENSE_FP8['targets']:
            assert sha(package / row['path']) == row['after_sha256'], row['path']
    else:
        assert sha(recipe / 'overlays/ablit_transplant.py') == sha(quant / 'ablit_transplant.py'), 'ablit_transplant.py'
    for source, installed in (('scripts/validate_ablit_artifacts.py', 'ablit_artifacts.py'),
                              ('overlays/trunk_w8a16.py', 'trunk_w8a16.py'),
                              ('overlays/instanttensor_audit.py', 'instanttensor_audit.py')):
        assert sha(recipe / source) == sha(quant / installed), installed
    return {'adaptive-k': adaptive, 'dense-fp8': dense}


def verify_ablation(value, mode, rank, manifest_sha):
    def require(condition, message):
        if not condition:
            raise RuntimeError(message)
    require(value.get('schema_version') == 1 and value.get('ablit') == mode and
            value.get('rank') == rank, 'ablation receipt rank/mode/schema mismatch')
    if mode == 0:
        if os.environ.get('JSPARK3_V16_PROFILE') == production_stock.PROFILE:
            production_stock.environment(os.environ)
            production_stock.disabled_receipt(value, rank)
        require(value.get('state') == 'DISABLED' and value.get('applied_layers') == [] and
                manifest_sha is None, 'disabled control has donor effects/configuration')
        return
    require(mode == 1 and value.get('state') == 'APPLIED_AND_PACKED' and
            value.get('manifest_sha256') == manifest_sha and
            value.get('donor_revision') == '80b6d18d77e3020f2384597081d405f19893f101',
            'missing or wrong donor application receipt')
    require(value.get('applied_layers') == list(range(15, 45)), 'active layer coverage mismatch')
    require(value.get('mtp', {}).get('active') is False and
            value['mtp'].get('artifact_verified') is True and value['mtp'].get('layer') == 45,
            'inactive MTP artifact proof missing')
    require(value.get('untouched_parameter_versions_and_storage') is True and
            value.get('untouched_parameter_count', 0) > 0 and
            set(value.get('anchor_hashes', {})) == {'0', '14'}, 'untouched parameter proof missing')
    rows = value.get('layers', [])
    require([row.get('layer') for row in rows] == list(range(15, 45)), 'packed layer coverage mismatch')
    for row in rows:
        require(row.get('packed_verified') is True and row.get('additional_packing_error') == 0.0,
                'packed donor reference mismatch')
        require(row.get('kernel_reference_rows') == 2 and row.get('kernel_reference_limit') == 0.05 and
                0 <= row.get('kernel_reference_relative_l2', float('inf')) <= 0.05,
                'missing or failed packed kernel donor comparison')
        for field in ('original_sha256', 'donor_sha256', 'slice_sha256', 'packed_sha256', 'scales_sha256'):
            require(re.fullmatch('[0-9a-f]{64}', row.get(field, '')) is not None,
                    'missing layer hash: ' + field)


def read_native_ablation(path, mode, rank, manifest_sha):
    value = json.loads(path.read_text())
    verify_ablation(value, mode, rank, manifest_sha)
    return value


def main():
    recipe = Path('/recipe')
    package = Path('/usr/local/lib/python3.12/dist-packages')
    loader_contract = json.loads((recipe / 'config/loader-audit.json').read_text())
    assert sha(package / 'vllm/model_executor/model_loader/default_loader.py') == loader_contract['after']
    loaders = {}
    loader_memory = {}
    for kind, shards in (('target', 120), ('draft', 1)):
        loaded = json.loads((Path('/evidence') / ('loader-' + kind + '.json')).read_text())
        assert loaded['state'] == 'COMPLETE' and loaded['loader'] == 'instanttensor'
        assert loaded['rank'] == int(os.environ['NODE_RANK']) and loaded['kind'] == kind
        assert loaded['shards'] == shards and loaded['tensors'] > 0 and loaded['copy_ownership'] is True
        assert loaded['first_tensor_descriptors']['nofile'] == [65536, 65536]
        assert 0 < loaded['first_tensor_descriptors']['open_fd_count'] < 65536
        if kind == 'target':
            assert loaded['bytes'] == 175622979576
        loaders[kind] = loaded
        memory = json.loads((Path('/evidence') / ('loader-memory-' + kind + '.json')).read_text())
        assert memory['rank'] == int(os.environ['NODE_RANK']) and memory['kind'] == kind
        assert memory['pass'] is True and memory['backend'] in ('URING', 'AIO')
        assert memory['completed_target_files_eligible'] == (120 if kind == 'draft' else 0)
        assert memory['required_cuda_free'] == 2 * max(memory['io_buffer_bound'], memory['tensor_buffer_bound']) + memory['kernel_high_watermark_bytes']
        assert memory['after']['cuda_free'] >= memory['required_cuda_free']
        assert memory['after_empty_cache']['host']['MemAvailable'] >= memory['required_cuda_free'] + 8 * 1024**3
        agreement = json.loads((Path('/evidence') / ('loader-memory-' + kind + '-agreement.json')).read_text())
        assert agreement['rank'] == int(os.environ['NODE_RANK']) and agreement['kind'] == kind
        assert agreement['all_ranks_pass'] is True and agreement['local_error'] is None
        loader_memory[kind] = memory
    contract = json.loads((recipe / 'config/swa-contract.json').read_text())
    receipt = json.loads(Path('/evidence/swa-receipt.json').read_text())
    assert receipt['contract_sha256'] == sha(recipe / 'config/swa-contract.json')
    assert receipt['rank'] == os.environ['NODE_RANK']
    mode = int(os.environ['ABLIT'])
    assert mode in (0, 1) and receipt['ablit'] == mode
    ablation = read_native_ablation(Path('/evidence/ablit-receipt.json'), mode,
                                    int(os.environ['NODE_RANK']),
                                    os.environ.get('JSPARK_ABLIT_MANIFEST_SHA256'))
    if mode == 1:
        from validate_ablit_artifacts import validate
        donor = validate(Path('/ablit'), os.environ['JSPARK_ABLIT_MANIFEST_SHA256'])
        expected_donors = {row['layer']: row['sha256'] for row in donor['tensors']}
        assert all(row['donor_sha256'] == expected_donors[row['layer']] for row in ablation['layers'])
    options = verify_overlays(recipe, package)
    profile = json.loads((recipe / 'config/profile.json').read_text())
    assert sha(package / 'vllm/model_executor/model_loader/base_loader.py') == profile['w8a16_overlay']['base_loader_after_sha256']
    assert receipt['state'] in ('APPLIED', 'ALREADY_APPLIED')
    expected = {name: row['after'] for name, row in contract['targets'].items()}
    assert receipt['targets'] == expected
    for name, digest in expected.items():
        assert sha(package / name) == digest, name
    for name, digest in json.loads(Path('/opt/jspark3-v13/instanttensor-files.json').read_text()).items():
        assert sha(package / name) == digest, name
    assert sha(package / 'exllamav3_ext.cpython-312-aarch64-linux-gnu.so') == '0eeb983b09dfe33451b8bc7625174320529b986a73f60b58bf4ebfa9f43ad9c0'
    assert not Path('/opt/cadence_campaign').exists()
    assert not (package / 'zzzz_campaign.pth').exists()
    assert not (package / 'vllm/models/glm5next/nvidia/glm53_ablit.py').exists()
    assert os.environ['VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA'] == '0'
    assert os.environ['GLM53_DENSE_FP8'] == 'off'
    assert os.environ['EXL3_FAT_GROUPED'] == '0'
    for name in ('GLM53_EXL3_MOE_FAST', 'GLM53_COOP_GEOMETRY', 'HAREM_KDA_FLASHKDA'):
        assert name not in os.environ, name
    record = {'status': 'PASS', 'ablit': mode, 'rank': int(os.environ['NODE_RANK']),
                      'ablation': ablation, 'loaders': loaders,
                      'loader_memory': loader_memory,
                      'swa_targets': expected, 'swa_contract_sha256': receipt['contract_sha256'],
                      'instanttensor': '0.2.0', 'native_exllama': 'exact-v11',
                      'optional_arms': (options if 'JSPARK3_V16_PROFILE' in os.environ
                                        else 'off')}
    diagnostics.retain(json.dumps(record, sort_keys=True))
    shared = {'status': 'PASS', 'ablit': mode, 'rank': record['rank'],
              'evidence_sha256': diagnostics.fingerprint(record),
              'loaders': production_stock.loader_facts(loaders)}
    if mode == 0:
        from production_stock import disabled_receipt
        disabled_receipt(ablation, record['rank'])
        shared['ablation'] = ablation
    print(json.dumps(shared, sort_keys=True))


if __name__ == '__main__':
    diagnostics.install_exception_hook()
    main()
