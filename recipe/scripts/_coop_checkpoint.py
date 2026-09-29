# SPDX-License-Identifier: Apache-2.0
"""Authenticate the layer-3 subset against the pinned publication ledger."""
import json
import struct

import _coop_qualification as q
import validate_checkpoint as checkpoint


def authority():
    pin = q.read(q.RECIPE / 'config/coop-checkpoint.json')
    target = q.read(q.RECIPE / 'config/checkpoint-contract.json')['target']
    accepted = [target, *target.get('accepted_alternates', [])]
    q.need(set(pin) == {'repository', 'revision', 'publication_ledger_sha256', 'files'}
           and any(all(pin[k] == source[k] for k in ('repository', 'revision')) for source in accepted)
           and pin['publication_ledger_sha256'] == target['publication_ledger_sha256']
           and pin['publication_ledger_sha256'] == checkpoint.LEDGER_SHA
           and pin['files'].get('model.safetensors.index.json') == target['model_index_sha256'] == checkpoint.INDEX_SHA
           and set(pin['files']) == {'model.safetensors.index.json',
                                    *[f'model-{n:05d}-of-00120.safetensors' for n in (57, 58, 59)]}
           and all(q.hash_ok(v) for v in pin['files'].values()), 'checkpoint authority drift')
    return pin


def tensors(index):
    grouped = {}
    for expert in range(288):
        for proj in ('gate_proj', 'up_proj', 'down_proj'):
            for suffix in ('trellis', 'suh', 'svh', 'mcg'):
                name = f'model.language_model.layers.3.mlp.experts.{expert}.{proj}.{suffix}'
                grouped.setdefault(index['weight_map'][name], []).append((name, proj, suffix))
    return grouped


def authenticate(model):
    pin = authority()
    q.need(model.name == pin['revision'], 'model-root must be the exact single snapshot revision directory')
    try:
        ledger = checkpoint.ledger(q.regular(model, 'SHA256SUMS'))
    except checkpoint.Refusal as exc:
        raise ValueError(str(exc)) from exc
    for name, expected in pin['files'].items():
        q.need(ledger.get(name) == expected, 'checkpoint subset differs from publication ledger: ' + name)
        q.need(checkpoint.sha(q.regular(model, name)) == expected, 'checkpoint publication digest mismatch: ' + name)
    index = q.read(model / 'model.safetensors.index.json')
    q.need(set(tensors(index)) == set(pin['files']) - {'model.safetensors.index.json'}, 'checkpoint subset mapping drift')
    return pin


def verify_shapes(model):
    """Read bounded safetensors headers only; authentication happens separately."""
    for shard, names in tensors(q.read(model / 'model.safetensors.index.json')).items():
        with q.regular(model, shard).open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            q.need(0 < size <= 16 * 1024 * 1024, 'invalid safetensors header size')
            header = json.loads(stream.read(size))
        for name, proj, suffix in names:
            hidden, inter = (2048, 4096) if proj == 'down_proj' else (4096, 2048)
            expected = {'trellis': ('I16', [hidden // 16, inter // 16, 64]),
                        'suh': ('F16', [hidden]), 'svh': ('F16', [inter]), 'mcg': ('I32', [1])}[suffix]
            row = header[name]
            q.need((row.get('dtype'), row.get('shape')) == expected, 'checkpoint tensor dtype/shape differs: ' + name)

