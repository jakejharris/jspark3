#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU fixture controls, including an opt-in real-index review reproduction."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'recipe/scripts'))
import _coop_checkpoint as cp
import _coop_qualification as q

REAL_FIXTURE = LEDGER = None


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / cp.authority()['revision']
        self.root.mkdir()
        # Header-only fixtures test metadata, never masquerade as authenticated
        # weights. Production always authenticates complete bytes first.
        self.index = {'weight_map':{}}
        self.headers = {}
        schema = {'gate_proj':{'trellis':('I16',[256,128,64]),'suh':('F16',[4096]),'svh':('F16',[2048]),'mcg':('I32',[1])},
                  'up_proj':{'trellis':('I16',[256,128,64]),'suh':('F16',[4096]),'svh':('F16',[2048]),'mcg':('I32',[1])},
                  'down_proj':{'trellis':('I16',[128,256,64]),'suh':('F16',[2048]),'svh':('F16',[4096]),'mcg':('I32',[1])}}
        for expert in range(288):
            shard = f'model-{57 + expert // 96:05d}-of-00120.safetensors'
            for proj, suffixes in schema.items():
                for suffix, (dtype, shape) in suffixes.items():
                    name = f'model.language_model.layers.3.mlp.experts.{expert}.{proj}.{suffix}'
                    self.index['weight_map'][name] = shard
                    self.headers.setdefault(shard,{})[name] = {'dtype':dtype,'shape':shape}
        (self.root / 'model.safetensors.index.json').write_text(json.dumps(self.index))
        self.write_headers()

    def write_headers(self):
        for shard, header in self.headers.items():
            data = json.dumps(header).encode()
            (self.root / shard).write_bytes(struct.pack('<Q',len(data)) + data)

    def test_all_288_experts_have_exact_dtypes_and_full_shapes(self):
        cp.verify_shapes(self.root)
        original = copy.deepcopy(self.headers)
        shard = 'model-00057-of-00120.safetensors'
        for proj in ('gate_proj','up_proj','down_proj'):
            for suffix in ('trellis','suh','svh','mcg'):
                name = f'model.language_model.layers.3.mlp.experts.0.{proj}.{suffix}'
                for field, value in (('dtype','F32'),('shape',[1]),('shape',[256,128])):
                    self.headers = copy.deepcopy(original)
                    self.headers[shard][name][field] = value
                    if self.headers == original:
                        continue
                    self.write_headers()
                    with self.subTest(name=name,field=field,value=value), self.assertRaisesRegex(ValueError,'dtype/shape'):
                        cp.verify_shapes(self.root)
        self.headers = original
        del self.headers['model-00059-of-00120.safetensors']['model.language_model.layers.3.mlp.experts.287.down_proj.mcg']
        self.write_headers()
        with self.assertRaises(KeyError): cp.verify_shapes(self.root)

    def test_ledger_and_each_observed_file_must_match_authority(self):
        pin = cp.authority()
        names = list(pin['files'])
        # This isolated authority allows the tiny files only for this unit test.
        pin['files'] = {name:cp.checkpoint.sha(self.root / name) for name in names}
        ledger = dict(pin['files'])
        (self.root / 'SHA256SUMS').write_text('synthetic authority fixture')
        with patch.object(cp,'authority',return_value=pin), patch.object(cp.checkpoint,'ledger',return_value=ledger):
            self.assertEqual(cp.authenticate(self.root),pin)
            for name in names:
                path = self.root / name
                before = path.read_bytes()
                path.write_bytes(before + b'changed')
                with self.assertRaisesRegex(ValueError,'publication digest mismatch'): cp.authenticate(self.root)
                path.write_bytes(before)
            ledger[names[-1]] = q.digest_value('substituted ledger entry')
            with self.assertRaisesRegex(ValueError,'differs from publication ledger'): cp.authenticate(self.root)
        with self.assertRaisesRegex(ValueError,'SHA256SUMS hash drift'): cp.authenticate(self.root)

    def test_real_index_changed_shard_review_reproduction(self):
        if REAL_FIXTURE is None or LEDGER is None:
            self.skipTest('supply --real-checkpoint-fixture and --publication-ledger to run retained review reproduction')
        # The actual publication index/ledger are verified without patched pins.
        pin = cp.authority()
        for name in pin['files']:
            shutil.copyfile(REAL_FIXTURE / name, self.root / name)
        shutil.copyfile(LEDGER, self.root / 'SHA256SUMS')
        self.assertEqual(cp.checkpoint.sha(self.root / 'model.safetensors.index.json'),cp.checkpoint.INDEX_SHA)
        self.assertEqual(cp.checkpoint.sha(self.root / 'SHA256SUMS'),cp.checkpoint.LEDGER_SHA)
        first = self.root / 'model-00057-of-00120.safetensors'
        self.assertNotEqual(cp.checkpoint.sha(first),pin['files'][first.name])
        with self.assertRaisesRegex(ValueError,'publication digest mismatch: model-00057'): cp.authenticate(self.root)
        with first.open('r+b') as stream:
            stream.seek(-1,2)
            value = stream.read(1)[0]
            stream.seek(-1,2)
            stream.write(bytes([value ^ 1]))
        with self.assertRaisesRegex(ValueError,'publication digest mismatch: model-00057'): cp.authenticate(self.root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real-checkpoint-fixture',type=Path)
    parser.add_argument('--publication-ledger',type=Path)
    args, rest = parser.parse_known_args()
    REAL_FIXTURE, LEDGER = args.real_checkpoint_fixture, args.publication_ledger
    unittest.main(argv=[sys.argv[0],*rest])
