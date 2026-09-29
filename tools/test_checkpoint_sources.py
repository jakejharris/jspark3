#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline snapshot-layout and corruption controls using tiny, fully hashed files."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'recipe/scripts'))
import validate_checkpoint as cp
import _coop_checkpoint as coop
import _fleetctl as fleet
import remote_preflight as preflight


def digest(data):
    return hashlib.sha256(data).hexdigest()


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.cards = {n: ('card for ' + n).encode() for n in ('brandon', 'mia', 'mirror')}
        self.shards = {f'model-{i:05d}-of-00120.safetensors': f'shard {i}'.encode() for i in range(1, 121)}
        self.publication = {f'.materialization/shards/{n}.json': b'materialization' for n in self.shards}
        self.publication.update({f'runtime/file-{i}.py': b'publication' for i in range(72)})
        self.files = {'.gitattributes': b'attributes', 'LICENSE': b'pinned license', 'README.md': b'old card',
                      'chat_template.jinja': b'template', 'config.json': b'native config',
                      'exl3-mcg-storage-abi.json': b'{}', 'generation_config.json': b'{}',
                      'materialization-receipt.json': b'{}', 'processor_config.json': b'{}',
                      'provenance/source-model-revision.json': b'{}', 'quantization/recipe.json': b'{}',
                      'quantization_config.json': b'{}', 'receipts/checkpoint.json': b'{}',
                      'tokenizer.json': b'tokenizer', 'tokenizer_config.json': b'{}',
                      'model.safetensors.index.json': json.dumps({'metadata': {'total_size': 123},
                          'weight_map': {f'tensor-{i}': n for i, n in enumerate(self.shards)}}).encode(),
                      **self.shards, **self.publication}
        hashes = {n: digest(b) for n, b in self.files.items()}
        hashes['LICENSE'] = digest(b'old license')
        self.ledger = ''.join(f'{h}  {n}\n' for n, h in sorted(hashes.items())).encode()
        stack = ExitStack()
        self.addCleanup(stack.close)
        values = {'LEDGER_SHA': digest(self.ledger), 'LICENSE_SHA': digest(self.files['LICENSE']),
                  'SOURCE_README': digest(self.cards['brandon']),
                  'MIRROR_READMES': {digest(self.cards['mia']), digest(self.cards['mirror'])},
                  'TARGET_NATIVE': digest(self.files['config.json']), 'TARGET_RUNTIME': digest(b'runtime config'),
                  'INDEX_SHA': digest(self.files['model.safetensors.index.json']),
                  'TOKENIZER': digest(self.files['tokenizer.json']),
                  'TOKENIZER_CONFIG': digest(self.files['tokenizer_config.json']),
                  'INDEXED_BYTES': 123, 'PHYSICAL_BYTES': sum(map(len, self.shards.values()))}
        for name, value in values.items():
            stack.enter_context(patch.object(cp, name, value))

    def snapshot(self, label):
        source = self.base / label
        source.mkdir()
        for name, data in self.files.items():
            if label != 'brandon' and name in self.publication:
                continue
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.cards[label] if name == 'README.md' else data)
        (source / 'SHA256SUMS').write_bytes(self.ledger)
        runtime = self.base / (label + '-runtime')
        runtime.mkdir()
        (runtime / 'config.json').write_bytes(b'runtime config')
        for item in source.iterdir():
            if item.name != 'config.json':
                (runtime / item.name).symlink_to('../' + label + '/' + item.name)
        return source, runtime

    def test_source_and_both_existing_mirrors_pass_without_writes(self):
        for label in self.cards:
            with self.subTest(snapshot=label):
                source, runtime = self.snapshot(label)
                before = {p.relative_to(source): p.read_bytes() for p in source.rglob('*') if p.is_file()}
                self.assertEqual(cp.validate_target(source, runtime, 2)['shards'], 120)
                self.assertEqual(before, {p.relative_to(source): p.read_bytes() for p in source.rglob('*') if p.is_file()})

    def test_each_snapshot_refuses_corrupt_shard_config_tokenizer_and_license(self):
        for label in self.cards:
            source, runtime = self.snapshot(label)
            for name in ('model-00001-of-00120.safetensors', 'config.json', 'tokenizer.json',
                         'tokenizer_config.json', 'LICENSE', 'README.md', 'SHA256SUMS'):
                with self.subTest(snapshot=label, file=name):
                    path = source / name
                    old = path.read_bytes()
                    path.write_bytes(old + b'changed')
                    with self.assertRaises(cp.Refusal):
                        cp.validate_target(source, runtime, 1)
                    path.write_bytes(old)

    def test_source_refuses_missing_or_corrupt_publication_files(self):
        source, runtime = self.snapshot('brandon')
        for name in (next(iter(self.publication)), 'runtime/file-0.py'):
            path = source / name
            before = path.read_bytes()
            path.unlink()
            with self.assertRaisesRegex(cp.Refusal, 'omission'):
                cp.validate_target(source, runtime, 1)
            path.write_bytes(before + b'changed')
            with self.assertRaisesRegex(cp.Refusal, 'checksum'):
                cp.validate_target(source, runtime, 1)
            path.write_bytes(before)

    def test_mirrors_refuse_partial_publication_downloads(self):
        for label in ('mia', 'mirror'):
            source, runtime = self.snapshot(label)
            name = next(iter(self.publication))
            path = source / name
            path.parent.mkdir(parents=True)
            path.write_bytes(self.publication[name])
            with self.assertRaisesRegex(cp.Refusal, 'omission'):
                cp.validate_target(source, runtime, 1)

    def test_missing_and_extra_shards_and_broken_runtime_links_refuse(self):
        source, runtime = self.snapshot('brandon')
        shard = source / 'model-00001-of-00120.safetensors'
        before = shard.read_bytes()
        shard.unlink()
        with self.assertRaisesRegex(cp.Refusal, 'omission'):
            cp.validate_target(source, runtime, 1)
        shard.write_bytes(before)
        extra = source / 'model-00121-of-00120.safetensors'
        extra.write_bytes(b'extra')
        with self.assertRaisesRegex(cp.Refusal, 'shard inventory'):
            cp.validate_target(source, runtime, 1)
        extra.unlink()
        link = runtime / shard.name
        link.unlink()
        link.symlink_to('../brandon/LICENSE')
        with self.assertRaisesRegex(cp.Refusal, 'runtime-view link'):
            cp.validate_target(source, runtime, 1)

    def test_symlinked_native_files_refuse(self):
        source, runtime = self.snapshot('brandon')
        for name in ('README.md', 'LICENSE', 'model-00001-of-00120.safetensors', 'SHA256SUMS'):
            path = source / name
            data = path.read_bytes()
            saved = self.base / 'saved'
            saved.write_bytes(data)
            path.unlink()
            path.symlink_to(saved)
            with self.subTest(file=name), self.assertRaises(cp.Refusal):
                cp.validate_target(source, runtime, 1)
            path.unlink()
            path.write_bytes(data)


class ContractTests(unittest.TestCase):
    def test_pins_and_compatibility_paths_agree(self):
        target = json.loads((ROOT / 'recipe/config/checkpoint-contract.json').read_text())['target']
        self.assertEqual(target['repository'], 'brandonmusic/GLM-5.3-Flash-tr3-4bpw')
        self.assertEqual(target['revision'], '5ab363a8dcf6405955fd5f99671e01a1c9fb124b')
        self.assertEqual(target['license_sha256'], cp.LICENSE_SHA)
        self.assertEqual(target['publication_ledger_sha256'], cp.LEDGER_SHA)
        self.assertEqual(target['local_directory'], preflight.TARGET_NATIVE)
        self.assertEqual('/models/' + preflight.TARGET_RUNTIME, fleet.TARGET_RUNTIME)
        self.assertIn(fleet.TARGET_RUNTIME, (ROOT / 'recipe/scripts/container_entry.sh').read_text())
        self.assertIn(f'--local-dir "$JSPARK_MODEL_ROOT/{target["local_directory"]}"',
                      (ROOT / 'docs/INSTALL.md').read_text())
        self.assertEqual(coop.authority()['revision'], '25a44fdbf16862a46b7cc9921142c6c81350af2f')

    def test_component_pin_must_remain_an_accepted_alternate(self):
        read = coop.q.read
        def changed(path):
            value = read(path)
            if path.name == 'checkpoint-contract.json':
                value['target']['accepted_alternates'] = []
            return value
        with patch.object(coop.q, 'read', side_effect=changed), self.assertRaisesRegex(ValueError, 'authority'):
            coop.authority()


if __name__ == '__main__':
    unittest.main()
