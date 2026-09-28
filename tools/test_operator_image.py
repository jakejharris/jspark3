#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline image-policy regression tests; no fleet or GPU contact."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "recipe/scripts"))
import _image_identity as image


def fixture():
    return {"schema_version": 1,
            "verification": "fixed-dockerfile-build-and-instanttensor-check",
            "manifest_digest": "sha256:" + "1" * 64,
            "config_digest": "sha256:" + "2" * 64,
            "diff_ids": ["sha256:" + "3" * 64],
            "build_policy": image.build_policy(),
            "source_recipe_sha256": image.sha(ROOT / "recipe/SHA256SUMS")}


def write_record(path, record):
    payload = {k: v for k, v in record.items() if k != "payload_sha256"}
    payload["payload_sha256"] = hashlib.sha256(image.canonical(payload)).hexdigest()
    path.write_bytes(image.canonical(payload))


class OperatorImageTests(unittest.TestCase):
    def test_real_preflight_launch_preserves_recipe(self):
        import _fleetctl
        with tempfile.TemporaryDirectory() as directory:
            recipe = Path(directory) / "recipe"
            shutil.copytree(ROOT / "recipe", recipe)
            values = _fleetctl.load_env(ROOT / "recipe/.env.example")
            values["JSPARK_RECIPE_ROOT"] = str(recipe)
            command = _fleetctl.preflight_argv(values, 0) + ["--recipe-only"]
            self.assertNotIn("-B", command)
            env = {k: v for k, v in os.environ.items() if k != "PYTHONDONTWRITEBYTECODE"}
            process = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertFalse(list(recipe.rglob("__pycache__")))

    def test_real_receipt_mint_preserves_recipe(self):
        import _fleetctl
        with tempfile.TemporaryDirectory() as directory:
            recipe = Path(directory) / "recipe"
            shutil.copytree(ROOT / "recipe", recipe)
            env = {k: v for k, v in os.environ.items() if k != "PYTHONDONTWRITEBYTECODE"}
            with patch.object(_fleetctl, "RECIPE_ROOT", recipe), patch.dict(os.environ, env, clear=True):
                receipt = json.loads(_fleetctl.mint_image_receipt(
                    container_id="4" * 64, rank=0, preflight_sha="5" * 64,
                    recipe_sha=image.sha(recipe / "SHA256SUMS")))
            self.assertEqual(receipt["container_id"], "4" * 64)
            self.assertFalse(list(recipe.rglob("__pycache__")))
            process = subprocess.run([sys.executable, "-S", str(recipe / "scripts/remote_preflight.py"),
                                      "--recipe-only", "--recipe-root", str(recipe)],
                                     env=env, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)

    def test_policy_matches_fixed_build_inputs(self):
        policy = image.build_policy()
        context = ROOT / "docker/stock-v13"
        self.assertEqual(image.sha(context / "Dockerfile"), policy["dockerfile_sha256"])
        self.assertEqual(json.loads((context / "instanttensor-files.json").read_text()),
                         policy["instanttensor_files"])
        bases = [line.split()[1] for line in (context / "Dockerfile").read_text().splitlines()
                 if line.startswith("FROM ")]
        self.assertEqual(bases, policy["base_images"])

    def test_unknown_inputs_refused_even_with_valid_self_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.json"
            for key in ("dockerfile_sha256", "base_images", "instanttensor_files"):
                record = fixture()
                record["build_policy"][key] = "changed"
                write_record(path, record)
                with self.assertRaisesRegex(image.ImageRefusal, "policy drift"):
                    image.read_operator_record(path)

    def test_receipt_tamper_and_symlink_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.json"
            write_record(path, fixture())
            record = json.loads(path.read_text())
            record["config_digest"] = "sha256:" + "4" * 64
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(image.ImageRefusal, "payload hash"):
                image.read_operator_record(path)
            path.unlink()
            path.symlink_to(Path(directory) / "absent")
            with self.assertRaisesRegex(image.ImageRefusal, "symlinked"):
                image.read_operator_record(path)

    def test_instanttensor_hash_refusal_cleans_stopped_container(self):
        original = b"verified package bytes"
        policy = {"instanttensor_files": {"instanttensor/_impl.py": hashlib.sha256(original).hexdigest()}}
        for data in (original, b"tampered package bytes"):
            calls = []
            def docker(args):
                calls.append(args)
                if args[0] == "create":
                    return "test-container\n"
                if args[0] == "cp":
                    Path(args[-1]).write_bytes(data)
                return ""
            with patch.object(image, "build_policy", return_value=policy), patch.object(image, "docker", docker):
                if data == original:
                    image.verify_instanttensor("sha256:" + "2" * 64)
                else:
                    with self.assertRaisesRegex(image.ImageRefusal, "InstantTensor hash mismatch"):
                        image.verify_instanttensor("sha256:" + "2" * 64)
            self.assertEqual(calls[-1], ["rm", "-v", "test-container"])
            self.assertFalse(any(row[0] in ("run", "start") for row in calls))

    def test_config_and_layers_drift_refused(self):
        identity = fixture()
        for change in ({"Id": "sha256:" + "9" * 64}, {"RootFS": {"Layers": []}}):
            observed = {"Id": identity["config_digest"], "RootFS": {"Layers": identity["diff_ids"]}}
            observed.update(change)
            with patch.object(image, "inspect_image", return_value=observed), patch.object(image, "verify_instanttensor") as check:
                with self.assertRaisesRegex(image.ImageRefusal, "identity drift"):
                    image.verify_local_image(identity)
                check.assert_not_called()

    def test_image_only_is_not_admission_receipt(self):
        import _fleetctl
        values = _fleetctl.load_env(ROOT / "recipe/.env.example")
        receipt = {"status": "PASS", "scope": "image-only", "hardware_qualified": False}
        with self.assertRaisesRegex(_fleetctl.Refusal, "exact configuration"):
            _fleetctl.validate_preflight_receipt(receipt, values, "5" * 64)

    def test_operator_identity_reaches_all_consumers(self):
        with tempfile.TemporaryDirectory() as directory:
            recipe = Path(directory) / "recipe"
            shutil.copytree(ROOT / "recipe", recipe)
            record_path = recipe / "config/operator-image.json"
            write_record(record_path, fixture())
            receipt_path = Path(directory) / "container.json"
            process = subprocess.run([sys.executable, "-B", str(recipe / "scripts/make_image_receipt.py"),
                                      "--output", str(receipt_path), "--container-id", "4" * 64,
                                      "--rank", "1", "--preflight-sha256", "5" * 64,
                                      "--recipe-manifest-sha256", "6" * 64], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            code = '''
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import _atomic, _fleetctl, remote_preflight, make_image_receipt, apply_coop_moe
from _image_identity import selected_identity, reference_identity
identity = selected_identity()
assert identity['config_digest'] == 'sha256:' + '2' * 64
assert _fleetctl.IMAGE == remote_preflight.IMAGE == make_image_receipt.CONFIG == identity['config_digest']
assert _fleetctl.IMAGE_MANIFEST == remote_preflight.IMAGE_MANIFEST == make_image_receipt.MANIFEST == identity['manifest_digest']
receipt = _atomic.read_image_receipt(Path(sys.argv[2]))
assert receipt['config_digest'] == identity['config_digest'] and receipt['rank'] == 1
r = Path(sys.argv[1]).parent
for path in [*r.glob('config/*-contract.json'), *r.glob('overlays/**/INSTALL_CONTRACT.json')]:
    contract = json.loads(path.read_text())
    if 'image' in contract:
        for transform, section in contract['transforms'].items():
            _atomic.load_contract(path, transform, section)
assert apply_coop_moe.reference_build_image()['config'] == reference_identity()['config_digest'][7:]
values = _fleetctl.load_env(r / '.env.example')
assert identity['config_digest'] in _fleetctl.container_argv(values, 1, '5' * 64, '6' * 64)
receipt['config_digest'] = reference_identity()['config_digest']
receipt.pop('payload_sha256')
receipt['payload_sha256'] = _atomic.sha_bytes(_atomic.canonical(receipt))
Path(sys.argv[2]).write_bytes(_atomic.canonical(receipt))
try:
    _atomic.read_image_receipt(Path(sys.argv[2]))
except _atomic.Refusal:
    pass
else:
    raise AssertionError('accepted reference receipt for operator image')
'''
            process = subprocess.run([sys.executable, "-B", "-c", code, str(recipe / "scripts"), str(receipt_path)],
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            # Binding the receipt to the recipe inventory detects later edits.
            files = sorted(p for p in recipe.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
            (recipe / "SHA256SUMS").write_text("".join(f"{image.sha(p)}  {p.relative_to(recipe)}\n" for p in files))
            command = [sys.executable, "-B", str(recipe / "scripts/remote_preflight.py"),
                       "--recipe-only", "--recipe-root", str(recipe)]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            record = fixture(); record['config_digest'] = 'sha256:' + '9' * 64
            write_record(record_path, record)
            process = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(process.returncode, 9)
            self.assertIn('recipe manifest mismatch', process.stderr)

    def test_image_only_rejects_different_source(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "image.json"
            record = fixture(); record["source_recipe_sha256"] = "0" * 64
            write_record(receipt, record)
            process = subprocess.run([sys.executable, "-B", str(ROOT / "recipe/scripts/remote_preflight.py"),
                                      "--recipe-root", str(ROOT / "recipe"), "--image-only",
                                      "--image-receipt", str(receipt)], capture_output=True, text=True)
            self.assertEqual(process.returncode, 9)
            self.assertIn("different source recipe", process.stderr)


if __name__ == "__main__":
    unittest.main()
