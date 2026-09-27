"""CPU contracts; these tests do not establish GPU arithmetic correctness."""
import ast
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
import torch
import runtime
import manifest

ROOT = Path(__file__).parent
ROW_SET = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20, 21, 24, 25, 28, 30, 32, 35, 40, 48, 56, 64)


class Contracts(unittest.TestCase):
    def test_dimensions_and_capacity_boundaries(self):
        self.assertEqual(runtime.HIDDEN, 4096)
        self.assertEqual(runtime.INTERMEDIATE_LOCAL, 2048)
        self.assertEqual(runtime.EXPERTS_MAX, 96)
        self.assertEqual(runtime.CTR_LENS, {32: 5635, 64: 11267})
        for rows in ROW_SET:
            self.assertEqual(runtime.row_capacity(rows), 32 if rows <= 32 else 64)
        for rows in (0, -1, 65, 4096):
            with self.assertRaises(ValueError):
                runtime.row_capacity(rows)

    def test_exhaustive_96_by_512_keys(self):
        # Mirrors the packed key expression, checks collision and lexicographic order.
        keys = [(expert << 9) | slot for expert in range(96) for slot in range(512)]
        self.assertEqual(len(set(keys)), 96 * 512)
        self.assertEqual(keys, sorted(keys))
        for expert in range(95):
            self.assertLess((expert << 9) | 255, (expert << 9) | 256)
            self.assertLess((expert << 9) | 511, (expert + 1) << 9)
        source = (ROOT / "native/cooperative_moe_kernel.cuh").read_text()
        self.assertIn("SLOT_BITS = 9", source)
        self.assertIn("(my_e << SLOT_BITS) | t", source)
        self.assertIn("(eu >= 0 && ((eu << SLOT_BITS) | u) < key)", source)
        self.assertNotIn("<< 8", source)

    def test_actual_overlay_mapping(self):
        # Execute the real overlay's pure mapping function without importing vLLM.
        file = Path(os.environ["GLM53_COOP_EXL3_SOURCE"])
        tree = ast.parse(file.read_text())
        func = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "map_topk_to_local")
        scope = {"torch": torch}
        exec(compile(ast.Module(body=[func], type_ignores=[]), str(file), "exec"), scope)
        mapping = scope["map_topk_to_local"]
        expert_map = torch.full((288,), -1, dtype=torch.long)
        expert_map[96:192] = torch.arange(96)
        ids = torch.tensor([[-1, 0, 95, 96, 191, 192, 288, 2**40]])
        self.assertEqual(mapping(ids, 96, expert_map).tolist(), [96, 96, 96, 0, 95, 96, 96, 96])
        self.assertEqual(mapping(ids, 96, None).tolist(), [96, 0, 95, 96, 96, 96, 96, 96])
        self.assertEqual(mapping(ids, 96, torch.empty(0, dtype=torch.long)).tolist(), [96] * 8)

    def test_manifest_integrity_and_abi_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copy2(ROOT / "runtime.py", root / "runtime.py")
            shutil.copy2(ROOT / "dispatch_policy.json", root / "dispatch_policy.json")
            shutil.copy2(ROOT / "PROVENANCE.json", root / "PROVENANCE.json")
            for name in ("LICENSE.MIT", "LICENSE.upstream-AGPL-3.0"):
                shutil.copy2(ROOT / name, root / name)
            shutil.copy2(ROOT / "vendor/LICENSE.exllamav3", root / "LICENSE.exllamav3")
            (root / "toolchain.txt").write_text("test fixture only")
            (root / "cooperative_moe.so").write_bytes(b"not a loadable library")
            policy = json.loads((root / "dispatch_policy.json").read_text())
            policy["native_sha256"] = manifest.digest(root / "cooperative_moe.so")
            (root / "dispatch_policy.json").write_text(json.dumps(policy))
            manifest.create(root)
            self.assertEqual(runtime.verify_bundle(root), root / "cooperative_moe.so")
            policy["native_sha256"] = "0" * 64
            (root / "dispatch_policy.json").write_text(json.dumps(policy))
            manifest.create(root)  # all content hashes now match, but policy provenance does not
            with self.assertRaisesRegex(runtime.CooperativeMoEError, "not measured for this native binary"):
                runtime.verify_bundle(root)
            policy["native_sha256"] = manifest.digest(root / "cooperative_moe.so")
            (root / "dispatch_policy.json").write_text(json.dumps(policy))
            manifest.create(root)
            original = (root / "manifest.json").read_text()
            changed = json.loads(original)
            changed["contract"]["abi"] = 1
            (root / "manifest.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(runtime.CooperativeMoEError, "ABI mismatch"):
                runtime.verify_bundle(root)
            (root / "manifest.json").write_text(original)
            (root / "cooperative_moe.so").write_bytes(b"tampered")
            with self.assertRaisesRegex(runtime.CooperativeMoEError, "unvalidated"):
                runtime.verify_bundle(root)

    def test_policy_complete_and_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = {"schema": 1, "rows": {str(n): "stock" for n in range(1, 65)}}
            policy["rows"]["64"] = 2
            (root / "dispatch_policy.json").write_text(json.dumps(policy))
            actual = runtime.load_row_policy(root)
            self.assertEqual(actual[1], "stock")
            self.assertEqual(actual[64], 2)
            for invalid in (True, 3, "1", None):
                policy["rows"]["64"] = invalid
                (root / "dispatch_policy.json").write_text(json.dumps(policy))
                with self.assertRaises(runtime.CooperativeMoEError):
                    runtime.load_row_policy(root)
            del policy["rows"]["64"]
            (root / "dispatch_policy.json").write_text(json.dumps(policy))
            with self.assertRaises(runtime.CooperativeMoEError):
                runtime.load_row_policy(root)

    def test_pinned_headers(self):
        manifest.verify_sources(ROOT)


if __name__ == "__main__":
    unittest.main()
