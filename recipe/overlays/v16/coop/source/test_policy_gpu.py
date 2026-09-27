#!/usr/bin/env python3
"""Verify the production policy itself, without the qualification override."""
import json
import os
os.environ.pop("GLM53_COOP_QUALIFICATION", None)
import torch
import test_cuda_integration as gate


def main():
    result = gate.real_expert_layer(torch.device("cuda"), 96)
    if result is None:
        raise RuntimeError("real weights unavailable")
    layer, _, _ = result
    gate.attach_map(layer)
    native = layer._glm53_coop_native
    original = native.launch
    choices = {}
    for rows in (*gate.ROWS, 33, 65):
        calls = []
        def counted(*args):
            calls.append(int(args[5]))
            return original(*args)
        native.launch = counted
        x = torch.randn(rows, 4096, dtype=torch.bfloat16, device="cuda") * .1
        ids, weights = gate.routes(rows, "random")
        def run(impl):
            return impl(x, ids, weights, layer, layer._exl3_inners, layer.expert_map, 10.)
        expected = native.row_policy.get(rows, "stock")
        ref = run(gate.stock)
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        actual = run(gate.wrapper)
        after = torch.cuda.memory_stats()["allocation.all.allocated"]
        actual = actual.clone()
        if expected == "stock":
            if calls:
                raise AssertionError("stock policy launched cooperative")
        elif calls != [expected] or after != before:
            raise AssertionError(f"policy/alloc mismatch rows={rows} calls={calls} delta={after-before}")
        gate.compare(actual, ref, f"production-policy/rows={rows}/choice={expected}")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(gate.wrapper)
        x.copy_(torch.randn_like(x) * .1)
        changed = run(gate.wrapper).clone()
        graph.replay()
        torch.cuda.synchronize()
        if expected != "stock" and not torch.equal(captured, changed):
            raise AssertionError("candidate graph/eager mismatch")
        gate.compare(captured, changed, f"production-policy/graph/rows={rows}")
        choices[str(rows)] = expected
    native.launch = original
    print(json.dumps({"stage": "production_policy_complete", "pass": True, "choices": choices}), flush=True)


if __name__ == "__main__":
    main()
