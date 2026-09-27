#!/usr/bin/env python3
"""Bounded candidate-only race gate; full shapes are covered by the main gate."""
import json
import torch
import test_cuda_integration as gate


def main():
    torch.manual_seed(20260916)
    result = gate.real_expert_layer(torch.device("cuda"), 96)
    if result is None:
        raise RuntimeError("real-weight snapshot unavailable")
    layer, hidden, intermediate = result
    if (hidden, intermediate) != (4096, 2048):
        raise AssertionError("wrong real-weight dimensions")
    gate.attach_map(layer)
    for rows in (32, 64):
        x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16) * .1
        ids, weights = gate.routes(rows, "concentrated")
        saved_ids = ids.clone()
        def run(impl):
            return impl(x, ids, weights, layer, layer._exl3_inners, layer.expert_map, 10.)
        ref = run(gate.stock)
        actual = run(gate.wrapper).clone()
        gate.compare(actual, ref, f"race-smoke/rank={gate.EP_RANK}/rows={rows}")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(gate.wrapper)
        graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(actual, captured):
            raise AssertionError("graph mismatch")
        ids.fill_(2**40)
        graph.replay()
        torch.cuda.synchronize()
        if bool(torch.any(captured)):
            raise AssertionError("invalid routes did not clear all rows")
        ids.copy_(saved_ids)
        graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(actual, captured):
            raise AssertionError("counter reuse after invalid routes")
        print(json.dumps({"stage": "race_smoke", "rank": gate.EP_RANK, "rows": rows, "pass": True}), flush=True)
    print(json.dumps({"stage": "race_smoke_complete", "rank": gate.EP_RANK, "pass": True}), flush=True)


if __name__ == "__main__":
    main()
