#!/usr/bin/env python3
"""Real-weight per-shape geometry measurement; includes adapter conversion/routing.

Only a maintenance microbenchmark. Kernel time ratios are not serving speedups.
Numerics use the frozen integration tolerances before timing each input.
"""
import json
import hashlib
import statistics
import torch
import test_cuda_integration as gate


def time_graph(graph):
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(25):
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b))
    return {"median_ms": statistics.median(samples), "min_ms": min(samples), "max_ms": max(samples)}


def main():
    print(json.dumps({"stage": "profile_identity", "native_sha256": hashlib.sha256((gate.BUNDLE / "cooperative_moe.so").read_bytes()).hexdigest()}), flush=True)
    torch.manual_seed(20260916)
    result = gate.real_expert_layer(torch.device("cuda"), 96)
    if result is None:
        raise RuntimeError("existing real-weight snapshot unavailable")
    layer, hidden, intermediate = result
    if (hidden, intermediate) != (4096, 2048):
        raise AssertionError("wrong shape")
    gate.attach_map(layer)
    geometry = layer._glm53_coop_native.geometry
    for rows in gate.ROWS:
        for pattern in ("ep_uniform", "concentrated"):
            x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16) * .1
            if pattern == "ep_uniform":
                ids = torch.stack([torch.randperm(288, device="cuda")[:8] for _ in range(rows)])
                weights = torch.rand(rows, 8, device="cuda").softmax(-1)
            else:
                ids, weights = gate.routes(rows, pattern)
            def run(impl):
                # Compare raw FP32 fused outputs before the common outer BF16 cast.
                return impl(x, ids, weights, layer, layer._exl3_inners, layer.expert_map, 10.)
            ref = run(gate.stock)
            actual = run(gate.wrapper).clone()
            gate.compare(actual, ref, f"profile/rank={gate.EP_RANK}/geometry={geometry}/rows={rows}/{pattern}")
            values = {}
            # Alternate timing order by shape to reduce systematic warm-order bias.
            impls = [("stock", gate.stock), ("candidate", gate.wrapper)]
            if rows % 2:
                impls.reverse()
            for name, impl in impls:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run(impl)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = run(impl)
                values[name] = time_graph(graph)
                if name == "candidate" and not torch.equal(captured, actual):
                    raise AssertionError(f"profile graph/eager mismatch rows={rows} geometry={geometry}")
                del graph, captured
            print(json.dumps({"stage": "profile", "rank": gate.EP_RANK, "rows": rows,
                              "pattern": pattern, "geometry": geometry, **values,
                              "serving_speedup_measured": False}), flush=True)
    print(json.dumps({"stage": "profile_complete", "rank": gate.EP_RANK, "geometry": geometry}), flush=True)


if __name__ == "__main__":
    main()
