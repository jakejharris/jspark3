"""Bounded GB10 KDA layout screen: full chain and recurrent step, stock A/B/A.

Synthetic inputs use the live TP3 rank's 22 heads and padded projection stride.
This measures one rank's KDA chain, not projection GEMMs, network, or service TTFT.
"""

import argparse
import hashlib
import json
import statistics


def main():
    import torch
    from tensorfold.families.glm5_next.cuda import kda, kda_tiles

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[64, 1000, 2048, 8192])
    parser.add_argument("--variants", type=int, nargs="+", default=list(range(1, 7)))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.rows) < 1 or args.iterations < 1 or args.repeats < 3 or not set(args.variants) <= set(range(1, 7)):
        parser.error("positive rows/iterations, at least three repeats, variants 1..6 required")
    stock, candidate = kda._ext(), kda_tiles._ext()
    h, c = 22, 3 * 22 * 128
    b_off, width = c + 256, 8768

    def measure(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / args.iterations

    def same(a, b):
        return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))

    for rows in args.rows:
        gen = torch.Generator(device="cuda").manual_seed(950 + rows)
        rand = lambda shape: torch.randn(shape, device="cuda", generator=gen)
        p, a, gate = (rand((rows, width)) * .5).bfloat16(), rand((rows, h * 128)).bfloat16(), rand((rows, h * 128)).bfloat16()
        cs, cw = (rand((3, c)) * .5).bfloat16(), (rand((c, 4)) * .3).bfloat16()
        state, log, bias, norm = rand((h, 128, 128)) * .05, rand((h,)), rand((h * 128,)), (1 + rand((128,)) * .1).bfloat16()
        fixed = (p, width, b_off, a, h * 128, gate, h * 128, cs, cw, state, log, bias, norm, 1e-5, -5., rows)
        sc = kda.KDAScratch(rows, h, "cuda")
        q, y, nxt = torch.empty_like(sc.k), torch.empty_like(sc.v), torch.empty_like(state)
        tail = (sc.out, nxt, sc.k, sc.v, sc.g, sc.b, q, y)
        old_chain = lambda: stock.chain_wide(*fixed, *tail)
        old_chain()
        reference_out, reference_state = sc.out.clone(), nxt.clone()
        state_sha = hashlib.sha256(reference_state.cpu().numpy().tobytes()).hexdigest()
        old_step = lambda: candidate.step(state, q, sc.k, sc.v, sc.g, sc.b, rows, y, nxt, 0)
        # Time each variant with local A/B/A controls rather than comparing across rows.
        for variant in args.variants:
            new_chain = lambda: candidate.chain_wide(*fixed, *tail, variant)
            new_step = lambda: candidate.step(state, q, sc.k, sc.v, sc.g, sc.b, rows, y, nxt, variant)
            new_chain()
            exact = same(sc.out, reference_out) and same(nxt, reference_state)
            if not exact:
                raise RuntimeError(f"state/output bytes differ at rows={rows}, variant={variant}")
            for phase, before, after in (("chain", old_chain, new_chain), ("step", old_step, new_step)):
                a_times, b_times = [], []
                for _ in range(args.repeats):
                    a_times.append(measure(before))
                    b_times.append(measure(after))
                    a_times.append(measure(before))
                base, new = statistics.median(a_times), statistics.median(b_times)
                print(json.dumps(dict(rows=rows, heads=h, variant=variant, phase=phase, exact=exact,
                                      reference_state_sha256=state_sha, stock_ms=base, candidate_ms=new,
                                      ratio=new / base, stock_trials_ms=a_times,
                                      candidate_trials_ms=b_times, peak_allocated=torch.cuda.max_memory_allocated())), flush=True)
        del fixed, tail, p, a, gate, cs, cw, state, log, bias, norm, sc, q, y, nxt, reference_out, reference_state
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
