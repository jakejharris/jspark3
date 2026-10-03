#!/usr/bin/env python3
"""Warm up, then gate cold prefill at 8K and 32K on an otherwise idle server.

    python3 scripts/prefill-check.py [URL] [--floor-scale 1.0]

Floors, receipt sources, margins and calibration status are declared only in
FLOOR_TABLE below. Replace those rows from this script's own live rate receipts.
Use --floor-scale to multiply both floors, not to change the measurement.

Each size warms once before either measured request; every request begins with a
fresh nonce and must report cached=0. /tokenize is already enabled by the shipped
TF_V2_PARITY=1; no local tokenizer or extra server settings are needed. Lengths
are within 32 tokens of target; report and divide by actual usage.prompt_tokens.
Rate = prompt tokens / tensorfold.prefill_s. A compile-sized outlier is a reply
whose client elapsed time OR server prefill_s+decode_s exceeds
max(10 seconds, 4 * actual_tokens / scaled_floor). This is a latency guard, not
proof of no compilation: the CUDA API/metrics and engine log messages expose no
reliable per-request kernel-compile counter (cuda/server.py:647-672).

Expected warmed idle runtime about 45-75 seconds. Fresh-boot warmup may take
longer: calibration is capped at 230 seconds, each warmup at 300 seconds, then
the measured phase at 230 seconds. Socket requests time out too. PREFILL PASS
2/2 means both cold rates meet their floors and neither reply is an outlier.
Exit 1: FAIL; 2: server unreachable or did not answer in time.
"""
import argparse
import math
from pathlib import Path
import random
import secrets
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _api_check import ERRORS, Failure, body, cached, deadline, endpoint, failure_exit, integer, json_call, stream, timing

# Replace each row's floor/source/margin/provisional from this script's own rate=
# receipt (target_tokens identifies a row); set provisional=False once calibrated.
# Live plan: >=3 runs per base/ablit x dflash2/none cell; floor per length is
# floor(min(all measured rates for that length) * 0.75). For each source, cite a
# shipped document's exact rows with length and reps; if those rows are absent, use
# "the release's cold-prefill floor calibration (<length>, reps <range>)".
# Keep sources public: no internal paths, candidate/build labels, host names,
# network addresses or user names. Do not substitute rates from a different workload.
# Not measured: ablit weights without the draft model.
FLOOR_TABLE = (
    {"tokens": 8000, "floor": 1397, "margin": 0.25, "provisional": False,
     "source": "the release's cold-prefill floor calibration (8K, reps 0-2) for base and ablit weights "
               "with the draft model and base weights without the draft model"},
    {"tokens": 32000, "floor": 1482, "margin": 0.25, "provisional": False,
     "source": "the release's cold-prefill floor calibration (32K, reps 0-2) for base and ablit weights "
               "with the draft model and base weights without the draft model"},
)
WORDS = "state kernel cache request model stream config review latency budget prefix memory layer rank window gate tool context river library harbor forest".split()


def sized_prompt(url, target):
    rng = random.Random(target)
    corpus = " ".join(rng.choice(WORDS) for _ in range(target * 2))
    nonce = secrets.token_hex(24) + "\n"
    suffix = "\nSummarize this in one sentence."
    # Bounded binary search over characters: no generation during calibration.
    low, high = 0, len(corpus)
    for _ in range(22):
        length = (low + high) // 2
        messages = [{"role": "user", "content": nonce + corpus[:length] + suffix}]
        count = integer(json_call(url + "/tokenize", body(messages), timeout=10).get("count"), "tokenize count", 1)
        if abs(count - target) <= 32:
            return messages, count
        if count < target:
            low = length + 1
        else:
            high = length - 1
        if low > high:
            break
    raise Failure("could not size the prompt within 32 tokens of its target")


def measure(url, messages, expected, timeout=90):
    reply = stream(url, body(messages, max_tokens=3), timeout=timeout)
    if reply["usage"]["prompt_tokens"] != expected:
        raise Failure("usage prompt length disagrees with /tokenize")
    if cached(reply) != 0:
        raise Failure("fresh prompt reused cached tokens; cold prefill was not measured")
    prefill, decode = timing(reply)
    return expected / prefill, max(reply["elapsed_s"], prefill + decode), prefill


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", nargs="?")
    parser.add_argument("--floor-scale", type=float, default=1.0, help="positive multiplier for the table's floors")
    args = parser.parse_args()
    step = "setup"
    try:
        if not math.isfinite(args.floor_scale) or args.floor_scale <= 0:
            raise Failure("--floor-scale must be finite and greater than zero")
        url = endpoint(args.url)
        deadline(230)
        print("Kernel compile telemetry unavailable through the shipped API/log contract; using the elapsed-time outlier rule.", flush=True)
        if any(row["provisional"] for row in FLOOR_TABLE):
            print("PROVISIONAL floors; live calibration is still required.", flush=True)
        print("Run with other clients idle.", flush=True)
        # Size every prompt before warmup so calibration is outside the timed gate.
        prompts = [(row, sized_prompt(url, row["tokens"]), sized_prompt(url, row["tokens"])) for row in FLOOR_TABLE]
        for row, warm, _ in prompts:
            step = "warmup-%d" % row["tokens"]
            deadline(300)
            measure(url, *warm, timeout=300)
            print("PASS  %s" % step, flush=True)
        deadline(230)
        for row, _, probe in prompts:
            step = "cold-%d" % row["tokens"]
            rate, elapsed, prefill = measure(url, *probe)
            floor = row["floor"] * args.floor_scale
            limit = max(10.0, 4 * probe[1] / floor)
            print("%s target_tokens=%d prompt_tokens=%d prefill_s=%.6f rate=%.6f tok/s "
                  "floor=%.1f floor_scale=%g elapsed=%.2fs outlier_limit=%.2fs" %
                  (step, row["tokens"], probe[1], prefill, rate, floor, args.floor_scale, elapsed, limit), flush=True)
            if elapsed > limit:
                raise Failure("post-warmup reply is a compile-sized latency outlier")
            if rate < floor:
                raise Failure("cold prefill rate is below its floor")
        print("PREFILL PASS %d/%d" % (len(FLOOR_TABLE), len(FLOOR_TABLE)))
        return 0
    except ERRORS as exc:
        if step.startswith("warmup-"):
            print("Warmup can compile kernels after a fresh boot; wait for readiness and retry once.", flush=True)
        return failure_exit("PREFILL", step, exc)
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    sys.exit(main())
