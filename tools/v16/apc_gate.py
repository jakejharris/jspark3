#!/usr/bin/env python3
"""Pre-admission prefix-cache (APC) gate for v1.6 with JSPARK3_V16_APC_LRU (S9.7).

Folds `../../docs/OPERATIONS.md` into one boot. Fixtures are rendered token IDs from the
served chat template (Pi-shaped prose and tool turns), frozen to a file so the outgoing and the
new boot run identical arrays. Per cell, in order and under distinct cache salts:

  producer (salt S, 96 forced tokens) -> 3 cold probes (salts C1..C3) -> warm probe (salt S)

The cold probes run while the producer's blocks are cached, so every cold hit above 0 is a
cache-salt isolation failure. Probes force 64 greedy tokens with top-20 logprobs. Message bodies are
one running count, so every slice point has a peaked greedy continuation: random words give flat
next-token distributions whose bf16 ties flip even between identical cold runs (outgoing-boot
capture, 2026-09-24 22:53Z), which leaves parity nothing to compare.

The gate FAILs on: a wrong coordinator priority line, any cold/producer cache hit, a reuse-cell
warm hit other than its expectation (5120/15360), a negative control reusing more than its ceiling
of 2560, foreign traffic, or a parity failure. The controls are the 5079-token accounting ceiling
and a one-token change at 5119, which must never be served from cache past the divergence.
Controls gate a ceiling, not an exact hit: whether a legitimate earlier checkpoint exists depends
on sparse retention at the producer's replay boundary. control-5079's producer boundary is 2560
(hit 2560); control-divergent's is 5120, which the change at 5119 invalidates, so it reuses 0
under either eviction policy (historical launch legacy r2 capture 23:02Z and historical launch, 2026-09-24).

Parity is referenced to this engine's own cold noise, because exact greedy parity is not
satisfiable: identical fresh-salt c1 requests on the shipping boot diverge from each other at
tokens 0-24 with shared-history logprob spreads up to 0.84 nats (outgoing-boot capture r2,
2026-09-24 23:02Z). The warm probe is scored as a fourth cold sample:
  - outlier: its longest agreement with the 3 colds is shorter than every cold/cold agreement.
    With 4 exchangeable runs P(outlier) <= 1/4 per cell; FAIL if more than half the cells are
    outliers (binomial(14, 0.25) >= 8 is ~0.4%). Corrupted cached state is an outlier everywhere.
  - gross: its first token (computed from the cached state plus the fresh prefill) is absent from
    every cold top-20 -> FAIL.
  - spread: the max warm/cold logprob spread on shared histories exceeds max(0.10, 2x the max
    cold/cold spread) -> FAIL.
Checker negative controls mutate the real capture and must FAIL.

With --baseline (the same fixtures captured on the outgoing boot under --expect legacy), the gate
also requires Pi-shaped numbers to be net positive: no reuse cell loses cached tokens, and the
summed warm TTFT over the reuse cells is lower than on the outgoing boot.

--expect finehit (S9.11.3-finehit side boot) keeps the lru identity and every other rule, but
scores hits under 640-token fine-grained lookup (TP3 hash grain gcd(2560, 640)). A producer
registers one partial prompt-tail entry at floor(len/640)*640 when that is not a 2560 multiple;
a probe may reuse it only through its shared prefix and at most len(probe)-4 (Kpool replay
floor). A reuse cell's warm hit must equal its 2560 expectation or that fine expectation (the
coarse value is the partial-replay fallback when no drafter window ends there). A control's
ceiling is its fine expectation: control-5079 may reuse the producer's 4480 tail entry;
control-divergent stays at 2560 because its producer tail (5120) lies past the divergence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import copy
import json
from pathlib import Path
import re
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v16_common import QAError, fetch_remote_logs, metric_delta, metrics_snapshot, sha256_json, write_json  # noqa: E402

BLOCK = 2560
FINE = 640
PRODUCER_TOKENS, PROBE_TOKENS, TOP_K = 96, 64, 20
MIN_TOL = 0.10
SUFFIXES = (4, 37, 639, 640, 641, 2047, 2048, 2049)
APC_LINE_RE = re.compile(r"\[glm53-apc-per-group\] retention_by_group=\S+ \(global=\S+ swa_env=0 "
                         r"eagle_min_exempt=\[([0-9, ]*)\] low_priority=\[([0-9, ]*)\] ")
TOOLS = [{"type": "function", "function": {"name": name, "description": desc, "parameters": {
    "type": "object", "properties": props, "required": list(props)}}}
    for name, desc, props in (
        ("read", "Read a file from the workspace.", {"path": {"type": "string"}}),
        ("bash", "Run a shell command and return its output.", {"command": {"type": "string"}}),
        ("edit", "Replace exact text in a file.", {"path": {"type": "string"}, "old": {"type": "string"},
                                                  "new": {"type": "string"}}),
        ("write", "Write a file.", {"path": {"type": "string"}, "content": {"type": "string"}}))]


# ---------------------------------------------------------------- fixtures

def counting(state: list[int], n: int) -> str:
    """The next n numbers of one running count: a peaked, context-dependent continuation."""
    start = state[0]
    state[0] += n
    return ", ".join(str(k) for k in range(start, start + n)) + ","


def render(base: str, model: str, messages: list[dict], tools: list | None = None) -> list[int]:
    body = {"model": model, "messages": messages, "add_generation_prompt": False}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(base.rstrip("/") + "/tokenize", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    tokens = json.loads(urllib.request.urlopen(req, timeout=60).read())["tokens"]
    if not tokens or not all(type(t) is int and t >= 0 for t in tokens):
        raise QAError("tokenize returned no token ids")
    return tokens


def conversation(base: str, model: str, kind: str, minimum: int) -> list[int]:
    """Pi-shaped rendered conversation of at least `minimum` tokens (synthetic text only)."""
    count = [{"prose": 1000, "tools": 3000, "long": 9000}[kind]]
    system = ("You are an expert coding assistant operating inside pi, a coding agent harness.\n\n"
              "# Project context: build counter log\n" + counting(count, 1300))
    messages = [{"role": "system", "content": system}]
    tools = TOOLS if kind == "tools" else None
    turn = 0
    while True:
        messages.append({"role": "user", "content": ("Continue the counter log: " if kind != "tools"
                                                     else "Check the config and fix the failing test. ") + counting(count, 30)})
        if kind == "tools":
            call_id = f"call_{turn}"
            messages.append({"role": "assistant", "content": "", "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": ("read", "bash")[turn % 2],
                             "arguments": json.dumps({"path": f"src/mod{turn}.py"} if turn % 2 == 0
                                                     else {"command": f"pytest -q tests/test_{turn}.py"})}}]})
            messages.append({"role": "tool", "tool_call_id": call_id, "content": counting(count, 110)})
        messages.append({"role": "assistant", "content": counting(count, 130)})
        turn += 1
        tokens = render(base, model, messages, tools)
        if len(tokens) >= minimum:
            return tokens
        if turn > 200:
            raise QAError(f"{kind} conversation never reached {minimum} tokens")


def cells_from(bases: dict[str, list[int]]) -> list[dict]:
    prose, tools, long = bases["prose"], bases["tools"], bases["long"]
    cells = [
        {"id": "pi-prose", "producer": prose[:5639], "probe": prose[:6248], "expected_hit": 5120},
        {"id": "pi-tools", "producer": tools[:5639], "probe": tools[:6248], "expected_hit": 5120},
        {"id": "large-jump", "producer": long[:15845], "probe": long[:16106], "expected_hit": 15360},
        {"id": "same-prompt", "producer": prose[:6248], "probe": prose[:6248], "expected_hit": 5120},
    ]
    cells += [{"id": f"suffix-{s}", "producer": tools[:5120 + s], "probe": tools[:5120 + s], "expected_hit": 5120}
              for s in SUFFIXES]
    # Negative controls: no 5120 checkpoint can exist for either probe.
    cells.append({"id": "control-5079", "producer": prose[:5079], "probe": prose[:5157], "expected_hit": 2560,
                  "control": "accounting"})
    alt = next(t for t in tools[5000:5119] if t != tools[5119])
    cells.append({"id": "control-divergent", "producer": tools[:5639],
                  "probe": tools[:5119] + [alt] + tools[5120:6248], "expected_hit": 2560, "control": "divergent"})
    return cells


def shared_prefix(a: list[int], b: list[int]) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def fine_expected(cell: dict) -> int:
    """Largest hit 640-token lookup can serve from the producer's partial tail entry, else the 2560 value."""
    tail = len(cell["producer"]) // FINE * FINE
    usable = (tail % BLOCK and tail <= shared_prefix(cell["producer"], cell["probe"])
              and tail <= len(cell["probe"]) - 4)
    return max(cell["expected_hit"], tail if usable else 0)


def allowed_hits(cell: dict, expect: str) -> set[int]:
    return {cell["expected_hit"], fine_expected(cell)} if expect == "finehit" else {cell["expected_hit"]}


def validate_cells(cells: list[dict]) -> None:
    if not cells or len({c["id"] for c in cells}) != len(cells):
        raise QAError("fixture cells missing or duplicated")
    for cell in cells:
        for key in ("producer", "probe"):
            if not cell[key] or not all(type(t) is int and t >= 0 for t in cell[key]):
                raise QAError(f"{cell['id']}: bad {key} token ids")
        hit = cell["expected_hit"]
        if hit <= 0 or hit % BLOCK or hit > shared_prefix(cell["producer"], cell["probe"]) or hit > len(cell["probe"]) - 4:
            raise QAError(f"{cell['id']}: expected hit {hit} impossible for these arrays")
    by_id = {c["id"]: c for c in cells}
    if shared_prefix(by_id["control-divergent"]["producer"], by_id["control-divergent"]["probe"]) != 5119:
        raise QAError("divergent control must differ at token 5119")


# ---------------------------------------------------------------- capture

def body(model: str, prompt: list[int], salt: str, output: int, logprobs: bool) -> dict:
    value = {"model": model, "prompt": list(prompt), "cache_salt": salt, "max_tokens": output,
             "min_tokens": output, "ignore_eos": True, "temperature": 0, "top_p": 1, "top_k": -1,
             "seed": 17, "add_special_tokens": False, "return_token_ids": True, "stream": True,
             "stream_options": {"include_usage": True}}
    if logprobs:
        value["logprobs"] = TOP_K
    return value


def consume(lines, started: float, now=time.monotonic) -> dict:
    ids, strs, lps, tops = [], [], [], []
    request_ids = set()
    usage, finish, first, done = None, None, None, False
    for raw in lines:
        line = raw.decode().strip() if isinstance(raw, bytes) else raw.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        event = json.loads(data)
        if event.get("id"):
            request_ids.add(event["id"])
        if "error" in event:
            raise QAError(f"server error: {event['error']}")
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices", []):
            new = choice.get("token_ids") or []
            if new and first is None:
                first = now() - started
            ids.extend(new)
            lp = choice.get("logprobs") or {}
            strs.extend(lp.get("tokens") or [])
            lps.extend(lp.get("token_logprobs") or [])
            tops.extend(lp.get("top_logprobs") or [])
            finish = choice.get("finish_reason") or finish
    if not done or first is None or usage is None or finish != "length":
        raise QAError("incomplete stream (DONE/ids/usage/length)")
    if len(ids) != usage.get("completion_tokens"):
        raise QAError("token ids disagree with usage")
    return {"token_ids": ids, "token_strs": strs, "logprobs": lps, "top_logprobs": tops,
            "request_ids": sorted(request_ids),
            "usage": usage, "ttft_s": round(first, 4), "e2e_s": round(now() - started, 4)}


def request(base: str, payload: dict) -> dict:
    get = lambda d, name: sum(v for k, v in d.items() if k.split("{", 1)[0] == "vllm:" + name)  # noqa: E731
    metrics_before_at = datetime.now(timezone.utc).isoformat()
    before = metrics_snapshot(base)
    req = urllib.request.Request(base.rstrip("/") + "/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    started = time.monotonic()
    request_started_at = datetime.now(timezone.utc).isoformat()
    with urllib.request.urlopen(req, timeout=900) as resp:
        result = consume(resp, started)
    response_ended_at = datetime.now(timezone.utc).isoformat()
    time.sleep(0.3)
    delta = metric_delta(before, metrics_snapshot(base))
    result.update({"salt": payload["cache_salt"], "prompt_tokens": len(payload["prompt"]),
                   "request_started_at": request_started_at, "response_ended_at": response_ended_at,
                   "metrics_before_at": metrics_before_at, "metrics_after_at": datetime.now(timezone.utc).isoformat(),
                   "metrics_delta": delta,
                   "request_sha256": sha256_json(payload),
                   "cached_tokens": int(get(delta, "prefix_cache_hits_total")),
                   "requests_delta": int(get(delta, "request_success_total")),
                   "prefill_s": round(get(delta, "request_prefill_time_seconds_sum"), 4)})
    usage_cached = (result["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
    if result["usage"].get("prompt_tokens") != len(payload["prompt"]):
        raise QAError("prompt token accounting drift")
    if usage_cached is not None and usage_cached != result["cached_tokens"]:
        raise QAError(f"usage cached {usage_cached} != metrics {result['cached_tokens']}")
    if payload.get("logprobs") and not (len(result["logprobs"]) == len(result["top_logprobs"])
                                        == len(result["token_ids"])):
        raise QAError("logprobs missing or misaligned (fail closed)")
    return result


def capture(base: str, model: str, cells: list[dict], run: str) -> list[dict]:
    rows = []
    for cell in cells:
        shared = f"{run}-{cell['id']}-shared"
        row = {"id": cell["id"], "producer": request(base, body(model, cell["producer"], shared, PRODUCER_TOKENS, False))}
        row["cold"] = [request(base, body(model, cell["probe"], f"{run}-{cell['id']}-cold-{i}", PROBE_TOKENS, True))
                       for i in range(3)]
        row["warm"] = request(base, body(model, cell["probe"], shared, PROBE_TOKENS, True))
        rows.append(row)
    return rows


# ---------------------------------------------------------------- analysis

def compare(warm: dict, cold: dict) -> dict:
    """Shared greedy history up to the first divergence; logprob spread on it; near-tie at the divergence.

    The tie gap is how far the other run's token sits below this run's choice, read from either
    run's top-k (the smaller when both have it). Neither top-k holding the other's token is not
    a tie (infinite gap)."""
    d = shared_prefix(warm["token_ids"], cold["token_ids"])
    spread = max((abs(warm["logprobs"][i] - cold["logprobs"][i]) for i in range(d)), default=0.0)
    gap = None
    if d < min(len(warm["token_ids"]), len(cold["token_ids"])):
        gaps = [run["logprobs"][d] - (run["top_logprobs"][d] or {})[other["token_strs"][d]]
                for run, other in ((cold, warm), (warm, cold))
                if other["token_strs"][d] in (run["top_logprobs"][d] or {})]
        gap = min(gaps) if gaps else float("inf")
    return {"diverge_at": d, "spread": round(spread, 6), "tie_gap": gap if gap is None else round(gap, 6)}


def identity_ok(lines: list, expect: str) -> bool:
    """Exactly one coordinator line; lru = empty low-priority set, legacy = the drafter group."""
    if len(lines) != 1 or not lines[0][0]:
        return False
    exempt, low = lines[0]
    return low == "" if expect in ("lru", "finehit") else low == exempt


def parity(rows: list[dict]) -> dict:
    """Warm probe as a fourth cold sample (see module docstring)."""
    cells, cold_spread, warm_spread, outliers, gross = {}, 0.0, 0.0, [], []
    for row in rows:
        colds, warm = row["cold"], row["warm"]
        pairs = [compare(colds[i], colds[j]) for i, j in ((1, 0), (2, 0), (2, 1))]
        against = [compare(warm, cold) for cold in colds]
        cold_spread = max([cold_spread] + [p["spread"] for p in pairs])
        warm_spread = max([warm_spread] + [a["spread"] for a in against])
        best, worst_pair = max(a["diverge_at"] for a in against), min(p["diverge_at"] for p in pairs)
        label = "IDENTICAL" if best == PROBE_TOKENS else "WITHIN_NOISE"
        if best < worst_pair:
            label = "OUTLIER"
            outliers.append(row["id"])
        if not any(warm["token_strs"][0] in (cold["top_logprobs"][0] or {}) for cold in colds):
            label = "GROSS"
            gross.append(row["id"])
        cells[row["id"]] = {"parity": label, "warm_agreement": [a["diverge_at"] for a in against],
                            "cold_agreement": [p["diverge_at"] for p in pairs],
                            "warm_spread": max(a["spread"] for a in against)}
    limit = max(MIN_TOL, 2 * cold_spread)
    findings = [f"{label}: warm first token absent from every cold top-{TOP_K}" for label in gross]
    if len(outliers) * 2 > len(rows):
        findings.append(f"warm is the outlier in {len(outliers)}/{len(rows)} cells: {outliers}")
    if warm_spread > limit:
        findings.append(f"warm/cold logprob spread {warm_spread:.3f} > {limit:.3f} (2x cold/cold)")
    return {"cells": cells, "findings": findings, "outliers": outliers, "cold_spread": round(cold_spread, 6),
            "warm_spread": round(warm_spread, 6), "spread_limit": round(limit, 6)}


def analyze(doc: dict) -> dict:
    cells, rows, expect = doc["cells"], doc["rows"], doc["expect"]
    findings, report = [], {}
    if [r["id"] for r in rows] != [c["id"] for c in cells]:
        return {"verdict": "FAIL", "findings": ["incomplete fixture coverage"], "cells": {}}
    if not identity_ok(doc.get("apc_identity") or [], expect):
        findings.append(f"coordinator priority line {doc.get('apc_identity')} does not show the {expect} policy")
    sent = sum(2 + len(r["cold"]) for r in rows)
    if doc.get("requests_delta") != sent:
        findings.append(f"foreign traffic: server finished {doc.get('requests_delta')} requests, gate sent {sent}")
    par = parity(rows)
    findings += par["findings"]
    for cell, row in zip(cells, rows):
        label, warm = cell["id"], row["warm"]
        hits = {"producer": row["producer"]["cached_tokens"], "cold": [c["cached_tokens"] for c in row["cold"]],
                "warm": warm["cached_tokens"]}
        if hits["producer"] or any(hits["cold"]):
            findings.append(f"{label}: cache hit across salts {hits} (isolation)")
        allowed = allowed_hits(cell, expect)
        if expect in ("lru", "finehit") and not cell.get("control") and hits["warm"] not in allowed:
            findings.append(f"{label}: warm hit {hits['warm']} != expected {'/'.join(map(str, sorted(allowed)))}")
        if cell.get("control") and hits["warm"] > max(allowed):
            findings.append(f"{label}: negative control reused {hits['warm']} > {max(allowed)} (false reuse)")
        report[label] = {"hits": hits, "expected_hit": cell["expected_hit"], **par["cells"][label],
                         **({"fine_expected_hit": fine_expected(cell),
                             "fine_hit": hits["warm"] == fine_expected(cell) > cell["expected_hit"]}
                            if expect == "finehit" else {}),
                         "warm_ttft_s": warm["ttft_s"], "cold_ttft_s": [c["ttft_s"] for c in row["cold"]],
                         "warm_prefill_s": warm["prefill_s"], "cold_prefill_s": [c["prefill_s"] for c in row["cold"]]}
    return {"verdict": "FAIL" if findings else "PASS", "findings": findings, "outliers": par["outliers"],
            "cold_spread": par["cold_spread"], "warm_spread": par["warm_spread"], "spread_limit": par["spread_limit"],
            "cells": report}


def net_positive(baseline: dict, candidate: dict, fixtures_sha256: str) -> dict:
    """Outgoing (legacy) boot vs this boot on identical fixtures: reuse cells only."""
    findings = []
    if baseline.get("expect") != "legacy" or baseline.get("fixtures_sha256") != fixtures_sha256:
        return {"verdict": "FAIL", "findings": ["baseline is not a legacy capture of these fixtures"]}
    before, after = baseline["analysis"]["cells"], candidate["cells"]
    reuse = [label for label in after if not label.startswith("control-")]
    cells = {}
    for label in reuse:
        a, b = before[label], after[label]
        cells[label] = {"hit": [a["hits"]["warm"], b["hits"]["warm"]], "warm_ttft_s": [a["warm_ttft_s"], b["warm_ttft_s"]]}
        if b["hits"]["warm"] < a["hits"]["warm"]:
            findings.append(f"{label}: reuse fell {a['hits']['warm']} -> {b['hits']['warm']}")
    total = [round(sum(before[label]["warm_ttft_s"] for label in reuse), 3),
             round(sum(after[label]["warm_ttft_s"] for label in reuse), 3)]
    if total[1] >= total[0]:
        findings.append(f"summed warm TTFT not lower: {total[0]} s -> {total[1]} s")
    return {"verdict": "FAIL" if findings else "PASS", "findings": findings, "warm_ttft_sum_s": total,
            "cached_tokens_sum": [sum(before[label]["hits"]["warm"] for label in reuse),
                                  sum(after[label]["hits"]["warm"] for label in reuse)], "cells": cells}


def checker_controls(doc: dict) -> dict:
    """Mutate the real capture; a checker that still passes is broken."""
    def mutated(fn):
        broken = copy.deepcopy(doc)
        fn(broken)
        return analyze(broken)["verdict"]

    def token(d):
        warm = d["rows"][0]["warm"]
        warm["token_ids"][0] += 1
        warm["token_strs"][0] = "\x00not-a-top-token"
        warm["logprobs"][0] = -0.01
        warm["top_logprobs"][0] = {"\x00not-a-top-token": -0.01}   # a confident, different token

    def logprob(d):
        row = next(r for r in d["rows"] if r["warm"]["token_ids"][:1] == r["cold"][0]["token_ids"][:1])
        row["warm"]["logprobs"][0] -= 5.0

    def outlier_everywhere(d):
        for r in d["rows"]:
            warm = r["warm"]
            warm["token_ids"][1:] = [10**9 + i for i in range(len(warm["token_ids"]) - 1)]
            warm["token_strs"][1:] = ["\x00diverged"] * (len(warm["token_strs"]) - 1)

    def warm_hit(d):
        d["rows"][0]["warm"]["cached_tokens"] -= BLOCK

    def cold_hit(d):
        d["rows"][0]["cold"][1]["cached_tokens"] = BLOCK

    def false_reuse(d):
        next(r for r in d["rows"] if r["id"] == "control-divergent")["warm"]["cached_tokens"] = 5120

    def identity(d):
        d["apc_identity"] = [["6", "6"]] if d["expect"] in ("lru", "finehit") else [["6", ""]]

    def foreign(d):
        d["requests_delta"] += 1

    controls = [("warm_token", token), ("warm_logprob", logprob), ("warm_outlier", outlier_everywhere),
                ("cold_hit", cold_hit),
                ("false_reuse", false_reuse), ("identity", identity), ("foreign_traffic", foreign)]
    if doc["expect"] in ("lru", "finehit"):
        controls.append(("warm_hit", warm_hit))
    return {name: mutated(fn) for name, fn in controls}


# ---------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--container-prefix", default="jspark3-v16-rank")
    parser.add_argument("--fixtures", type=Path, required=True,
                        help="frozen rendered fixtures; created from the served template if absent")
    parser.add_argument("--expect", choices=("lru", "legacy", "finehit"), default="lru",
                        help="lru gates the new policy; legacy records the outgoing boot (mechanism control)")
    parser.add_argument("--baseline", type=Path,
                        help="outgoing boot's legacy report; adds the net-positive Pi-shaped check")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.out.exists():
        print(f"REFUSED: {args.out} exists", file=sys.stderr)
        return 2
    try:
        baseline = json.loads(args.baseline.read_text()) if args.baseline else None
        if args.fixtures.exists():
            fixtures = json.loads(args.fixtures.read_text())
        else:
            bases = {"prose": conversation(args.base_url, args.model, "prose", 6400),
                     "tools": conversation(args.base_url, args.model, "tools", 7300),
                     "long": conversation(args.base_url, args.model, "long", 16300)}
            fixtures = {"schema": "jspark3-v16-apc-fixtures/1", "model": args.model, "cells": cells_from(bases)}
            fixtures["cells_sha256"] = sha256_json(fixtures["cells"])
            write_json(args.fixtures, fixtures)
        if fixtures.get("cells_sha256") != sha256_json(fixtures["cells"]):
            raise QAError("fixture hash mismatch")
        validate_cells(fixtures["cells"])
        logs, sources = fetch_remote_logs(args.env_file, args.container_prefix)
        identity = [list(m) for m in APC_LINE_RE.findall(re.sub(r"\x1b\[[0-9;]*m", "", logs))]
        get = lambda d, name: sum(v for k, v in d.items() if k.split("{", 1)[0] == "vllm:" + name)  # noqa: E731
        before = metrics_snapshot(args.base_url)
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rows = capture(args.base_url, args.model, fixtures["cells"], f"apc-{time.time_ns()}")
        time.sleep(0.5)
        total = metric_delta(before, metrics_snapshot(args.base_url))
    except (QAError, OSError, ValueError, KeyError, StopIteration) as exc:
        print(f"REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    doc = {"schema": "jspark3-v16-apc-gate/1", "base_url": args.base_url, "expect": args.expect, "started": started,
           "fixtures": str(args.fixtures), "fixtures_sha256": fixtures["cells_sha256"], "log_sources": sources,
           "apc_identity": identity, "requests_delta": int(get(total, "request_success_total")),
           "cells": fixtures["cells"], "rows": rows}
    result = analyze(doc)
    controls = checker_controls(doc)
    if any(verdict != "FAIL" for verdict in controls.values()):
        result["findings"].append(f"checker negative control passed: {controls}")
        result["verdict"] = "FAIL"
    if baseline is not None:
        net = net_positive(baseline, result, fixtures["cells_sha256"])
        result["net_positive"] = net
        if net["verdict"] != "PASS":
            result["findings"].extend(f"net: {item}" for item in net["findings"])
            result["verdict"] = "FAIL"
    doc.update({"analysis": result, "checker_controls": controls})
    doc.pop("cells")
    doc["receipt_sha256"] = sha256_json(doc)
    write_json(args.out, doc)
    print(json.dumps({"verdict": result["verdict"], "expect": args.expect, "outliers": result["outliers"],
                      "spread": [result["warm_spread"], result["spread_limit"]],
                      "hits": {k: v["hits"]["warm"] for k, v in result["cells"].items()},
                      "parity": {k: v["parity"] for k, v in result["cells"].items()},
                      "net_warm_ttft_sum_s": (result.get("net_positive") or {}).get("warm_ttft_sum_s"),
                      "findings": result["findings"][:6]}))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
