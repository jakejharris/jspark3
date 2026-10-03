#!/usr/bin/env python3
"""Frozen class-S quality ruler. See README.md for the scoring wire contract."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import re
import resource
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

DOMAINS = {
    "code": [
        ("Python review: explain why a mutable default argument can leak state between calls.", "A default list is created once when the function is defined, so later calls can reuse mutations."),
        ("Explain the invariant of binary search over a sorted array.", "All possible matches remain inside the current search interval after each comparison."),
        ("What does a transaction rollback do after a failed insert?", "Rollback discards the transaction's uncommitted changes and restores its earlier database state."),
        ("Explain why a lock around only the final write does not make read-modify-write atomic.", "Another worker can change the value between the read and the protected write."),
    ],
    "prose": [
        ("Continue this short note about a city library opening on Sunday.", "The doors opened at noon, and families found the reading room quiet and bright."),
        ("Write a neutral summary of a garden after a week of rain.", "The soil remained damp, while new leaves appeared on several plants near the path."),
        ("Continue a travel journal about a delayed train without making up a cause.", "We waited on the platform and checked the board again before choosing a later connection."),
        ("Write a plain-language sentence about a repaired footbridge.", "The bridge is open again, and people can cross the stream on the marked path."),
    ],
    "reasoning": [
        ("If all blue cards are numbered and card seven is blue, what follows?", "Card seven is numbered because it belongs to the set of blue cards."),
        ("Mira has three apples and gives one to Sol. How many remain?", "Mira has two apples after giving one of the three away."),
        ("A meeting starts at 09:20 and lasts 45 minutes. When does it end?", "The meeting ends at 10:05."),
        ("A box holds five parts. Two boxes are full and one holds three. Count the parts.", "There are thirteen parts in total: five plus five plus three."),
    ],
    "tool_json": [
        ("Return a tool call as JSON for add(a=2,b=3).", '{"name":"add","arguments":{"a":2,"b":3}}'),
        ("Return a tool call as JSON for lookup(key='map').", '{"name":"lookup","arguments":{"key":"map"}}'),
        ("Return a tool call as JSON for schedule(day='Friday',hour=14).", '{"name":"schedule","arguments":{"day":"Friday","hour":14}}'),
        ("Return a tool call as JSON for search(query='orbital period').", '{"name":"search","arguments":{"query":"orbital period"}}'),
    ],
    "long_context": [
        ("In a long log, a checkpoint records the key Cedar-719.", "The checkpoint key is Cedar-719."),
        ("In a long report, the fallback port is 48231.", "The fallback port is 48231."),
        ("A long memo says the archive label is VIOLET-RIVER-42.", "The archive label is VIOLET-RIVER-42."),
        ("In a long ledger, the closing balance is 691 units.", "The closing balance is 691 units."),
    ],
}
CODING = [
    ("Define function solve(nums) returning the sum of even integers in nums. Return only Python code.", "solve", [([[2, 3, 4]], 6), ([[-2, 1, 0]], -2), ([[]], 0)]),
    ("Define function solve(text) returning the number of vowels a,e,i,o,u ignoring case. Return only Python code.", "solve", [(["AeIoU"], 5), (["xyz"], 0), (["Ocean"], 3)]),
    ("Define function solve(nums) returning a new list with adjacent duplicate values removed. Return only Python code.", "solve", [([[1, 1, 2, 1]], [1, 2, 1]), ([[]], []), ([[3, 3]], [3])]),
    ("Define function solve(n) returning the nth Fibonacci number, with F(0)=0 and F(1)=1. Return only Python code.", "solve", [([0], 0), ([1], 1), ([8], 21)]),
]
TOOLS = [
    ("add", {"a": 17, "b": -4}, "Add 17 and -4 using the add tool."),
    ("lookup", {"key": "copper"}, "Look up the key copper using the lookup tool."),
    ("schedule", {"day": "Friday", "hour": 14}, "Schedule Friday at hour 14 using the schedule tool."),
    ("search", {"query": "orbital period"}, "Search for orbital period using the search tool."),
]
NEEDLES = [(32768, ["KITE-418"]), (32768, ["EMBER-62", "MAPLE-903"]),
           (131072, ["KITE-418"]), (131072, ["EMBER-62", "MAPLE-903"])]
MARGINS = {"overall_nll_upper": 0.010, "domain_nll_upper": 0.020,
           "coding_min": 3, "tool_min": 4, "recall_min": 3}
VERSION = "quality-v2-20261001"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


FIXTURE_HASH = digest({"domains": DOMAINS, "coding": CODING, "tools": TOOLS, "needles": NEEDLES, "margins": MARGINS, "version": VERSION})


def post(base, path, body, timeout):
    data = json.dumps(body, separators=(",", ":")).encode()
    request = Request(base.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def score(base, model, domain, index, prefix, continuation, baseline, timeout, tokenizer):
    previous = baseline.get("scores", {}).get(f"{domain}-{index}") if baseline else None
    if previous:
        ids, start = previous["token_ids"], previous["score_start"]
    else:
        leading = tokenizer.encode(prefix, add_special_tokens=False).ids
        ids = tokenizer.encode(prefix + "\n" + continuation, add_special_tokens=False).ids
        start = 0
        for left, right in zip(leading, ids):
            if left != right:
                break
            start += 1
        # The newline and first target token may merge into one token. Score from
        # the first changed token; the baseline freezes this exact boundary.
        if start == len(ids):
            raise ValueError("scoring continuation produced no tokens")
    reply = post(base, "/v1/quality/score", {"model": model, "token_ids": ids, "score_start": start}, timeout)
    got_ids = reply["token_ids"]
    got_start = reply["score_start"]
    nll = reply["nll"]
    if not isinstance(got_ids, list) or not all(type(x) is int and x >= 0 for x in got_ids):
        raise ValueError("score response needs nonnegative integer token_ids")
    if got_ids != ids or got_start != start or not 0 < start < len(ids) or len(nll) != len(ids) - start:
        raise ValueError("score response token IDs, score_start, or NLL count mismatch")
    if not all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in nll):
        raise ValueError("score response has invalid NLL")
    return {"token_ids": ids, "score_start": start, "nll": nll, "mean_nll": sum(nll)/len(nll)}


def chat(base, model, prompt, max_tokens, timeout, tools=None):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "top_p": 1,
            "chat_template_kwargs": {"reasoning_effort": "low"}}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "required"
    reply = post(base, "/v1/chat/completions", body, timeout)
    choice = reply["choices"][0]
    if choice.get("finish_reason") in (None, "length"):
        raise ValueError("chat response missing finish or truncated")
    return choice["message"], reply.get("usage", {}), choice["finish_reason"]


def bare_tool_call(message, finish, name, arguments):
    calls = message.get("tool_calls") or []
    if finish != "tool_calls" or message.get("content") not in (None, "") or len(calls) != 1:
        return False
    try:
        if calls[0]["type"] != "function":
            return False
        call = calls[0]["function"]
        return call["name"] == name and json.loads(call["arguments"]) == arguments
    except (KeyError, TypeError, ValueError):
        return False


def checked_code(raw, name, cases):
    match = re.search(r"```(?:python)?\s*(.*?)```", raw, re.I | re.S)
    code = match.group(1) if match else raw
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.AugAssign,
               ast.For, ast.If, ast.While, ast.Break, ast.Continue, ast.Pass, ast.Expr, ast.Name, ast.Load,
               ast.Store, ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.BinOp, ast.UnaryOp, ast.BoolOp,
               ast.Compare, ast.Call, ast.Subscript, ast.Slice, ast.ListComp, ast.GeneratorExp, ast.comprehension, ast.IfExp,
               ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.Div, ast.Pow, ast.USub, ast.UAdd,
               ast.And, ast.Or, ast.Not, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In,
               ast.NotIn, ast.Is, ast.IsNot)
    if any(not isinstance(node, allowed) for node in ast.walk(tree)):
        return False
    if any(isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in
           {"range", "len", "sum", "min", "max", "enumerate", "abs", "int", "list", "str", "append"}) for node in ast.walk(tree)):
        return False
    if any(isinstance(node, ast.Name) and node.id.startswith("__") for node in ast.walk(tree)):
        return False
    runner = "import json,sys\nns={'__builtins__':{k:getattr(__builtins__,k) for k in ('range','len','sum','min','max','enumerate','abs','int','list','str')}}\n"
    runner += f"exec(compile({code!r},'<answer>','exec'),ns)\n"
    runner += f"f=ns[{name!r}]\nfor args,want in json.loads(sys.stdin.read()):\n assert f(*args)==want\n"
    def limit():
        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        resource.setrlimit(resource.RLIMIT_AS, (256 << 20, 256 << 20))
    try:
        p = subprocess.run([sys.executable, "-I", "-c", runner], input=json.dumps(cases), text=True,
                           capture_output=True, timeout=4, preexec_fn=limit)
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def token_count(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def recall_prompt(tokenizer, target, keys):
    # Frozen arithmetic and text. Places needles at separated points, then pads to target.
    record = "Log entry 0000: routine status; no checkpoint key or archive value here.\n"
    pieces = ["Read the following log. Answer with only the requested key or keys, separated by commas.\n"]
    marks = [0.15, 0.50, 0.85][:len(keys)]
    def pad_to(goal, suffix=""):
        prefix = "".join(pieces)
        lo, hi = 0, max(1, (goal - token_count(tokenizer, prefix + suffix)) // 8 + 100)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if token_count(tokenizer, prefix + record * mid + suffix) <= goal:
                lo = mid
            else:
                hi = mid - 1
        pieces.append(record * lo)
    for i, (key, fraction) in enumerate(zip(keys, marks)):
        pad_to(int(target * fraction))
        pieces.append(f"Log entry {i + 1}: checkpoint key = {key}.\n")
    question = "\nWhat are the checkpoint keys in their original order? Answer only the keys."
    pad_to(target - 32, question)
    prompt = "".join(pieces) + question
    count = token_count(tokenizer, prompt)
    if abs(count - target) > target * .01:
        raise ValueError(f"recall prompt target {target} actual {count}")
    return prompt, count


def long_nll_prefix(tokenizer, fact):
    record = "Observation 0000: ordinary archival material with no checkpoint value.\n"
    start = fact + "\n"
    end = "\nSummarize the checkpoint in one sentence: "
    target = 32768
    lo, hi = 0, target // 4
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if token_count(tokenizer, start + record * mid + end) <= target:
            lo = mid
        else:
            hi = mid - 1
    prefix = start + record * lo + end
    if abs(token_count(tokenizer, prefix) - target) > target * .01:
        raise ValueError("long-context NLL prefix outside 32k tolerance")
    return prefix


def upper(means):
    rng = random.Random(72191)
    samples = sorted(sum(rng.choice(means) for _ in means)/len(means) for _ in range(5000))
    return samples[4749]


def verdict(result, baseline):
    if baseline is None:
        return {"status": "BASELINE_CAPTURED"}
    differences = {domain: [result["scores"][f"{domain}-{i}"]["mean_nll"] - baseline["scores"][f"{domain}-{i}"]["mean_nll"] for i in range(4)] for domain in DOMAINS}
    domain_upper = {domain: upper(values) for domain, values in differences.items()}
    all_upper = upper([x for values in differences.values() for x in values])
    b = baseline["checks"]
    c = result["checks"]
    checks = {
        "overall_nll": all_upper <= MARGINS["overall_nll_upper"],
        "domain_nll": all(x <= MARGINS["domain_nll_upper"] for x in domain_upper.values()),
        "coding": sum(c["coding"]) >= max(MARGINS["coding_min"], sum(b["coding"])) and all(not bp or cp for bp, cp in zip(b["coding"], c["coding"])),
        "tools": sum(c["tools"]) == MARGINS["tool_min"],
        "recall": sum(c["recall"]) >= max(MARGINS["recall_min"], sum(b["recall"])) and all(not bp or cp for bp, cp in zip(b["recall"], c["recall"])),
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "overall_upper": all_upper, "domain_upper": domain_upper, "domain_deltas": differences}


def run(args):
    baseline = json.loads(Path(args.baseline).read_text()) if args.baseline else None
    harness_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.mode == "candidate" and baseline is None:
        raise ValueError("candidate mode requires --baseline")
    if baseline and (baseline["fixture_hash"] != FIXTURE_HASH or baseline["model"] != args.model
                     or baseline.get("harness_sha256") != harness_sha256):
        raise ValueError("baseline fixture, harness, or model mismatch")
    from tokenizers import Tokenizer
    tokenizer_sha256 = hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest()
    if baseline and baseline.get("tokenizer_sha256") != tokenizer_sha256:
        raise ValueError("baseline tokenizer hash mismatch")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    began = time.monotonic()
    result = {"version": VERSION, "fixture_hash": FIXTURE_HASH, "harness_sha256": harness_sha256, "model": args.model,
              "mode": args.mode, "tokenizer_sha256": tokenizer_sha256, "scores": {}, "checks": {"coding": [], "tools": [], "recall": []},
              "recall_tokens": [], "elapsed_seconds": None}
    def deadline():
        remaining = args.budget_seconds - (time.monotonic() - began)
        if remaining <= 0:
            raise TimeoutError("quality budget exceeded")
        return min(args.request_timeout, remaining)
    for domain, examples in DOMAINS.items():
        for i, (prefix, continuation) in enumerate(examples):
            deadline()
            if domain == "long_context":
                prefix = long_nll_prefix(tokenizer, prefix)
            result["scores"][f"{domain}-{i}"] = score(args.url, args.model, domain, i, prefix, continuation, baseline, deadline(), tokenizer)
    for prompt, name, cases in CODING:
        deadline()
        message, _, _ = chat(args.url, args.model, prompt, 384, deadline())
        result["checks"]["coding"].append(checked_code(message.get("content") or "", name, cases))
    tool_specs = [{"type": "function", "function": {"name": name, "description": "Fixed quality probe",
                 "parameters": {"type": "object", "properties": {key: {"type": "integer" if type(value) is int else "string"} for key, value in arguments.items()}, "required": list(arguments)}}} for name, arguments, _ in TOOLS]
    for name, arguments, prompt in TOOLS:
        deadline()
        message, _, finish = chat(args.url, args.model, prompt, 128, deadline(), tool_specs)
        result["checks"]["tools"].append(bare_tool_call(message, finish, name, arguments))
    for target, keys in NEEDLES:
        deadline()
        prompt, count = recall_prompt(tokenizer, target, keys)
        message, usage, _ = chat(args.url, args.model, prompt, 64, deadline())
        answer = (message.get("content") or "").strip()
        expected = ", ".join(keys)
        result["checks"]["recall"].append(answer == expected)
        server_count = usage.get("prompt_tokens")
        if isinstance(server_count, int) and server_count > 0 and abs(server_count - count) > target * .01:
            raise ValueError(f"server/local tokenizer count mismatch: {server_count} vs {count}")
        result["recall_tokens"].append({"target": target, "local": count, "server": server_count})
    deadline()
    result["elapsed_seconds"] = round(time.monotonic() - began, 3)
    result["verdict"] = verdict(result, baseline)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"{result['verdict']['status']} elapsed={result['elapsed_seconds']}s coding={sum(result['checks']['coding'])}/4 tools={sum(result['checks']['tools'])}/4 recall={sum(result['checks']['recall'])}/4 output={args.output}")
    return 0 if result["verdict"]["status"] in ("PASS", "BASELINE_CAPTURED") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "candidate"], required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8002")
    parser.add_argument("--model", default="glm53")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget-seconds", type=int, default=1800)
    parser.add_argument("--request-timeout", type=int, default=180)
    raise SystemExit(run(parser.parse_args()))
