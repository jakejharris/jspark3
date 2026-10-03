#!/usr/bin/env python3
"""Check TensorFold's exactness claim on your own boot: each reply drafted must equal the same request with "draft": false.

"draft": false makes TensorFold decode serially (one token per round, no drafter, no MTP) from a fresh prefill. Every
prompt runs twice on the same serving start, greedy (temperature 0), non-streaming, and the two replies must have the
same token ids and the same text. The check also proves the drafted reply really drafted (fewer verify rounds than
tokens) and the reference really did not.

  python3 scripts/exactness.py [--url http://127.0.0.1:8002] [--out exactness.json]

Exit 0 PASS (every pair identical, drafting proven), 20 REJECT (a pair differs: the first differing token is printed),
21 INVALID (missing evidence, a failed request, unproven drafting, or an unverifiable serving start).
This prompt set does not reproduce the published exactness measurements. Standard library only.
"""
import argparse, hashlib, json, math, re, sys, time, urllib.request

PROMPTS = [
    ("code", "Write a Python function that merges overlapping intervals, with a docstring and two doctests.", 512),
    ("prose", "Write three paragraphs about the history of the lighthouse at Alexandria.", 512),
    ("structured", "Return a JSON array of 12 objects with fields id (int), name (string) and tags (array of 2 strings) "
                   "describing kitchen tools. JSON only.", 512),
    # about 30k prompt tokens: exercises the long-context path and chunked prefill
    ("long", "Here is a ledger:\n" + "".join(f"Entry {i}: item {i * 7919 % 1000:03d} moved from shelf {i % 17} to shelf "
                                              f"{(i * 5) % 23}.\n" for i in range(1, 1601))
             + "\nWhich shelf did entry 1234 move to, and what item was it? Then summarise the ledger's pattern.", 384),
]


def chat(url, content, max_tokens, **extra):
    """Send one greedy, non-streaming chat request and return the reply with TensorFold's decode evidence."""
    body = {"model": "glm53", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "stream": False, "return_token_ids": True, "chat_template_kwargs": {"reasoning_effort": "low"}, **extra}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.monotonic()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    choice, stats = r["choices"][0], r.get("tensorfold") or {}
    message = choice["message"]
    # Reasoning and answer are compared together; NUL keeps their boundary visible.
    text = (message.get("reasoning_content") or "") + "\x00" + (message.get("content") or "")
    return {"ids": stats.get("token_ids"), "token_sha": stats.get("token_sha"), "text": text,
            "tokens": r["usage"]["completion_tokens"], "finish": choice.get("finish_reason"),
            "drafts": stats.get("drafts"), "rounds": stats.get("rounds"), "policy": stats.get("policy"),
            "seconds": round(time.monotonic() - t, 2)}


def evidence_present(reply):
    """True when the reply carries complete, well-typed token evidence and decode stats."""
    return (isinstance(reply["ids"], list) and bool(reply["ids"])
            and all(type(token) is int for token in reply["ids"])
            and isinstance(reply["token_sha"], str) and bool(re.fullmatch(r"[0-9a-f]{12}", reply["token_sha"]))
            and type(reply["tokens"]) is int and reply["tokens"] == len(reply["ids"])
            and type(reply["rounds"]) is int and reply["rounds"] >= 0
            and type(reply["drafts"]) is bool and reply["policy"] is not None)


def compare(name, d, s):
    """Judge one drafted/serial pair. The result is PASS or REJECT only when both replies carry evidence and modes are proven."""
    row = {"name": name, "drafted": d, "serial": s}
    if not (evidence_present(d) and evidence_present(s)):
        row.update(result="INVALID", error="no token evidence (ids, token_sha or stats missing/invalid)")
        return row
    drafted = d["drafts"] is True and d["rounds"] < d["tokens"] - 1
    serial = s["drafts"] is False and str(s["policy"]) == "0"
    row.update(drafting_proven=drafted, reference_serial=serial)
    if not (drafted and serial):
        row.update(result="INVALID", error="drafting or serial reference not proven")
        return row
    same = d["ids"] == s["ids"] and d["token_sha"] == s["token_sha"] and d["text"] == s["text"]
    row.update(identical=same, result="PASS" if same else "REJECT")
    if d["ids"] != s["ids"]:
        row["first_divergence"] = next(
            (i for i, (x, y) in enumerate(zip(d["ids"], s["ids"])) if x != y),
            min(len(d["ids"]), len(s["ids"])))
    return row


def overall_verdict(rows, identity_errors):
    # Invalid evidence/identity takes precedence over any earlier comparison.
    if identity_errors or any(r["result"] == "INVALID" for r in rows) or len(rows) != len(PROMPTS):
        return "INVALID"
    if any(r["result"] == "REJECT" for r in rows):
        return "REJECT"
    return "PASS"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8002")
    ap.add_argument("--out", default="exactness.json")
    a = ap.parse_args()
    a.url = a.url.rstrip("/")
    rows, samples, identity_errors = [], [], []

    def sample_start(stage):
        """Record /metrics process_start_time_seconds; any change or gap means the pair did not share one serving start."""
        sample = {"stage": stage, "process_start_time_seconds": None}
        try:
            with urllib.request.urlopen(a.url + "/metrics", timeout=10) as response:
                metrics = response.read().decode()
            values = re.findall(r"^process_start_time_seconds(?:\{[^}]*\})?\s+(\S+)", metrics, re.M)
            if len(values) != 1 or not math.isfinite(float(values[0])) or float(values[0]) <= 0:
                raise ValueError("missing or ambiguous process_start_time_seconds")
            value = float(values[0])
            sample["process_start_time_seconds"] = value
            if samples and value != samples[0]["process_start_time_seconds"]:
                identity_errors.append("serving start changed")
        except Exception as e:  # noqa: BLE001 - absent identity is INVALID
            sample["error"] = str(e)
            identity_errors.append("cannot verify one serving start")
        samples.append(sample)

    sample_start("before")
    if not identity_errors:
        for name, prompt, n in PROMPTS:
            try:
                d = chat(a.url, prompt, n)
                sample_start(name + ":between-passes")
                if identity_errors:
                    break
                s = chat(a.url, prompt, n, draft=False)
                sample_start(name + ":after-pair")
            except Exception as e:  # noqa: BLE001 - a failed request is INVALID
                rows.append({"name": name, "result": "INVALID", "error": "request failed: " + str(e)})
                print(f"{name:10s} INVALID: request failed: {e}", flush=True)
                continue
            if identity_errors:
                break
            row = compare(name, d, s)
            rows.append(row)
            detail = row.get("error", "token ids, token_sha and text compared")
            if "first_divergence" in row:
                detail += f", first differing token at {row['first_divergence']}"
            print(f"{name:10s} {row['result']:7s} {detail}", flush=True)
    sample_start("after")
    verdict = overall_verdict(rows, identity_errors)
    res = {"verdict": verdict, "url": a.url, "prompts": rows,
           "process_start_time_seconds": samples[0]["process_start_time_seconds"],
           "serving_start_samples": samples, "invalid_reasons": sorted(set(identity_errors)),
           "prompt_set_sha256": hashlib.sha256(json.dumps(PROMPTS).encode()).hexdigest()}
    with open(a.out, "w") as output:
        json.dump(res, output, indent=2)
        output.write("\n")
    for reason in res["invalid_reasons"]:
        print("INVALID: " + reason)
    print(f"EXACTNESS {verdict} ({sum(r.get('result') == 'PASS' for r in rows)}/{len(PROMPTS)} prompts); details in {a.out}")
    sys.exit({"PASS": 0, "REJECT": 20, "INVALID": 21}[verdict])


if __name__ == "__main__":
    main()
