#!/usr/bin/env python3
"""Offline reply prefill strict-prefix gate on captured request/result/next-request JSONL.

Each row contains body, result (GlmApp.run's normalized result), next_body, and
optionally expect_hit. Supply the exact serving tokenizer/template. No endpoint
or GPU is used; the receipt contains counts and hashes, never conversation text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from tensorfold.families.glm5_next.cuda.reply_prefill import candidate


def make_app(model_dir, *, parity=False):
    from tokenizers import Tokenizer
    from tensorfold.cuda.server import ChatTemplate
    from tensorfold.families.glm5_next.cuda.app import GlmApp, ThinkingOffTemplate

    app = GlmApp.__new__(GlmApp)
    app.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    app.template = ChatTemplate(model_dir)
    if not parity:
        app.template = ThinkingOffTemplate(app.template)
    app.engine = SimpleNamespace(limit=262144)
    app.default_thinking, app.max_tokens = True, 1
    return app


def check_pair(app, pair):
    # _prepare does not perform engine admission and keeps the same tool/effort rules.
    prepared = app._prepare(pair["body"], True)
    try:
        ids, reason = candidate(app, pair["body"], prepared, pair["result"])
        actual = app._prepare(pair["next_body"], True)
        try:
            hit = ids is not None and len(ids) < len(actual.prompt) and ids == actual.prompt[:len(ids)]
            return {"candidate_status": reason, "hit": hit, "base_tokens": len(prepared.prompt),
                    "candidate_tokens": len(ids or []), "next_tokens": len(actual.prompt),
                    "useful_reply_tokens": len(ids) - len(prepared.prompt) if hit else 0}
        finally:
            actual.close()
    finally:
        prepared.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--parity", action="store_true", help="TF_V2_PARITY serving template (no thinking-off wrapper)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    app = make_app(args.model_dir, parity=args.parity)
    rows, failures = [], 0
    for number, line in enumerate(args.pairs.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            pair = json.loads(line)
            row = check_pair(app, pair)
            if "expect_hit" in pair and row["hit"] != pair["expect_hit"]:
                failures += 1
                row["expectation_failed"] = True
        except Exception:
            # Template failures can embed private message text; do not print them.
            row = {"error": "pair-render-failed"}
            failures += 1
        rows.append({"row": number, **row})
    files = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    receipt = {"schema": "tf-reply-prefill/v1", "pairs_sha256": hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
               "model_files": {name: hashlib.sha256((args.model_dir / name).read_bytes()).hexdigest()
                               for name in files if (args.model_dir / name).exists()},
               "parity": args.parity, "pairs": len(rows), "hits": sum(r.get("hit", False) for r in rows),
               "useful_reply_tokens": sum(r.get("useful_reply_tokens", 0) for r in rows),
               "failures": failures, "rows": rows}
    text = json.dumps(receipt, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(json.dumps({k: v for k, v in receipt.items() if k != "rows"}, sort_keys=True))
    return int(bool(failures or not rows))


if __name__ == "__main__":
    raise SystemExit(main())
