#!/usr/bin/env python3
"""Compare captured OpenAI responses from fresh and disk-resumed GLM requests.

Enable TF_GLM_SESSION_HASH_GATE on every rank and return_token_ids on both
requests. This checker is offline: it never loads the serving endpoint.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def check_pair(fresh: dict, restored: dict, *, world: int = 3, expected_cached: int | None = None) -> dict:
    def receipt(body):
        if "error" in body:
            raise ValueError("request failed")
        stats = body.get("tensorfold", body.get("stats", {}))
        hashes, tokens = stats.get("session_state_sha256"), stats.get("token_ids")
        if (not isinstance(hashes, list) or len(hashes) != world or
                any(not isinstance(h, str) or re.fullmatch("[0-9a-f]{64}", h) is None for h in hashes)):
            raise ValueError("missing or invalid per-rank state hashes")
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int for t in tokens):
            raise ValueError("return_token_ids is required; text or truncated hashes are not the output gate")
        return stats, hashes, tokens
    cold, cold_hashes, cold_tokens = receipt(fresh)
    warm, warm_hashes, warm_tokens = receipt(restored)
    if cold.get("cached") != 0 or cold.get("session_cache_source") != "cold":
        raise ValueError("fresh reference was not a cold prefill")
    if warm.get("session_cache_source") != "disk" or not isinstance(warm.get("cached"), int) or warm["cached"] <= 0:
        raise ValueError("candidate did not restore from disk")
    if expected_cached is not None and warm["cached"] != expected_cached:
        raise ValueError("candidate restored the wrong checkpoint")
    if cold.get("policy") != warm.get("policy") or cold.get("drafts") != warm.get("drafts"):
        raise ValueError("drafter policy differs between the paired requests")
    if cold_hashes != warm_hashes:
        raise ValueError("restored != fresh canonical state")
    if cold_tokens != warm_tokens:
        raise ValueError("restored != fresh generated continuation")
    return {"pass": True, "world": world, "cached": warm["cached"], "tokens": len(warm_tokens),
            "state_sha256": warm_hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", type=Path, required=True)
    parser.add_argument("--restored", type=Path, required=True)
    parser.add_argument("--world", type=int, default=3)
    parser.add_argument("--expected-cached", type=int)
    args = parser.parse_args()
    try:
        result = check_pair(json.loads(args.fresh.read_text()), json.loads(args.restored.read_text()),
                            world=args.world, expected_cached=args.expected_cached)
    except (ValueError, KeyError, TypeError) as exc:
        parser.exit(1, json.dumps({"pass": False, "reason": str(exc)}) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
