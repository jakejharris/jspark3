#!/usr/bin/env python3
"""Desk-only protocol fake for the quality ruler. No model/GPU involved."""
import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=18742)
parser.add_argument("--nll-shift", type=float, default=0)
parser.add_argument("--bad-tool", action="store_true")
parser.add_argument("--tool-prose", action="store_true")
parser.add_argument("--tool-stop", action="store_true")
parser.add_argument("--bad-code", action="store_true")
parser.add_argument("--bad-recall", action="store_true")
args = parser.parse_args()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        if self.path == "/v1/quality/score":
            if "token_ids" in body:
                ids = body["token_ids"]
                start = body["score_start"]
            else:
                text = body["text"]
                ids = list(text.encode())
                start = len(body["score_from_text"].encode())
            reply = {"token_ids": ids, "score_start": start,
                     "nll": [2.0 + args.nll_shift] * (len(ids) - start)}
        elif self.path == "/v1/chat/completions":
            prompt = body["messages"][0]["content"]
            calls = []
            content = ""
            if body.get("tools"):
                if "Add 17" in prompt:
                    name, arguments = "add", {"a": 17, "b": -4}
                elif "copper" in prompt:
                    name, arguments = "lookup", {"key": "copper"}
                elif "Friday" in prompt:
                    name, arguments = "schedule", {"day": "Friday", "hour": 14}
                else:
                    name, arguments = "search", {"query": "orbital period"}
                if args.bad_tool:
                    arguments = {}
                calls = [{"type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]
                if args.tool_prose:
                    content = "I will use the tool now."
            elif "sum of even" in prompt:
                content = "def solve(nums):\n    return sum(x for x in nums if x % 2 == 0)"
            elif "number of vowels" in prompt:
                content = "def solve(text):\n    return sum(1 for c in text if c in 'aeiouAEIOU')"
            elif "adjacent duplicate" in prompt:
                content = "def solve(nums):\n    return [x for i,x in enumerate(nums) if i == 0 or x != nums[i-1]]"
            elif "Fibonacci" in prompt:
                content = "def solve(n):\n    a,b=0,1\n    for i in range(n):\n        a,b=b,a+b\n    return a"
            else:
                content = ", ".join(re.findall(r"checkpoint key = ([A-Z]+-[0-9]+)", prompt))
            if args.bad_code and "Define function solve" in prompt:
                content = "def solve(*args): return None"
            if args.bad_recall and "What are the checkpoint keys" in prompt:
                content = "WRONG-0"
            reply = {"choices": [{"finish_reason": "stop" if args.tool_stop else "tool_calls" if calls else "stop",
                                  "message": {"content": content, "tool_calls": calls}}],
                     "usage": {"prompt_tokens": 0}}
        else:
            self.send_error(404)
            return
        payload = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
