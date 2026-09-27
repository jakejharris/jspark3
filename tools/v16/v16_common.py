#!/usr/bin/env python3
"""Shared, dependency-free helpers for the JSpark3 v1.6 QA tools.

The clients deliberately bypass environment proxy variables.  A hardware operator
chooses ``--base-url`` explicitly; the offline tests use only loopback servers.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterable

MODEL = "glm-5.3-flash"
METRIC_KEYS = (
    "vllm:request_success_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_preemptions_total",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)
_ERROR_PATTERN = re.compile(
    r"(?:NVRM:\s*)?Xid(?:\s*\([^)]*\))?\s*[:=,]|device-side assert|cuda error|"
    r"enginecore[^\r\n]*(?:died|failed|error)|engine[^\r\n]*error|"
    r"enginedeaderror|vectorized_gather_kernel|ind\s*>=\s*0[^\r\n]*ind_dim_size|"
    r"traceback \(most recent call last\)|illegal memory access|"
    # NCCL's own "NCCL INFO" lines are diagnostics, e.g. the per-init
    # "ncclOsDlopen(libnccl-env.so) fail" probe for an optional plugin; real
    # failures surface as NCCL WARN, ncclSystemError or "NCCL error".
    r"^(?![^\r\n]*\bnccl\s+info\b)[^\r\n]*nccl[^\r\n]*(?:error|fail|abort|unhandled)",
    re.I | re.M,
)
# vLLM's usage-stats thread raises on GB10 once per boot (cpuinfo returns
# non-JSON); it is telemetry only and never touches serving. Only a complete
# block from that thread, with usage_lib frames and a terminal exception line,
# is dropped; any other traceback still matches. Block lines are matched by
# their process prefix, so another worker's lines interleaved into the window
# are kept and still scanned; `docker logs --timestamps` stamps are tolerated.
_USAGE_THREAD_RE = re.compile(r"Exception in thread [^\r\n]*\(_report_usage_worker\):")
_LINE_PREFIX_RE = re.compile(r"(?:\d{4}-\d\d-\d\dT[\d:.]+Z\s+)?(\([^)]*\))?\s*")
_TERMINAL_EXCEPTION_RE = re.compile(r"[A-Za-z_][\w.]*(?:Error|Exception): ")


def _split_prefix(line: str) -> tuple[str, str]:
    match = _LINE_PREFIX_RE.match(line)
    return match.group(1) or "", line[match.end():]


def _strip_benign_blocks(text: str) -> str:
    lines = text.splitlines(keepends=True)
    drop: set[int] = set()
    for index, line in enumerate(lines):
        if index in drop or not _USAGE_THREAD_RE.search(line):
            continue
        owner = _split_prefix(line)[0]
        members = [other for other in range(index + 1, min(index + 60, len(lines)))
                   if _split_prefix(lines[other])[0] == owner]
        end = next((position for position, other in enumerate(members)
                    if _TERMINAL_EXCEPTION_RE.match(_split_prefix(lines[other])[1])), None)
        if end is not None and any("vllm/usage/usage_lib.py" in lines[other] for other in members[:end]):
            drop.update([index, *members[:end + 1]])
    return "".join(line for index, line in enumerate(lines) if index not in drop)


class _ErrorScanner:
    """``ERROR_RE.search(text)`` over engine logs minus the known-benign blocks above."""

    pattern = _ERROR_PATTERN

    def search(self, text: str):
        return _ERROR_PATTERN.search(_strip_benign_blocks(text))


ERROR_RE = _ErrorScanner()
IDENTITY_RE = re.compile(
    r"\[jspark3-v16:(coop|adaptive-k|dense-fp8)\]\s+rank=(\d+)\s+state=([^\s]+)"
)
EPOCH_KEYS = frozenset(("epoch", "mode", "fault_accept_delta"))
BOOT_CONFIG_KEYS = frozenset(("coop", "adaptive_boot", "dense_fp8"))
RUNTIME_CONFIG_KEYS = BOOT_CONFIG_KEYS | {"adaptive_epoch"}

DECODE_PROSE_PROMPT = (
    "Write a detailed step-by-step explanation of how a hash map works, "
    "including collision handling, resizing, and time complexity. Be thorough."
)
DECODE_STRUCTURED_PROMPT = (
    "Count from 1 to 200. Output only the numbers, separated by spaces. No other text."
)
CODE_TASK_TAIL = (
    "Output only Python source. No comments, no docstrings, no markdown fences. "
    "Then add tests and the helpers this needs. Keep writing code."
)
DECODE_CODE_TASKS = (
    "binary_search\ndef binary_search(nums, target) -> int: index of target in a sorted list, or -1.\n" + CODE_TASK_TAIL,
    "merge_sort\ndef merge_sort(nums) -> list: stable sort of a list of ints, returning a new list.\n" + CODE_TASK_TAIL,
    "lru_cache\nclass LRUCache: get(key) and put(key, value) with a fixed capacity, evicting the least recently used.\n" + CODE_TASK_TAIL,
    "token_bucket\nclass TokenBucket: allow(n) consumes n tokens refilled at a fixed rate, else returns False.\n" + CODE_TASK_TAIL,
    "ring_buffer\nclass RingBuffer: push and pop over a fixed-capacity array, raising on overflow and underflow.\n" + CODE_TASK_TAIL,
    "dijkstra\ndef dijkstra(graph, src) -> dict: shortest path weights from src on a non-negative weighted graph.\n" + CODE_TASK_TAIL,
    "edit_distance\ndef edit_distance(a, b) -> int: Levenshtein distance between two strings.\n" + CODE_TASK_TAIL,
    "semver_cmp\ndef semver_cmp(a, b) -> int: compare dotted numeric versions, negative if a < b.\n" + CODE_TASK_TAIL,
)

FIRST_PROMPTS = {
    "prose": "In four sentences, explain why leaves change color in autumn.",
    "structured": "Return only a JSON object with keys name, primes, and valid; use name='probe', primes=[2,3,5], valid=true.",
    "code": "Write a Python function clamp(x, low, high). Output only code.",
}

QUALITY_PROMPTS = {
    "prose": DECODE_PROSE_PROMPT,
    "structured": "Count from 1 to 10. Output only the numbers, separated by spaces.",
    "code": "Write only Python source for def add(a, b): returning a + b. No tests, comments or markdown.",
    "arithmetic": "Return only the integer answer: 17 + 25.",
    "arithmetic_sub": "Return only the integer answer: 144 - 57.",
    "arithmetic_mul": "Return only the integer answer: 12 * 13.",
    "fact": "Return only the city name: what is the capital of France?",
    "fact_author": "Return only the author name: who wrote Pride and Prejudice?",
    "fact_planet": "Return only the planet name: what is the largest planet in the Solar System?",
    "code_answer": "Return only a Python expression that evaluates to 42 using multiplication and addition.",
    "json_answer": 'Return only this JSON value with correct JSON types: {"answer":42,"ok":true}',
    "json_nested": 'Return only this JSON value: {"items":[1,2,3],"meta":{"count":3,"valid":true}}',
    "json_array": 'Return only this JSON array: [{"id":1},{"id":2},{"id":3}]',
}
FLEET_ENV_REQUIRED = (
    "JSPARK_RANK0_HOST",
    "JSPARK_RANK1_HOST",
    "JSPARK_RANK2_HOST",
    "JSPARK_WORK_ROOT",
)


class QAError(RuntimeError):
    """Expected refusal or evidence error."""


def load_fleet_env(path: Path) -> dict[str, str]:
    """Parse the non-expanding fleet env syntax used by fleetctl."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise QAError(f"cannot read env file: {exc}") from exc
    values = {}
    for number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, mark, value = line.partition("=")
        if not mark or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise QAError(f"invalid env syntax at line {number}")
        if key in values:
            raise QAError(f"duplicate env key: {key}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if "$" in value or "`" in value or "\n" in value:
            raise QAError(f"shell expansion is not supported: {key}")
        values[key] = value
    missing = [key for key in FLEET_ENV_REQUIRED if not values.get(key)]
    if missing:
        raise QAError("missing env keys: " + ",".join(missing))
    work = values["JSPARK_WORK_ROOT"]
    if not work.startswith("/") or work.endswith("/") or any(char.isspace() for char in work):
        raise QAError("JSPARK_WORK_ROOT must be an absolute path without trailing slash or whitespace")
    return values


def ssh_argv(values: dict[str, str], rank: int, remote_argv: list[str]) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "--",
            values[f"JSPARK_RANK{rank}_HOST"], shlex.join(remote_argv)]


def normalize_runtime_config(value: Any, *, include_epoch: bool = True) -> dict[str, str]:
    """Validate the caller label used to bind a capture to observed runtime state."""
    expected_keys = RUNTIME_CONFIG_KEYS if include_epoch else BOOT_CONFIG_KEYS
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise QAError(f"config must contain exactly {','.join(sorted(expected_keys))}")
    coop = value["coop"]
    if isinstance(coop, bool):
        coop = "on" if coop else "off"
    result = {
        "coop": coop,
        "adaptive_boot": value["adaptive_boot"],
        "dense_fp8": value["dense_fp8"],
    }
    if result["coop"] not in ("off", "on"):
        raise QAError("config coop must be off or on")
    if result["adaptive_boot"] not in ("off", "ema"):
        raise QAError("config adaptive_boot must be off or ema")
    if result["dense_fp8"] not in ("off", "trunk", "negative-coarse"):
        raise QAError("config dense_fp8 must be off, trunk, or negative-coarse")
    if include_epoch:
        result["adaptive_epoch"] = value["adaptive_epoch"]
        if result["adaptive_epoch"] not in ("off", "ema"):
            raise QAError("config adaptive_epoch must be off or ema")
        if result["adaptive_boot"] == "off" and result["adaptive_epoch"] != "off":
            raise QAError("an off boot cannot activate an ema epoch")
    return result


def expected_identity_config(config: dict[str, str]) -> dict[str, str]:
    return {
        "coop": config["coop"],
        "adaptive-k": config["adaptive_boot"],
        "dense-fp8": config["dense_fp8"],
    }


def observe_epoch(document: dict, sources: list[dict], env_file: Path | None) -> None:
    """Re-read live rank files; a prior write receipt alone is not current state."""
    if env_file is None:
        for row in sources:
            if "receipt" not in row and validate_epoch_document(load_json(Path(row["path"]))) != document:
                raise QAError("epoch changed during the measurement")
        return
    values = load_fleet_env(env_file)
    for rank in range(3):
        path = f"{values['JSPARK_WORK_ROOT']}/rank{rank}/evidence/adaptive-k-epoch.json"
        proc = subprocess.run(ssh_argv(values, rank, ["cat", "--", path]),
                              capture_output=True, text=True, check=False)
        try:
            actual = validate_epoch_document(json.loads(proc.stdout))
        except (ValueError, QAError) as exc:
            raise QAError(f"rank{rank}: cannot observe live epoch") from exc
        if proc.returncode or actual != document:
            raise QAError(f"rank{rank}: live epoch differs from capture receipt")


def capture_identity(log_text: str, config: dict, legacy_verify: Path | None = None):
    if legacy_verify is None:
        return identity_binding(log_text, expected_identity_config(config))
    # H0 runs on sealed v1.5, which has no v1.6 init lines or policy module.
    # Its existing verifier is the identity authority; never invent v1.6 logs.
    receipt = load_json(legacy_verify)
    unsigned = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    digest = hashlib.sha256((json.dumps(unsigned, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
    if (receipt.get("status") != "VERIFY_PASS" or receipt.get("payload_sha256") != digest
            or {row.get("rank") for row in receipt.get("runtime_identity", [])} != {0, 1, 2}
            or any(value != "off" for value in config.values())):
        raise QAError("H0 requires a hash-valid all-rank v1.5 verify receipt and all-off config")
    return [], {"legacy_v15_verify_sha256": digest}, expected_identity_config(config)


def identity_binding(
    log_text: str,
    expected: dict[str, str],
    *,
    ranks: tuple[int, ...] = (0, 1, 2),
) -> tuple[list[str], dict, dict[str, str]]:
    """Require exactly one matching init identity per option and rank."""
    findings = []
    observed: dict[str, dict[int, list[str]]] = {}
    for option, rank_text, state in IDENTITY_RE.findall(log_text):
        observed.setdefault(option, {}).setdefault(int(rank_text), []).append(state)
    for option in ("coop", "adaptive-k", "dense-fp8"):
        if option not in expected:
            findings.append(f"missing expected identity for {option}")
            continue
        for rank in ranks:
            states = observed.get(option, {}).get(rank, [])
            if states != [expected[option]]:
                findings.append(
                    f"identity mismatch {option} rank={rank}: observed={states!r}, "
                    f"expected exactly [{expected[option]!r}]"
                )
    serializable = {
        option: {str(rank): states for rank, states in sorted(by_rank.items())}
        for option, by_rank in sorted(observed.items())
    }
    bound = dict(expected) if not findings else {}
    return findings, serializable, bound


def identity_lines(log_text: str) -> list[str]:
    """Preserve complete init identities, including native/weight hashes."""
    return sorted(line[match.start():] for line in log_text.splitlines()
                  if (match := IDENTITY_RE.search(line)))


def fetch_remote_logs(
    env_file: Path,
    container_prefix: str,
    *,
    since: str | None = None,
) -> tuple[str, list[str]]:
    """Fetch all-rank container logs read-only, using the fleet's SSH binding."""
    values = load_fleet_env(env_file)
    combined = ""
    sources = []
    for rank in range(3):
        name = f"{container_prefix}{rank}"
        remote_argv = ["docker", "logs", "--timestamps"]
        if since is not None:
            remote_argv.extend(["--since", since])
        remote_argv.append(name)
        argv = ssh_argv(values, rank, remote_argv)
        try:
            proc = subprocess.run(
                argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
            )
        except OSError as exc:
            raise QAError(f"rank{rank} log collection failed: {exc}") from exc
        if proc.returncode:
            raise QAError(
                f"rank{rank} docker logs failed rc={proc.returncode}: {proc.stderr.strip()[-300:]}"
            )
        combined += f"\n===== remote rank{rank} {name} =====\n{proc.stdout}{proc.stderr}"
        sources.append(f"{values[f'JSPARK_RANK{rank}_HOST']}:{name}")
    return combined, sources


def collect_log_evidence(
    engine_logs: Iterable[Path] | None,
    env_file: Path | None,
    container_prefix: str,
    *,
    since: str | None = None,
) -> tuple[str, list[str]]:
    text = ""
    sources = []
    for path in engine_logs or []:
        try:
            text += f"\n===== {path} =====\n" + Path(path).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError as exc:
            raise QAError(f"cannot read engine log {path}: {exc}") from exc
        sources.append(str(path))
    if env_file is not None:
        remote_text, remote_sources = fetch_remote_logs(
            env_file, container_prefix, since=since
        )
        text += remote_text
        sources.extend(remote_sources)
    if not sources:
        raise QAError("all-rank engine logs are required")
    return text, sources


def validate_epoch_document(value: Any) -> dict[str, str | int]:
    if not isinstance(value, dict) or set(value) != EPOCH_KEYS:
        raise QAError("adaptive-k epoch must contain exactly epoch, mode, fault_accept_delta")
    epoch = value["epoch"]
    mode = value["mode"]
    fault = value["fault_accept_delta"]
    if not isinstance(epoch, (str, int)) or isinstance(epoch, bool):
        raise QAError("adaptive-k epoch id must be a string or integer")
    if mode not in ("off", "ema"):
        raise QAError("adaptive-k epoch mode must be off or ema")
    if type(fault) is not int or not -7 <= fault <= 7:
        raise QAError("adaptive-k fault_accept_delta must be an integer in [-7, 7]")
    return {"epoch": epoch, "mode": mode, "fault_accept_delta": fault}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def json_safe(value: Any) -> Any:
    """Make anomalous non-finite evidence representable in strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def load_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QAError(f"cannot read JSON {path}: {exc}") from exc


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _url(base_url: str, path: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise QAError("--base-url must be an absolute http(s) URL")
    return base_url.rstrip("/") + path


def http_get_text(base_url: str, path: str, timeout: float = 20.0) -> str:
    request = urllib.request.Request(_url(base_url, path), headers={"Accept": "text/plain"})
    try:
        with _opener().open(request, timeout=timeout) as response:
            return response.read().decode("utf-8", "replace")
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        raise QAError(f"GET {path} failed: {exc}") from exc


def http_post_json(base_url: str, path: str, body: dict, timeout: float = 360.0) -> dict:
    request = urllib.request.Request(
        _url(base_url, path),
        data=canonical_bytes(body),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with _opener().open(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:1000]
        raise QAError(f"POST {path} returned HTTP {exc.code}: {detail}") from exc
    except (OSError, urllib.error.URLError) as exc:
        raise QAError(f"POST {path} failed: {exc}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise QAError(f"POST {path} returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise QAError(f"POST {path} returned a non-object")
    return value


def parse_prometheus(text: str) -> dict[str, float]:
    """Aggregate series by metric name (labels are intentionally collapsed)."""
    result: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([^\s{]+)(?:\{[^}]*\})?\s+([^\s]+)(?:\s+\d+)?$", line)
        if not match:
            continue
        try:
            value = float(match.group(2))
        except ValueError:
            continue
        result[match.group(1)] = result.get(match.group(1), 0.0) + value
    return result


def metrics_snapshot(base_url: str, timeout: float = 20.0) -> dict[str, float]:
    return parse_prometheus(http_get_text(base_url, "/metrics", timeout))


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    keys = set(before) | set(after)
    return {key: after.get(key, 0.0) - before.get(key, 0.0) for key in sorted(keys)}


def idle_issues(metrics: dict[str, float]) -> list[str]:
    issues = []
    for key in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
        if key not in metrics:
            issues.append(f"missing metric {key}")
        elif metrics[key] != 0:
            issues.append(f"server not idle: {key}={metrics[key]:g}")
    return issues


def chat_body(
    prompt: str,
    *,
    model: str = MODEL,
    max_tokens: int,
    stream: bool,
    cache_salt: str | None = None,
    logprobs: bool = False,
    top_logprobs: int = 5,
    fill: bool = False,
) -> dict:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "stream": stream,
        "chat_template_kwargs": {
            "enable_thinking": False,
            "thinking": False,
            "thinking_mode": "disabled",
        },
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    if cache_salt is not None:
        body["cache_salt"] = cache_salt
    if logprobs:
        body.update(
            {
                "logprobs": True,
                "top_logprobs": top_logprobs,
                "return_token_ids": True,
                "return_tokens_as_token_ids": True,
            }
        )
    if fill:
        body.update({"min_tokens": max_tokens, "ignore_eos": True, "stop": []})
    return body


def _token_key(item: dict) -> str:
    token = item.get("token")
    if isinstance(token, str):
        return token
    raw = item.get("bytes")
    if isinstance(raw, list):
        return "bytes:" + ",".join(str(x) for x in raw)
    return repr(token)


def parse_chat_response(response: dict) -> dict:
    error = response.get("error")
    if error:
        raise QAError(f"server error object: {error}")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise QAError("chat response has no choice")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = (message.get("content") or "") if isinstance(message, dict) else ""
    reasoning = ""
    if isinstance(message, dict):
        reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    token_ids = choice.get("token_ids")
    if token_ids is None:
        token_ids = response.get("token_ids")
    if not isinstance(token_ids, list):
        token_ids = []
    rows = []
    lp = choice.get("logprobs")
    if isinstance(lp, dict) and isinstance(lp.get("content"), list):
        for item in lp["content"]:
            if not isinstance(item, dict):
                continue
            top = []
            for candidate in item.get("top_logprobs") or []:
                if isinstance(candidate, dict):
                    top.append({"token": _token_key(candidate), "logprob": candidate.get("logprob")})
            rows.append({"token": _token_key(item), "logprob": item.get("logprob"), "top_logprobs": top})
    return {
        "content": str(content),
        "reasoning": str(reasoning),
        "token_ids": token_ids,
        "logprobs": rows,
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage") if isinstance(response.get("usage"), dict) else {},
        "response_id": response.get("id"),
    }


def nonfinite_paths(value: Any, prefix: str = "$") -> list[str]:
    bad = []
    if isinstance(value, float) and not math.isfinite(value):
        bad.append(prefix)
    elif isinstance(value, dict):
        for key, child in value.items():
            bad.extend(nonfinite_paths(child, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            bad.extend(nonfinite_paths(child, f"{prefix}[{index}]"))
    return bad


def run_nonstream(base_url: str, body: dict, timeout: float = 360.0) -> dict:
    started = time.monotonic()
    raw = http_post_json(base_url, "/v1/chat/completions", body, timeout)
    parsed = parse_chat_response(raw)
    parsed.update(
        {
            "elapsed_s": time.monotonic() - started,
            "request_sha256": sha256_json(body),
            "nonfinite_paths": nonfinite_paths(raw),
        }
    )
    return parsed


def run_stream(base_url: str, body: dict, timeout: float = 360.0, barrier: threading.Barrier | None = None) -> dict:
    if barrier is not None:
        barrier.wait(timeout=min(timeout, 30.0))
    request = urllib.request.Request(
        _url(base_url, "/v1/chat/completions"),
        data=canonical_bytes(body),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.monotonic()
    first = None
    last = None
    event_times: list[float] = []
    usage: dict = {}
    finish_reason = None
    content: list[str] = []
    token_ids: list[int] = []
    saw_done = False
    error = None
    try:
        with _opener().open(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    saw_done = True
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    error = "invalid SSE JSON"
                    break
                if event.get("error"):
                    error = f"SSE error: {event['error']}"
                    break
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices or not isinstance(choices[0], dict):
                    continue
                choice = choices[0]
                if choice.get("finish_reason") is not None:
                    finish_reason = choice.get("finish_reason")
                delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
                text = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                ids = choice.get("token_ids") or []
                if text or ids:
                    now = time.monotonic()
                    if first is None:
                        first = now
                    last = now
                    event_times.append(now)
                    content.append(text)
                    if isinstance(ids, list):
                        token_ids.extend(x for x in ids if isinstance(x, int))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:1000]
        error = f"HTTP {exc.code}: {detail}"
    except Exception as exc:  # network/protocol evidence belongs in the result
        error = f"{type(exc).__name__}: {exc}"
    ended = time.monotonic()
    completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if not isinstance(completion_tokens, int):
        completion_tokens = len(token_ids) or None
    decode_s = (last - first) if first is not None and last is not None else None
    decode_tps = None
    if completion_tokens is not None and completion_tokens > 1 and decode_s is not None and decode_s > 0:
        decode_tps = (completion_tokens - 1) / decode_s
    return {
        "error": error,
        "complete": saw_done and error is None,
        "finish_reason": finish_reason,
        "content": "".join(content),
        "token_ids": token_ids,
        "usage": usage,
        "completion_tokens": completion_tokens,
        "prompt_tokens": usage.get("prompt_tokens") if isinstance(usage, dict) else None,
        "start_mono": started,
        "first_mono": first,
        "last_mono": last,
        "end_mono": ended,
        "ttft_s": first - started if first is not None else None,
        "decode_s": decode_s,
        "decode_tps": decode_tps,
        "step_gaps_s": [b - a for a, b in zip(event_times, event_times[1:])],
        "request_sha256": sha256_json(body),
    }


def concurrent_calls(function, arguments: list[tuple], timeout: float) -> list[Any]:
    """Start all calls behind a barrier and retain input order."""
    if not arguments:
        return []
    barrier = threading.Barrier(len(arguments))
    results: list[Any] = [None] * len(arguments)
    errors: list[BaseException | None] = [None] * len(arguments)

    def target(index: int, args: tuple) -> None:
        try:
            results[index] = function(*args, barrier=barrier)
        except BaseException as exc:  # surface after every peer has joined
            errors[index] = exc

    threads = [threading.Thread(target=target, args=(i, args), daemon=True) for i, args in enumerate(arguments)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + timeout
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    if any(thread.is_alive() for thread in threads):
        raise QAError("concurrent request wave timed out")
    if any(errors):
        raise QAError("concurrent request failed: " + "; ".join(str(x) for x in errors if x))
    return results


def median(values: Iterable[float]) -> float | None:
    xs = [float(x) for x in values if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return statistics.median(xs) if xs else None


def min_median_max(values: Iterable[float]) -> dict[str, float | None]:
    xs = sorted(float(x) for x in values if isinstance(x, (int, float)) and math.isfinite(float(x)))
    return {
        "min": xs[0] if xs else None,
        "median": statistics.median(xs) if xs else None,
        "max": xs[-1] if xs else None,
        "n": len(xs),
    }


def decode_prompts(kind: str, concurrency: int) -> list[str]:
    if kind == "code":
        return [DECODE_CODE_TASKS[i % len(DECODE_CODE_TASKS)] for i in range(concurrency)]
    base = DECODE_PROSE_PROMPT if kind == "prose" else DECODE_STRUCTURED_PROMPT
    if concurrency == 1:
        return [base]
    return [f"{base} (stream {i + 1}/{concurrency})" for i in range(concurrency)]


def redact_monotonic(rows: list[dict]) -> None:
    """Replace process-local absolute monotonic stamps with wave-relative stamps."""
    starts = [r.get("start_mono") for r in rows if isinstance(r.get("start_mono"), (int, float))]
    origin = min(starts) if starts else 0.0
    for row in rows:
        for key in ("start_mono", "first_mono", "last_mono", "end_mono"):
            if isinstance(row.get(key), (int, float)):
                row[key.replace("_mono", "_relative_s")] = row[key] - origin
            row.pop(key, None)
