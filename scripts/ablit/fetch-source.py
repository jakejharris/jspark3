#!/usr/bin/env python3
"""Fetch the pinned refusal-removed source after the user accepts its HF gate.

Usage: HF_TOKEN must be set in the environment; python3 fetch-source.py OUTPUT
The account must have accepted the source repository's terms on Hugging Face.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

REPO = "orcarouter/GLM-5.3-Flash-Uncensored-MLX"
REV = "c02a5f6fa06f0aa444877b44d19fd5c96390329f"


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, endpoint="https://huggingface.co"):
        self.auth_origin = self.origin(endpoint)

    @staticmethod
    def origin(url):
        parts = urllib.parse.urlsplit(url)
        port = parts.port if parts.port is not None else {"https": 443, "http": 80}.get(parts.scheme)
        return parts.scheme, parts.hostname, port

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(req.full_url).scheme == "https" and urllib.parse.urlsplit(newurl).scheme == "http":
            raise urllib.error.HTTPError(newurl, code, "HTTPS to HTTP redirect refused", headers, fp)
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected and self.origin(newurl) != self.auth_origin:
            redirected.remove_header("Authorization")
        return redirected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--variant", choices=("ablit", "base"), default="ablit")
    a = p.parse_args()
    repo = REPO if a.variant == "ablit" else "TensorFold/GLM-5.3-Flash-MLX-4bit-MTP"
    rev = REV if a.variant == "ablit" else "76add2a341a1cd90ad0e86bb69839ea9c35827c6"
    token = os.environ.get("HF_TOKEN", "").strip() if a.variant == "ablit" else ""
    if a.variant == "ablit" and not token:
        raise SystemExit("Set HF_TOKEN after accepting the terms at https://huggingface.co/" + REPO)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), SafeRedirect())
    base = f"https://huggingface.co/{repo}/resolve/{rev}/"
    auth_headers = {"Authorization": "Bearer " + token} if token else {}
    started = time.monotonic()
    try:
        first_shard = "model-00001-of-00062.safetensors" if a.variant == "ablit" else "model-00001-of-00043.safetensors"
        req = urllib.request.Request(base + first_shard, headers={**auth_headers, "Range": "bytes=0-0"})
        with opener.open(req, timeout=60) as response:
            if response.status not in (200, 206) or len(response.read(1)) != 1:
                raise SystemExit("The pinned source could not be read; check HF_TOKEN and gate acceptance.")
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"Source access denied (HTTP {exc.code}). Accept the repository terms yourself and supply an authorized HF_TOKEN.") from None
    except urllib.error.URLError:
        raise SystemExit("Could not connect to the pinned source.") from None
    # The pinned hash inventory itself is public and is fetched anonymously.
    with opener.open(f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true", timeout=60) as response:
        info_bytes = response.read()
    info = json.loads(info_bytes)
    if info["sha"] != rev:
        raise SystemExit("Unexpected source revision")
    files = sorted((x for x in info["siblings"] if "/" not in x["rfilename"]), key=lambda x: x["rfilename"])
    shards = [x for x in files if x["rfilename"].endswith(".safetensors")]
    if len(shards) != (62 if a.variant == "ablit" else 43) or not all(x.get("lfs", {}).get("sha256") for x in shards):
        raise SystemExit("Pinned source shard/hash inventory is incomplete")
    a.output.mkdir(parents=True, exist_ok=False)
    receipt = a.output.parent / (a.output.name + "-fetch")
    receipt.mkdir(exist_ok=False)
    (receipt / "public-api.json").write_bytes(info_bytes)
    count_lock = threading.Lock()
    network_bytes = 1
    results = []

    def fetch(meta):
        nonlocal network_bytes
        name = meta["rfilename"]
        if name in ("", ".", "..") or Path(name).name != name:
            raise RuntimeError("Invalid source path")
        size = meta["size"]
        dest = a.output / name
        for attempt in range(1, 5):
            start = time.monotonic()
            try:
                request = urllib.request.Request(base + urllib.parse.quote(name) + "?download=true", headers=auth_headers)
                h256 = hashlib.sha256()
                hgit = hashlib.sha1(f"blob {size}\0".encode())
                received = 0
                with opener.open(request, timeout=90) as response, dest.with_name(name + ".partial").open("wb") as output:
                    if response.status != 200:
                        raise ValueError("Unexpected partial response")
                    while chunk := response.read(8 * 1024 * 1024):
                        output.write(chunk)
                        h256.update(chunk)
                        if "lfs" not in meta:
                            hgit.update(chunk)
                        received += len(chunk)
                        with count_lock:
                            network_bytes += len(chunk)
                expected = meta.get("lfs", {}).get("sha256", meta["blobId"])
                got = h256.hexdigest() if "lfs" in meta else hgit.hexdigest()
                if received != size or got != expected:
                    raise ValueError("Pinned size or hash mismatch")
                os.replace(dest.with_name(name + ".partial"), dest)
                result = {"file": name, "bytes": received, "sha256": h256.hexdigest(), "reference_kind": "lfs-sha256" if "lfs" in meta else "git-blob-sha1", "reference": expected, "seconds": round(time.monotonic() - start, 3), "attempt": attempt}
                print(json.dumps(result), flush=True)
                return result
            except Exception as exc:
                # Never print exception messages, response headers, tokens or signed URLs.
                print(json.dumps({"file": name, "attempt": attempt, "error_type": type(exc).__name__, "http_status": getattr(exc, "code", None)}), flush=True)
                if attempt == 4:
                    raise RuntimeError("Download failed for " + name) from None
                time.sleep(2 ** attempt)

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(fetch, files))
    manifest = "".join(f"{r['sha256']}  {r['file']}\n" for r in results)
    (receipt / "SHA256SUMS").write_text(manifest)
    summary = {"repo": repo, "revision": rev, "authentication": "HF_TOKEN environment; gate accepted by account owner" if token else "anonymous", "wall_seconds": time.monotonic() - started, "network_body_bytes": network_bytes + len(info_bytes), "source_bytes": sum(x["bytes"] for x in results), "files": results, "manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest()}
    (receipt / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "files"}), flush=True)


if __name__ == "__main__":
    main()
