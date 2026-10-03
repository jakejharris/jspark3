#!/usr/bin/env python3
"""Convert the pinned OrcaRouter MLX checkpoint to uniform four-bit weights.

The converter preserves compatible tensors, requantizes mixed-precision
matrices, and copies the native prediction layer from the pinned base model.
Uses NumPy on the CPU. Read the input model licenses before use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time

import numpy as np

GROUP = 64
F32 = np.float32
DTYPE_BYTES = {"U32": 4, "I32": 4, "F32": 4, "BF16": 2, "F16": 2, "I16": 2, "U16": 2, "U8": 1, "I8": 1, "I64": 8,
               "F64": 8, "BOOL": 1}
OUT_TOTAL = 63                        # 62 orcarouter shards + the layer-45 graft
GRAFT_RE = re.compile(r"^model\.language_model\.layers\.45\.")
CHUNK_ELEMS = 1 << 22                 # rows per chunk keep a module's fp32 working set near 16 MB per array
SMALL_FROM_VONTRA = ("config.json", "generation_config.json", "processor_config.json", "tokenizer.json",
                     "tokenizer_config.json")
STACK_TEMPLATE_SHA = "77c01ab2ea2013fb68161c8c0de6bf6c96c95b3e699e518554da384a695ab482"   # Modified template built from the pinned public source
CODEC_VERSION = "abl-codec-2 numpy mlx-affine rtn signed-edge g64 bf16-rne-after-codes"
ORCA_REV = "orcarouter/GLM-5.3-Flash-Uncensored-MLX@c02a5f6fa06f0aa444877b44d19fd5c96390329f"
VONTRA_REV = "Vontra/GLM-5.3-Flash-MLX-4bit-MTP@76add2a341a1cd90ad0e86bb69839ea9c35827c6"
with open(__file__, "rb") as _f:
    CODE_SHA = hashlib.sha256(_f.read()).hexdigest()


def log(*a):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), *a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- codec (codec validation: bit-exact against mlx.core)

def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Round to nearest even (finite values), as mlx's astype(bfloat16)."""
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((b + ((b >> 16) & 1) + np.uint32(0x7FFF)) >> 16).astype(np.uint16)


def f16_to_f32(u16: np.ndarray) -> np.ndarray:
    return u16.view(np.float16).astype(np.float32)


def unpack(wq: np.ndarray, bits: int) -> np.ndarray:
    """MLX-packed U32 [R, W] -> uint8 codes [R, W*32/bits]. Each row is a little-endian bitstream, code j at bits
    [j*bits, (j+1)*bits): 8 codes per `bits` bytes (so 2/4/8-bit is also 32/bits codes per u32, low bits first)."""
    r = wq.shape[0]
    by = np.ascontiguousarray(wq).view(np.uint8).reshape(r, -1)
    if bits == 8:
        return by
    if bits == 4:
        out = np.empty((r, by.shape[1], 2), np.uint8)
        np.bitwise_and(by, 15, out=out[..., 0])
        np.right_shift(by, 4, out=out[..., 1])
        return out.reshape(r, -1)
    grp = by.reshape(r, -1, bits)
    v = grp[..., 0].astype(np.uint64)
    for j in range(1, bits):
        v |= grp[..., j].astype(np.uint64) << np.uint64(8 * j)
    shifts = (np.arange(8, dtype=np.uint64) * np.uint64(bits))
    codes = (v[..., None] >> shifts) & np.uint64((1 << bits) - 1)
    return codes.astype(np.uint8).reshape(r, -1)


def pack4(codes: np.ndarray) -> np.ndarray:
    """uint8 codes [R, K] (< 16) -> MLX 4-bit U32 [R, K/8], code k of a word at bits 4k."""
    r, k = codes.shape
    c = codes.reshape(r, k // 2, 2)
    by = c[..., 0] | (c[..., 1] << 4)
    return np.ascontiguousarray(by, dtype=np.uint8).view(np.uint32)


def dequant(codes: np.ndarray, s: np.ndarray, b: np.ndarray) -> np.ndarray:
    """uint8 codes [R, K], fp32 s/b [R, K/64] -> fp32 [R, K]: q*s is exact in fp32, so one rounding, like fma."""
    r, k = codes.shape
    w = codes.reshape(r, k // GROUP, GROUP).astype(np.float32)
    w *= s[..., None]
    w += b[..., None]
    return w.reshape(r, k)


def _scale_bias(w_min: np.ndarray, w_max: np.ndarray, bits: int):
    """MLX affine (mlx/backend/cpu/quantized.cpp): the larger-magnitude edge sits exactly on the grid, so the step
    is negative when the max is the edge, and 0 is exactly representable when the edge rounds off zero."""
    n_bins = F32((1 << bits) - 1)
    mask = np.abs(w_min) > np.abs(w_max)
    scale = np.maximum((w_max - w_min) / n_bins, F32(1e-7))
    scale = np.where(mask, scale, -scale)
    edge = np.where(mask, w_min, w_max)
    q0 = np.rint(edge / scale)
    nz = q0 != 0
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(nz, edge / q0, scale).astype(np.float32)
    bias = np.where(nz, edge, F32(0)).astype(np.float32)
    return scale, bias


def quantize(w: np.ndarray, bits: int = 4):
    """fp32 [R, K] -> (uint8 codes [R, K], fp32 scale [R, K/64], fp32 bias [R, K/64]); codes use the fp32 s/b."""
    r, k = w.shape
    g = w.reshape(r, k // GROUP, GROUP)
    scale, bias = _scale_bias(g.min(-1), g.max(-1), bits)
    q = g - bias[..., None]
    q /= scale[..., None]
    np.rint(q, out=q)
    np.clip(q, 0, F32((1 << bits) - 1), out=q)
    return q.astype(np.uint8).reshape(r, k), scale, bias


def requant_lut(codes: np.ndarray, s: np.ndarray, b: np.ndarray, bits_in: int, bits: int = 4):
    """quantize(dequant(codes, s, b)) through a per-group table of the 2**bits_in levels: identical output (every
    step is a function of the code within its group, and fp32 rounding is monotonic, so the group's min and max
    sit at its min and max codes), in about a third of the passes."""
    r, k = codes.shape
    cg = codes.reshape(r, k // GROUP, GROUP)
    lev = np.arange(1 << bits_in, dtype=np.float32)
    table = lev * s[..., None]
    table += b[..., None]                                   # [R, G, L] = the dequantized value of each level
    lo = np.take_along_axis(table, cg.min(-1)[..., None], -1)[..., 0]
    hi = np.take_along_axis(table, cg.max(-1)[..., None], -1)[..., 0]
    scale, bias = _scale_bias(np.minimum(lo, hi), np.maximum(lo, hi), bits)
    table -= bias[..., None]
    table /= scale[..., None]
    np.rint(table, out=table)
    np.clip(table, 0, F32((1 << bits) - 1), out=table)
    new = np.take_along_axis(table.astype(np.uint8), cg, -1)
    return new.reshape(r, k), scale, bias


# ---------------------------------------------------------------- safetensors headers

def read_header(path: str):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def headers_from_dir(d: str, pattern=r"^model-\d{5}-of-\d{5}\.safetensors$"):
    files, bases = {}, {}
    for fn in sorted(os.listdir(d)):
        if re.match(pattern, fn):
            files[fn], bases[fn] = read_header(os.path.join(d, fn))
    return files, bases


def headers_from_json(path: str):
    """The saved header inventories: {"files": {shard: {tensor: info}}}."""
    files = json.load(open(path))["files"]
    return {f: {k: v for k, v in t.items() if k != "__metadata__"} for f, t in files.items()}, None


def nbytes(dtype: str, shape) -> int:
    n = DTYPE_BYTES[dtype]
    for d in shape:
        n *= d
    return n


# ---------------------------------------------------------------- plan (pure function of the headers)

def load_target(path: str):
    """{name: [dtype, shape, vontra_file]} from Vontra's full header set (built from the verified base checkpoint)."""
    t = json.load(open(path))
    return t["tensors"] if "tensors" in t else t


def build_plan(orca: dict, target: dict):
    """Every output tensor exactly once, grouped into output shards. Raises with the offending names on any tensor
    that has no rule, any Vontra name with no source, or any orca name outside Vontra's set (conversion rules)."""
    src = {}                                         # orca name -> (file, info)
    for fn, t in orca.items():
        for k, v in t.items():
            src[k] = (fn, v)
    stray = sorted(k for k in src if k not in target)
    if stray:
        raise SystemExit(f"STOP: {len(stray)} orcarouter names outside Vontra's set, first: {stray[:20]}")

    shards, counts, used = {}, {}, set()
    bump = lambda key: counts.__setitem__(key, counts.get(key, 0) + 1)
    # output order: each orca shard's tensors in data order; a module's outputs sit together at its .weight
    for fn in sorted(orca):
        m = re.match(r"^model-(\d{5})-of-\d{5}\.safetensors$", fn)
        out_fn = f"model-{m.group(1)}-of-{OUT_TOTAL:05d}.safetensors"
        ops = shards.setdefault(out_fn, [])
        for name, info in sorted(orca[fn].items(), key=lambda kv: (kv[1].get("data_offsets", [0])[0], kv[0])):
            if name in used:
                continue
            base, _, leaf = name.rpartition(".")
            t_dt, t_shape = target[name][0], list(target[name][1])
            quant_target = leaf in ("weight", "scales", "biases") and base + ".scales" in target \
                and base + ".biases" in target and target.get(base + ".weight", [None])[0] == "U32"
            if not quant_target:
                if info["dtype"] == t_dt and list(info["shape"]) == t_shape:
                    ops.append({"op": "copy", "outs": [[name, t_dt, t_shape]], "src": {name: src[name][0]}})
                    used.add(name)
                    bump("copy")
                    continue
                raise SystemExit(f"STOP: {name}: orcarouter {info['dtype']}{info['shape']} vs Vontra {t_dt}{t_shape}"
                                 f" and no rule for it")
            outs = [[base + "." + p, target[base + "." + p][0], list(target[base + "." + p][1])]
                    for p in ("weight", "scales", "biases")]
            w_t, s_t, b_t = outs
            if w_t[1] != "U32" or s_t[1] != "BF16" or b_t[1] != "BF16" or s_t[2] != b_t[2]:
                raise SystemExit(f"STOP: {base}: Vontra triple is not U32/BF16/BF16: {outs}")
            if len(s_t[2]) != 2:
                raise SystemExit(f"STOP: {base}: Vontra scales are not 2-d: {outs}")
            rows, k = s_t[2][0], s_t[2][1] * GROUP
            if w_t[2] != [rows, k // 8]:
                raise SystemExit(f"STOP: {base}: Vontra shapes are not a 4-bit g64 [out, in] triple: {outs}")
            w_src = src.get(base + ".weight")
            if w_src is None:
                raise SystemExit(f"STOP: {base}.weight missing in orcarouter")
            wi = w_src[1]
            if wi["dtype"] == "BF16":
                if base + ".scales" in src or base + ".biases" in src or list(wi["shape"]) != [rows, k]:
                    raise SystemExit(f"STOP: {base}: BF16 weight {wi['shape']} does not match [{rows}, {k}]")
                op = {"op": "quant_bf16", "bits_in": 16, "outs": outs, "src": {base + ".weight": w_src[0]}}
                used.add(base + ".weight")
                bump("quant_bf16")
            elif wi["dtype"] == "U32":
                si, bi = src.get(base + ".scales"), src.get(base + ".biases")
                if si is None or bi is None:
                    raise SystemExit(f"STOP: {base}: U32 weight without scales/biases")
                if si[1]["dtype"] != "F16" or bi[1]["dtype"] != "F16" or list(si[1]["shape"]) != [rows, k // GROUP] \
                        or list(bi[1]["shape"]) != [rows, k // GROUP] or wi["shape"][0] != rows:
                    raise SystemExit(f"STOP: {base}: orcarouter s/b {si[1]} {bi[1]} vs rows {rows} in {k}")
                bits, rem = divmod(wi["shape"][1] * 32, k)
                if rem or bits not in (2, 3, 4, 5, 6, 8):
                    raise SystemExit(f"STOP: {base}: U32 {wi['shape']} is not a whole bit width over in={k}")
                kind = "keep4" if bits == 4 else f"requant{bits}"
                op = {"op": kind, "bits_in": bits, "outs": outs,
                      "src": {base + ".weight": w_src[0], base + ".scales": si[0], base + ".biases": bi[0]}}
                used.update((base + ".weight", base + ".scales", base + ".biases"))
                bump(kind)
            else:
                raise SystemExit(f"STOP: {base}.weight dtype {wi['dtype']} has no rule")
            ops.append(op)

    # layer 45 graft: every Vontra name orcarouter lacks must be a layer-45 name (and nothing else)
    covered = {o[0] for ops in shards.values() for op in ops for o in op["outs"]}
    missing = sorted(k for k in target if k not in covered)
    bad = [k for k in missing if not GRAFT_RE.match(k)]
    if bad:
        raise SystemExit(f"STOP: {len(bad)} Vontra names with no source, first: {bad[:20]}")
    gops = shards.setdefault(f"model-{OUT_TOTAL:05d}-of-{OUT_TOTAL:05d}.safetensors", [])
    for k in missing:
        dt, shape, vf = target[k][0], list(target[k][1]), target[k][2]
        gops.append({"op": "graft", "outs": [[k, dt, shape]], "src": {k: vf}})
        bump("graft")
    return shards, counts


def plan_diff(shards: dict, target: dict):
    """Validate the plan: the (name, dtype, shape) set equals Vontra's; each name produced once."""
    got, dup = {}, []
    for fn, ops in shards.items():
        for op in ops:
            for name, dt, shape in op["outs"]:
                if name in got:
                    dup.append(name)
                got[name] = (dt, list(shape))
    want = {k: (v[0], list(v[1])) for k, v in target.items()}
    diffs = [f"only-out {k}" for k in sorted(set(got) - set(want))]
    diffs += [f"only-vontra {k}" for k in sorted(set(want) - set(got))]
    diffs += [f"differs {k} {got[k]} vs {want[k]}" for k in sorted(set(got) & set(want)) if got[k] != want[k]]
    diffs += [f"duplicate {k}" for k in dup]
    tb = sum(nbytes(*v) for v in got.values())
    vb = sum(nbytes(*v) for v in want.values())
    return diffs, len(got), tb, vb


# ---------------------------------------------------------------- run (one output shard per task)

class MemStop(Exception):
    """The box's MemAvailable fell under the floor: the unit aborts and releases its memory."""


def mem_available_gib(path: str = "/proc/meminfo") -> float:
    with open(path) as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576
    return 0.0


class Guard:
    """Second layer under the hard cgroup cap: before every row chunk, read the host's MemAvailable (docker shows the
    host's /proc/meminfo) and raise MemStop under the floor, so the unit drops its buffers instead of sleeping on
    them; the run then exits 3 and a rerun resumes from the shards already validated."""

    def __init__(self, floor_gib: float, meminfo: str = "/proc/meminfo"):
        self.floor, self.meminfo, self.low = floor_gib, meminfo, 1e9

    def __call__(self):
        m = mem_available_gib(self.meminfo)
        self.low = min(self.low, m)
        if m < self.floor:
            raise MemStop(f"MemAvailable {m:.2f} GiB < floor {self.floor} GiB")


class Reader:
    def __init__(self, dirs: dict):
        self.dirs, self.fds, self.hdr = dirs, {}, {}

    def _open(self, key: str, fn: str):
        if (key, fn) not in self.fds:
            path = os.path.join(self.dirs[key], fn)
            self.fds[(key, fn)] = os.open(path, os.O_RDONLY)
            self.hdr[(key, fn)] = read_header(path)
        return self.fds[(key, fn)], self.hdr[(key, fn)]

    def info(self, key: str, fn: str, name: str):
        return self._open(key, fn)[1][0][name]

    def raw(self, key: str, fn: str, name: str, row0: int = 0, row1: int | None = None) -> bytearray:
        """Rows [row0, row1) of one tensor's bytes (dim 0 is the row; a 0-d tensor is one row)."""
        fd, (h, base) = self._open(key, fn)
        info = h[name]
        a, b = info["data_offsets"]
        rows = info["shape"][0] if info["shape"] else 1
        per = (b - a) // max(rows, 1)
        row1 = rows if row1 is None else row1
        n, pos = (row1 - row0) * per, base + a + row0 * per
        buf = bytearray(n)
        mv, got = memoryview(buf), 0
        while got < n:
            k = os.preadv(fd, [mv[got:]], pos + got)
            if k <= 0:
                raise IOError(f"short read {fn}:{name} at {pos + got}")
            got += k
        return buf

    def drop_cache(self):
        for fd in self.fds.values():
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except (AttributeError, OSError):
                pass

    def close(self):
        self.drop_cache()
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()


def _rows_per_chunk(k: int) -> int:
    return max(1, CHUNK_ELEMS // max(k, 1))


def module_chunks(rd: Reader, op: dict, guard: Guard, stats: dict):
    """Yield (weight_u32_bytes, scales_bf16, biases_bf16) per row chunk of one quantized module."""
    (wn, _, wshape), _, _ = op["outs"]
    rows, k = wshape[0], wshape[1] * 8
    step = _rows_per_chunk(k)
    kind, bits = op["op"], op["bits_in"]
    src = op["src"]
    base = wn[: -len(".weight")]
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        guard()
        if kind == "quant_bf16":
            w = bf16_to_f32(np.frombuffer(rd.raw("orca", src[wn], wn, r0, r1), np.uint16).reshape(r1 - r0, k))
            codes, s, b = quantize(w)
            yield pack4(codes).tobytes(), f32_to_bf16(s), f32_to_bf16(b)
            continue
        wq = np.frombuffer(rd.raw("orca", src[wn], wn, r0, r1), np.uint32).reshape(r1 - r0, -1)
        s16 = np.frombuffer(rd.raw("orca", src[base + ".scales"], base + ".scales", r0, r1), np.uint16)
        b16 = np.frombuffer(rd.raw("orca", src[base + ".biases"], base + ".biases", r0, r1), np.uint16)
        s = f16_to_f32(s16).reshape(r1 - r0, -1)
        b = f16_to_f32(b16).reshape(r1 - r0, -1)
        if kind == "keep4":
            yield wq.tobytes(), f32_to_bf16(s), f32_to_bf16(b)
            continue
        codes = unpack(wq, bits)
        if bits <= 5:
            new, s2, b2 = requant_lut(codes, s, b, bits)
        else:
            new, s2, b2 = quantize(dequant(codes, s, b))
        if stats is not None and r0 == 0:          # first chunk of each requant: relative error vs the source
            src_w = dequant(codes, s, b)
            out_w = dequant(new, bf16_to_f32(f32_to_bf16(s2)), bf16_to_f32(f32_to_bf16(b2)))
            err = float(np.sqrt(np.mean((out_w - src_w) ** 2)) / (np.sqrt(np.mean(src_w ** 2)) + 1e-30))
            stats.setdefault(kind, []).append(err)
        yield pack4(new).tobytes(), f32_to_bf16(s2), f32_to_bf16(b2)


def write_shard(task):
    out_dir, fn, ops, dirs, floor, meminfo = task
    try:
        return _write_shard(out_dir, fn, ops, dirs, floor, meminfo)
    except MemStop as e:
        try:
            os.unlink(os.path.join(out_dir, fn + ".partial"))
        except FileNotFoundError:
            pass
        return {"file": fn, "stop": str(e)}


def _write_shard(out_dir, fn, ops, dirs, floor, meminfo):
    t0 = time.time()
    final = os.path.join(out_dir, fn)
    state = os.path.join(os.path.dirname(os.path.abspath(out_dir)), ".state", fn + ".json")   # STAGE/.state
    plan_sig = hashlib.sha256((CODE_SHA + json.dumps(ops, sort_keys=True)).encode()).hexdigest()
    if os.path.exists(final) and os.path.exists(state):
        st = json.load(open(state))
        if st.get("plan_sha256") == plan_sig and os.path.getsize(final) == st.get("bytes") \
                and sha256_file(final) == st.get("sha256"):
            st["resumed"] = True
            return st
    header, off = {}, 0
    for op in ops:
        for name, dt, shape in op["outs"]:
            n = nbytes(dt, shape)
            header[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + n]}
            off += n
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    guard = Guard(floor, meminfo)
    guard()                                           # no unit starts under the floor
    rd, stats = Reader(dirs), {}
    sha = hashlib.sha256()
    tmp = final + ".partial"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    written = 0

    def put(data):
        nonlocal written
        mv = memoryview(data).cast("B")
        sha.update(mv)
        while len(mv):
            k = os.write(fd, mv)
            mv = mv[k:]
            written += k

    try:
        put(struct.pack("<Q", len(hb)) + hb)
        for op in ops:
            kind = op["op"]
            if kind in ("copy", "graft"):
                name, dt, shape = op["outs"][0]
                key = "vontra" if kind == "graft" else "orca"
                info = rd.info(key, op["src"][name], name)
                if info["dtype"] != dt or list(info["shape"]) != list(shape):
                    raise SystemExit(f"STOP: {name}: source {info['dtype']}{info['shape']} vs plan {dt}{shape}")
                rows = shape[0] if shape else 1
                step = max(1, (64 << 20) // max(nbytes(dt, shape) // max(rows, 1), 1))
                for r0 in range(0, rows, step):
                    guard()
                    put(rd.raw(key, op["src"][name], name, r0, min(rows, r0 + step)))
                continue
            ss, bb = [], []
            for wbytes, s16, b16 in module_chunks(rd, op, guard, stats):
                put(wbytes)
                ss.append(s16)
                bb.append(b16)
            put(np.concatenate(ss).tobytes())
            put(np.concatenate(bb).tobytes())
        os.fsync(fd)
    finally:
        os.close(fd)
        rd.close()
    if written != 8 + len(hb) + off:
        raise SystemExit(f"STOP: {fn}: wrote {written} bytes, planned {8 + len(hb) + off}")
    os.rename(tmp, final)
    try:
        dfd = os.open(final, os.O_RDONLY)
        os.posix_fadvise(dfd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(dfd)
    except (AttributeError, OSError):
        pass
    st = {"file": fn, "bytes": written, "sha256": sha.hexdigest(), "tensors": len(header), "plan_sha256": plan_sig,
          "seconds": round(time.time() - t0, 1), "mem_low_gib": round(guard.low, 2),
          "rel_err": {k: [round(min(v), 5), round(sum(v) / len(v), 5), round(max(v), 5), len(v)]
                      for k, v in (stats or {}).items()}}
    os.makedirs(os.path.dirname(state), exist_ok=True)
    with open(state + ".tmp", "w") as f:
        json.dump(st, f)
    os.rename(state + ".tmp", state)
    return st


def sha256_file(path: str, bs: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bs)
            if not b:
                return h.hexdigest()
            h.update(b)


def read_sums(path: str) -> dict:
    """sha256sum format -> {basename: sha}."""
    out = {}
    for line in open(path):
        if line.strip():
            sha, fn = line.split(None, 1)
            out[os.path.basename(fn.strip().lstrip("*"))] = sha
    return out


def op_cost(op) -> float:
    (_, dt, shape) = op["outs"][0]
    n = nbytes(dt, shape)
    return {"requant5": 12.0, "requant6": 14.0, "requant8": 14.0, "quant_bf16": 10.0}.get(op["op"], 1.0) * n


def cmd_plan(a):
    orca = headers_from_json(a.orca_headers)[0] if a.orca_headers else headers_from_dir(a.orca)[0]
    target = load_target(a.target)
    shards, counts = build_plan(orca, target)
    diffs, n, tb, vb = plan_diff(shards, target)
    print(json.dumps({"tensors": n, "target": len(target), "shards": len(shards), "counts": counts,
                      "bytes": tb, "vontra_bytes": vb, "diffs": len(diffs), "first_diffs": diffs[:20]}, indent=1))
    if a.json_out:
        json.dump({"shards": shards, "counts": counts}, open(a.json_out, "w"))
    return 1 if diffs or tb != vb else 0


def cmd_run(a):
    t0 = time.time()
    orca, _ = headers_from_dir(a.orca)
    if len(orca) != a.expect_shards:
        raise SystemExit(f"STOP: {len(orca)} orcarouter shards in {a.orca}, expected {a.expect_shards}")
    target = load_target(a.target)
    shards, counts = build_plan(orca, target)
    diffs, n, tb, vb = plan_diff(shards, target)
    if diffs or tb != vb:
        raise SystemExit(f"STOP: plan header diff {len(diffs)} (bytes {tb} vs {vb}): {diffs[:20]}")
    log(f"plan ok: {n} tensors, {len(shards)} shards, {tb} bytes, counts {counts}")
    graft_fn = f"model-{OUT_TOTAL:05d}-of-{OUT_TOTAL:05d}.safetensors"
    for fn in {op["src"][k] for op in shards.get(graft_fn, []) for k in op["src"]}:
        if not os.path.exists(os.path.join(a.vontra, fn)):
            raise SystemExit(f"STOP: graft source {fn} missing in {a.vontra}")
    # small files and template are checked before any shard is written
    want = read_sums(a.small_sha)
    for fn in SMALL_FROM_VONTRA:
        src = os.path.join(a.vontra, fn)
        if not os.path.exists(src):
            raise SystemExit(f"STOP: {fn} missing in {a.vontra} (abl-fetch.sh fetches it)")
        if sha256_file(src) != want.get(fn):
            raise SystemExit(f"STOP: {src} sha256 is not the pinned {want.get(fn)}")
    if sha256_file(a.template) != a.template_sha:
        raise SystemExit(f"STOP: template {a.template} does not hash to the stack template {a.template_sha}")
    wdir, sdir = os.path.join(a.out, "weights"), os.path.join(a.out, ".state")
    os.makedirs(wdir, exist_ok=True)
    legacy = os.path.join(wdir, ".state")         # Keep resume state outside the weight-file set
    if os.path.isdir(legacy):
        os.rename(legacy, os.path.join(a.out, f".state-legacy-{int(time.time())}"))
    for f in os.listdir(wdir):                    # a killed run leaves partials; they are never resumed from
        if f.endswith(".partial"):
            os.unlink(os.path.join(wdir, f))
    stop_file = os.path.join(a.out, "STOP.json")
    if os.path.exists(stop_file):
        os.rename(stop_file, stop_file + f".{int(time.time())}")
    only = set(a.shards.split(",")) if a.shards else None
    dirs = {"orca": a.orca, "vontra": a.vontra}
    tasks = sorted(((wdir, fn, ops, dirs, a.mem_floor_gib, a.meminfo) for fn, ops in shards.items()
                    if only is None or fn in only), key=lambda t: -sum(op_cost(op) for op in t[2]))
    log(f"writing {len(tasks)} shards with {a.workers} workers, floor {a.mem_floor_gib} GiB via {a.meminfo}")
    results, stops = [], []
    if a.workers <= 1:
        for t in tasks:
            st = write_shard(t)
            if "stop" in st:
                stops.append(st)
                break
            results.append(st)
            log(f"done {st['file']} {st['seconds']}s ({len(results)}/{len(tasks)}) err {st['rel_err']}")
    else:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.workers, maxtasksperchild=4) as pool:
            for st in pool.imap_unordered(write_shard, tasks, chunksize=1):
                if "stop" in st:
                    stops.append(st)
                    pool.terminate()                  # release every worker's memory now; resume later
                    break
                results.append(st)
                log(f"done {st['file']} {st['seconds']}s ({len(results)}/{len(tasks)}) err {st['rel_err']}")
    if stops:
        for f in os.listdir(wdir):
            if f.endswith(".partial"):
                os.unlink(os.path.join(wdir, f))
        info = {"stopped": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "reason": stops[0]["stop"],
                "unit": stops[0]["file"], "shards_done": len(results), "shards_total": len(tasks)}
        json.dump(info, open(stop_file, "w"), indent=1)
        log(f"STOP {info}")
        return 3
    if only is not None:
        return 0
    for fn in SMALL_FROM_VONTRA:
        with open(os.path.join(a.vontra, fn), "rb") as f, open(os.path.join(wdir, fn), "wb") as g:
            g.write(f.read())
    with open(a.template, "rb") as f, open(os.path.join(wdir, "chat_template.jinja"), "wb") as g:
        g.write(f.read())
    wm = {name: fn for fn, ops in shards.items() for op in ops for name, _, _ in op["outs"]}
    with open(os.path.join(wdir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": tb}, "weight_map": dict(sorted(wm.items()))}, f, indent=2)
    # Content identity: every weights file hashed; shard hashes come from the streamed write
    by_file = {st["file"]: st["sha256"] for st in results}
    names = sorted(os.listdir(wdir))
    extra = [f for f in names if f not in by_file and f not in SMALL_FROM_VONTRA
             and f not in ("chat_template.jinja", "model.safetensors.index.json")]
    if extra or len(by_file) != len(shards):
        raise SystemExit(f"STOP: weights dir holds unexpected files {extra[:10]} or {len(by_file)} != {len(shards)}")
    sums = "".join(f"{by_file.get(f) or sha256_file(os.path.join(wdir, f))}  weights/{f}\n" for f in names)
    with open(os.path.join(a.out, "SHA256SUMS"), "w") as f:
        f.write(sums)
    ident = {"schema": "abl-identity/1", "source": ORCA_REV, "graft": VONTRA_REV + " layers.45",
             "source_sums_sha256": sha256_file(a.source_sums) if a.source_sums else None,
             "codec_version": CODEC_VERSION, "codec_sha256": CODE_SHA, "numpy": np.__version__,
             "rule_table_sha256": hashlib.sha256(json.dumps(shards, sort_keys=True).encode()).hexdigest(),
             "counts": counts, "tensors": n, "payload_bytes": tb, "files": len(names),
             "sha256sums_sha256": hashlib.sha256(sums.encode()).hexdigest()}
    ib = (json.dumps(ident, indent=1, sort_keys=True) + "\n").encode()
    with open(os.path.join(a.out, "IDENTITY.json"), "wb") as f:
        f.write(ib)
    cid = hashlib.sha256(ib).hexdigest()
    with open(os.path.join(a.out, "CONVERSION-ID"), "w") as f:
        f.write(cid + "\n")
    errs = {}
    for st in results:
        for k, (lo, mean, hi, c) in st["rel_err"].items():
            e = errs.setdefault(k, [9e9, 0.0, 0.0, 0])
            e[0], e[2] = min(e[0], lo), max(e[2], hi)
            e[1] += mean * c
            e[3] += c
    run = {"conversion_id": cid, "identity": ident,
           "rel_err_first_chunk": {k: {"min": v[0], "mean": v[1] / max(v[3], 1), "max": v[2], "n": v[3]}
                                   for k, v in errs.items()},
           "shards": sorted(results, key=lambda s: s["file"]), "resumed": sum(1 for s in results if s.get("resumed")),
           "wall_s": round(time.time() - t0, 1), "workers": a.workers, "mem_floor_gib": a.mem_floor_gib,
           "mem_low_gib": min((s.get("mem_low_gib", 1e9) for s in results), default=None),
           "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(a.out, "REPACK-RUN.json"), "w") as f:
        json.dump(run, f, indent=1)
    log(f"REPACK-DONE {n} tensors {tb} bytes in {run['wall_s']}s conversion-id {cid}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("plan")
    q.add_argument("--orca")
    q.add_argument("--orca-headers")
    q.add_argument("--target", required=True)
    q.add_argument("--json-out")
    r = sub.add_parser("run")
    r.add_argument("--orca", required=True)
    r.add_argument("--vontra", required=True)
    r.add_argument("--target", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--template", required=True)
    r.add_argument("--template-sha", default=STACK_TEMPLATE_SHA)
    r.add_argument("--small-sha", required=True, help="sha256sum file with the pinned Vontra small files")
    r.add_argument("--source-sums", help="orcarouter per-file LFS sha256 list, hashed into the identity")
    r.add_argument("--expect-shards", type=int, default=62)
    r.add_argument("--meminfo", default="/proc/meminfo", help="override for the guard's negative control")
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--mem-floor-gib", type=float, default=10.0)
    r.add_argument("--shards", help="comma list of output shard names (test runs; no index or receipt)")
    a = p.parse_args(argv)
    return cmd_plan(a) if a.cmd == "plan" else cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
