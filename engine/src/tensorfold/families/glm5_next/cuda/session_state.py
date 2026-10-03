"""Canonical committed GLM state, shared by session persistence and its exactness gate."""

from __future__ import annotations

import hashlib
import json

from .disk_io import tensor_chunks


def target_rows(st, n: int, m: int):
    """Name only committed rows. The index pool's extra fence is future scratch."""
    rows = [(f"kc{i}", c[:n]) for i, c in enumerate(st.kc)]
    rows += [(f"vc{i}", c[:n]) for i, c in enumerate(st.vc) if c is not None]
    for i, (ik, ig, pk) in enumerate((st.index or [])[:len(st.kc)]):
        rows += [(f"ik{i}", ik[:n]), (f"ig{i}", ig[:n]), (f"pk{i}", pk[:n // 4])]
    if m > 0 and hasattr(st, "mtp_kc"):
        rows.append(("mtp_kc", st.mtp_kc[:m]))
        if getattr(st, "mtp_vc", None) is not None:
            rows.append(("mtp_vc", st.mtp_vc[:m]))
        if len(st.index or []) > len(st.kc):
            ik, ig, pk = st.index[-1]
            rows += [("mtp_ik", ik[:m]), ("mtp_ig", ig[:m]), ("mtp_pk", pk[:m // 4])]
    return rows


def fixed_state(snap):
    parts = [("rec", snap.rec), ("conv", snap.conv)]
    if snap.pending is not None:
        parts.append(("pending", snap.pending))
    parts += [(f"draft{i}", t) for i, t in enumerate(snap.drafter_rows or [])]
    return parts


def saved_target_rows(e, source, snap):
    """Read immutable committed rows from the memory cache, never a reusable live span."""
    from .disk_io import TensorParts

    if source.rows is None or source.ids[:len(snap.ids)] != snap.ids:
        raise ValueError("disk write requires an owned memory prefix")
    names = target_rows(e.st, len(source.ids), max(source.mtp_len, 0))
    owned = {name: TensorParts(parts if isinstance(parts, tuple) else (parts,))
             for (name, _), parts in zip(names, source.rows, strict=True)}
    return [(name, owned[name].slice(0, view.shape[0]))
            for name, view in target_rows(e.st, len(snap.ids), max(snap.mtp_len, 0))]


def image_count(ids) -> int:
    return sum(token < 0 and (i == 0 or ids[i - 1] != token) for i, token in enumerate(ids))


def snapshot_key(ids, digests=()) -> str:
    import numpy as np

    h = hashlib.sha256(np.asarray(ids, dtype="<i4").tobytes())
    h.update(b"".join(digests))
    return h.hexdigest()


def state_hash(e, snap) -> str:
    """Hash observable state, independent of slot, pointers, segmentation and unused scratch.

    Call with live target rows and a freshly captured/restored Snapshot. Compare
    the per-rank hashes of a fresh prefill with a disk-resumed prefill of the same
    prompt. Signed zero/NaN payloads are hashed as bytes, with no numeric casts.
    This is an opt-in gate, not a checksum-based claim about fresh CUDA arithmetic.
    """
    h = hashlib.sha256()
    h.update(json.dumps({"ids": snap.ids, "mtp_len": snap.mtp_len, "drafter_end": snap.drafter_end,
                         "images": [d.hex() for d in snap.image_digests]}, sort_keys=True).encode())
    for name, t in target_rows(e.st, len(snap.ids), max(snap.mtp_len, 0)) + fixed_state(snap):
        h.update(json.dumps([name, str(t.dtype), list(t.shape)]).encode())
        for raw in tensor_chunks(t):
            h.update(raw)
    return h.hexdigest()


def refresh_transferred_pool_fences(e, old, drafter=None) -> None:
    """Refresh the parent's unused fence before transferring ownership to a child.

    Legacy saved views include pk[:n//4+1]. Only pk[:n//4] was committed at
    the parent; its last copied row can become a completed pool in the child.
    extend_rows transfers the allocation, so rewriting that formerly unused row
    costs no allocation and leaves the parent's semantic state unchanged. The
    worker already restored any active readers before replacing this cache entry.
    """
    from .decode import _snapshot_row_views

    pools = {pk.data_ptr() for _, _, pk in e.st.index or []}
    for live, saved in zip(_snapshot_row_views(e, old, drafter), old.rows, strict=True):
        if live.data_ptr() in pools:
            parts = saved if isinstance(saved, tuple) else (saved,)
            last = next(part for part in reversed(parts) if part.shape[0])
            last[-1].copy_(live[-1])
