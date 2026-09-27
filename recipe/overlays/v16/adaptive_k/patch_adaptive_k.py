#!/usr/bin/env python3
"""Fail-closed source patch for the post-v1.5 scheduler and target CG family."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


SCHEDULER_V15_SHA256 = "0b086dd3cc0febc4924202ca1fef8809b8290b9a59ef15fa80004cef4edeb937"
CUDAGRAPH_V15_SHA256 = "c183937e6eb5b9c28c79d98fb4c64f562e7649d5f6d65743e6640b2f378ecf9f"
SCHEDULER_EMA_SHA256 = "0d58e688019ddfa5952990be2ea751b71d96bea7a35c4c31c19c672b0890f764"
CUDAGRAPH_EMA_SHA256 = "6b44f24e65e51a0a43c7a5d7d93ef5def8b880cff9cb671ba1e1302a79c757aa"
SCHED_MARKER = "# [jspark3-v16:adaptive-k scheduler]"
CG_MARKER = "# [jspark3-v16:adaptive-k target graphs]"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"{label}: anchor count {count}, expected 1")
    return text.replace(old, new, 1)


def patch_scheduler(text: str) -> str:
    if SCHED_MARKER in text:
        if text.count(SCHED_MARKER) != 1 or text.count("_GLM53_ADAPTIVE_K.") != 3:
            raise ValueError("adaptive-k scheduler marker/hook drift")
        return text

    import_anchor = "from vllm.compilation.cuda_graph import CUDAGraphStat\n"
    text = _replace_once(
        text,
        import_anchor,
        import_anchor
        + "from vllm.v1.core.sched.jspark3_adaptive_k import (\n"
        + "    POLICY as _GLM53_ADAPTIVE_K,\n"
        + f")  {SCHED_MARKER}\n",
        "scheduler import",
    )

    schedule_anchor = "        # Check if the scheduling constraints are satisfied.\n"
    text = _replace_once(
        text,
        schedule_anchor,
        "        if self.dynamic_sd_lookup is not None:\n"
        + "            raise RuntimeError('adaptive-k cannot compose with native dynamic SD')\n"
        + "        _GLM53_ADAPTIVE_K.prepare_batch(\n"
        + "            self, num_scheduled_tokens, scheduled_spec_decode_tokens,\n"
        + "            scheduled_new_reqs, scheduled_resumed_reqs,\n"
        + "        )\n\n"
        + schedule_anchor,
        "current batch trim before accounting",
    )

    rollback_anchor = """                if not output_is_stale:
                    if request.num_computed_tokens > 0:
                        request.num_computed_tokens -= num_rejected
                    if request.num_output_placeholders > 0:
                        request.num_output_placeholders -= num_rejected
"""
    text = _replace_once(
        text,
        rollback_anchor,
        rollback_anchor
        + "                    _GLM53_ADAPTIVE_K.observe_output(\n"
        + "                        scheduler_output, req_id, num_draft_tokens, num_accepted\n"
        + "                    )\n",
        "accept observation",
    )

    bind_anchor = '        _GLM53_MIXED.finish_step(self, scheduler_output)  # [glm53-decode-floor:v5]\n'
    text = _replace_once(
        text,
        bind_anchor,
        "        _GLM53_ADAPTIVE_K.bind_output(self, scheduler_output)\n" + bind_anchor,
        "observation snapshot before async accounting",
    )
    return text


def patch_cudagraph(text: str) -> str:
    if CG_MARKER in text:
        if (
            text.count(CG_MARKER) != 1
            or text.count("_glm53_adaptive_query_lens") != 2
            or text.count("_glm53_adaptive_capture_shape") != 2
        ):
            raise ValueError("adaptive-k cudagraph marker/hook drift")
        return text

    logger_anchor = "logger = init_logger(__name__)\n\n\n"
    helper = f'''logger = init_logger(__name__)\n\n\n{CG_MARKER}
def _glm53_adaptive_query_lens(manager, query_lens):
    # The compact DFlash2 drafter keeps its sealed query-8 graph family.
    # Only the target verifier gains B5's M=4 shape.
    if manager.__class__.__name__ != "ModelCudaGraphManager":
        return query_lens
    if os.environ.get("GLM53_ADAPTIVE_K") != "ema":
        return query_lens
    return sorted(set((*query_lens, 4)))


def _glm53_adaptive_capture_shape(manager, num_tokens, query_len):
    # C9+ is outside this lane's operational envelope. In particular, do not
    # turn the retained full-width M48 capture into an unreachable query-4 C12.
    return not (
        manager.__class__.__name__ == "ModelCudaGraphManager"
        and os.environ.get("GLM53_ADAPTIVE_K") == "ema"
        and query_len == 4
        and num_tokens > 32
    )


'''
    # cudagraph_utils does not otherwise import os in the sealed image.
    text = _replace_once(text, "import gc\n", "import gc\nimport os\n", "cudagraph os import")
    text = _replace_once(text, logger_anchor, helper, "cudagraph helper")
    lens_anchor = """        else:
            decode_query_lens = [self.decode_query_len]

        capture_varlen_decode = (
"""
    text = _replace_once(
        text,
        lens_anchor,
        "        else:\n"
        + "            decode_query_lens = [self.decode_query_len]\n"
        + "        decode_query_lens = _glm53_adaptive_query_lens(\n"
        + "            self, decode_query_lens\n"
        + "        )\n\n"
        + "        capture_varlen_decode = (\n",
        "cudagraph query lengths",
    )
    loop_anchor = "                for decode_query_len in decode_query_lens:\n"
    text = _replace_once(
        text,
        loop_anchor,
        loop_anchor
        + "                    if not _glm53_adaptive_capture_shape(\n"
        + "                        self, num_tokens, decode_query_len\n"
        + "                    ):\n"
        + "                        continue\n",
        "cudagraph adaptive capture envelope",
    )
    return text


def patch_bytes(scheduler: bytes, cudagraph: bytes) -> tuple[bytes, bytes]:
    if sha(scheduler) not in (SCHEDULER_V15_SHA256, SCHEDULER_EMA_SHA256):
        raise ValueError("adaptive-k requires exact v1.5 or sealed ema scheduler bytes")
    if sha(cudagraph) not in (CUDAGRAPH_V15_SHA256, CUDAGRAPH_EMA_SHA256):
        raise ValueError("adaptive-k requires exact v1.5 or sealed ema cudagraph bytes")
    return (
        patch_scheduler(scheduler.decode("utf-8")).encode("utf-8"),
        patch_cudagraph(cudagraph.decode("utf-8")).encode("utf-8"),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=Path, required=True)
    parser.add_argument("--cudagraph", type=Path, required=True)
    args = parser.parse_args()
    try:
        scheduler, cudagraph = patch_bytes(
            args.scheduler.read_bytes(), args.cudagraph.read_bytes()
        )
        args.scheduler.write_bytes(scheduler)
        args.cudagraph.write_bytes(cudagraph)
        return 0
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=os.sys.stderr)
        return 9


if __name__ == "__main__":
    raise SystemExit(main())
