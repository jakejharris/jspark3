# DRAFT social posts

Status: drafts, not posted. Every number must match `results/results.json`;
the release validator checks this file. Posting remains a separate maintainer
action.

## Long form (Mastodon, LinkedIn, Bluesky thread opener)

JSpark3 v1: one GLM-5.3 Flash endpoint across three NVIDIA DGX Sparks. TP3 +
EP3 over a RoCE-v2 triangle, EXL3 4-bpw target, DFlash2 draft, and a
selective INT8 trunk overlay that frees 1,595,392,320 bytes per rank.

Where it sits next to what was already public, using each author's own
reported numbers and computing no deltas against them: FlyCockpit TP3 reports
structured count 69.0 / 68.5 / 71.2 tok/s; Mia TP2 reports a sparkDash C1 of
62.9; jetnet TP3 reports 35.2 at 512K with MTP-4. Different node counts,
quantization lanes, contexts, and estimators, so they are context, not a
scoreboard.

We also ran three of those recipes ourselves, all adapted and labelled as
such, at 24.913, 24.728, and 29.042 tok/s on a shared agent prompt with
independent trajectories. We never ran jetnet.

Internally, against the same recipe with the overlay disabled, an unreleased
development control: single-stream code decode 63.861 to 66.257 tok/s
(+3.75%), same-day paired +7.27%; smoother pacing (median interval -6.83%,
worst interval -59.29%); C48 aggregate +3.47%. Also measured: long prefill
-3.38%, three-stream waves variable, and two of our own promotion gates
missed. All of it is in the benchmarks page with estimators and receipts,
because a number you cannot trace is not a result.

The GitHub recipe/assets contain no checkpoint weights and no patched vLLM.
The attributed, byte-identical Hugging Face target mirror is authorized;
metadata is live, while its weight transfer is in progress on a separate
review branch and not merged. Every serving input is pinned and verified.
Apache-2.0 for our code; the checkpoint is
attribution-required and the draft is non-commercial, so this is not
commercial-ready.

Release: https://github.com/jakejharris/jspark3/releases/tag/v1.0.0

## Short form (X)

JSpark3 v1: GLM-5.3 Flash on three DGX Sparks as one endpoint. TP3+EP3, EXL3
4-bpw, DFlash2, INT8 trunk overlay. Public recipes before it reported their
own numbers, and we quote them without deltas; our own three adapted local
reproductions and the internal overlay A/B are published separately, with the
prefill regression and two missed gates left in. GitHub ships recipe/assets;
the attributed unchanged Hugging Face target mirror is authorized and its
weight transfer is pending. Everything is pinned. Apache-2.0 code; upstream
terms apply.
github.com/jakejharris/jspark3

## Thread follow-ups

1. Why three is awkward: 64 heads, a 154,880-row vocabulary, and a 2,048-wide
   shared expert do not divide by three. Padding to 66, 154,944, and 2,112 at
   runtime fixes the layout without touching a checkpoint byte. Credit to the
   FlyCockpit lineage.
2. No patched vLLM ships. Five transactional transform programs rebuild the
   pinned upstream changes inside the container at start, hash-checked before
   and after.
3. The recipe refuses to start on any drift: image digest, checkpoint bytes,
   overlay hash, cgroup limits, NCCL overrides. Preflight rows are compared
   byte for byte.
4. Credits: Z.AI, Brandon M. Music and Mia-AiLab (ShapleyMcg EXL3/TR3), Inco AI
   (DFlash2), MiaAI-Lab, FlyCockpit, vcruz305, vLLM, ExLlamaV3.
