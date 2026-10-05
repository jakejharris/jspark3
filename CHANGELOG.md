# JSpark3 v2.0.2 (draft)

v2.0.1 plus two image fixes:

- Keep a session checkpoint before the assistant/tool turn containing the oldest retained image. Exact prompt
  and image matching remain required. The root cause is VERIFIED offline; live validation is pending.
- Accept GIF input and decode frame zero, retaining the existing size and error handling. The supplied-file
  offline HTTP repro changed from 400 to 200; the first live GIF request passed.

The first live swap at 17:37Z on 2026-10-05 failed its rolling-image check: the eight-image request resumed
0 tokens and did not return the expected color. The swap was rolled back. The draft stays open until a live
retry passes. [Evidence and limits](release/v2.0.2/ROOTCAUSE.md).

There are no new speed measurements. Previous benchmark files remain labeled v2.0.1. Other previously
planned fixes are outside this release. The wheel content, release metadata and session identity are repinned.

# JSpark3 v2.0.1

A new engine and new weights for GLM-5.3 Flash on three DGX Sparks.

- **Engine:** a fork of TensorFold 0.3.6.2 (MIT) replaces vLLM. Its source is vendored in `engine/`, and the wheel is
  built from it inside the pinned NVIDIA container, without network, and checked against a pinned content digest.
- **Weights:** public 4-bit MLX-format GLM-5.3 Flash weights (MIT), downloaded at a pinned revision, checked file by
  file, and split into one part per box that must match the shipped per-rank manifests. Optional refusal-removed
  (abliterated) weights for ablit development, red-teaming and refusal research, converted on your machine from a
  gated source with your own Hugging Face token.
- **Draft model:** DFlash2 by default; `--drafter none` runs without it, drafting with the weights' own multi-token
  prediction head.
- **Session tier:** conversation state is kept on each box's disk (up to 64 GiB) so long conversations resume
  without recomputing; `SESSION_TIER=off` turns it off.
- **Install:** one script per step, each with `--help` and `--dry-run`; a preflight for each stage; a readiness wait
  that returns when all three ranks answer; failure messages that name what failed and what to do without your
  addresses, paths or tokens, with detail in private logs.
- **Results:** three measured sets (base and ablit weights with the draft model, base weights without it).
  v2.0.1 also saves conversation state to each Spark's disk by default: with base weights and the draft model, a conversation of at least 100,000 tokens that had been pushed out of memory showed its first visible text 1.5 s after it was continued, against 49.9 s to read it from scratch.

v2.0.0: internal build, not published.

The previous release, v1.8.4 (vLLM), remains the rollback ([UPGRADING.md](UPGRADING.md)).
