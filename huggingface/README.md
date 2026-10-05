---
license: other
license_name: shapleymcg-license-1.0
license_link: LICENSE
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
language:
  - en
  - zh
tags:
  - glm
  - glm-5.3-flash
  - dgx-spark
  - serving-recipe
  - tensorfold
  - speculative-decoding
  - dflash
  - shapleymcg
---

<div style="display:flex;align-items:center;gap:14px;margin:6px 0 2px;">
<img src="https://huggingface.co/jakejharris/jspark3/resolve/1cf9f930085d3e4c3880b438964ff11db2bc1a70/jspark3/assets/jspark3-mark.svg" width="48" height="48" alt="">
<h1 style="margin:0;font-size:2.4em;line-height:1;letter-spacing:-0.01em;">JSpark3 <span style="font-weight:500;color:#8b8b90;">v2.0.2 · GLM-5.3 Flash</span></h1>
</div>

**JSpark3 v2.0.2 fixes image-history checkpoint reuse and adds GIF frame-zero support.**

Live 8/9/10-image resumes were 0 -> 68 -> 174 tokens, both nonzero hits from disk; GIF passed and smoke passed 6/6. Acceptance waited for durable checkpoints; continuous traffic can still skip optional disk saves.

The live image-cache acceptance passed on the prepared runtime. A clean installation of the final v2.0.2 public recipe has not been demonstrated; the installation and performance receipts remain v2.0.1 evidence. No new speed measurements are claimed. See the [live evidence](https://github.com/jakejharris/jspark3/blob/v2.0.2/release/v2.0.2/ROOTCAUSE.md).

<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:14px;margin:18px 0 8px;">
  <div style="background:linear-gradient(160deg,#1d1d20 0%,#0e0e10 100%);border:1px solid rgba(255,255,255,0.09);border-radius:12px;padding:18px 20px 16px;box-shadow:2px 2px 0 #f0a8a8,inset 0 1px 0 rgba(255,255,255,0.05);">
    <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9a9a9f;">Code, decode estimate</div>
    <div style="margin-top:10px;font-size:36px;line-height:1;font-weight:800;letter-spacing:-0.02em;color:#f0a8a8;">91.3 <span style="font-size:18px;font-weight:600;letter-spacing:0;color:#c9c9ce;">tok/s</span></div>
    <div style="margin-top:8px;font-size:13px;line-height:1.45;color:#a4a4a9;">v1.8.4: 61.1 tok/s</div>
  </div>
  <div style="background:linear-gradient(160deg,#1d1d20 0%,#0e0e10 100%);border:1px solid rgba(255,255,255,0.09);border-radius:12px;padding:18px 20px 16px;box-shadow:2px 2px 0 #f2c6a6,inset 0 1px 0 rgba(255,255,255,0.05);">
    <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9a9a9f;">Prose, decode estimate</div>
    <div style="margin-top:10px;font-size:36px;line-height:1;font-weight:800;letter-spacing:-0.02em;color:#f2c6a6;">51.6 <span style="font-size:18px;font-weight:600;letter-spacing:0;color:#c9c9ce;">tok/s</span></div>
    <div style="margin-top:8px;font-size:13px;line-height:1.45;color:#a4a4a9;">v1.8.4: 31.5 tok/s</div>
  </div>
  <div style="background:linear-gradient(160deg,#1d1d20 0%,#0e0e10 100%);border:1px solid rgba(255,255,255,0.09);border-radius:12px;padding:18px 20px 16px;box-shadow:2px 2px 0 #a8e6c7,inset 0 1px 0 rgba(255,255,255,0.05);">
    <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9a9a9f;">Cold prefill, 64K prompt</div>
    <div style="margin-top:10px;font-size:36px;line-height:1;font-weight:800;letter-spacing:-0.02em;color:#a8e6c7;">2,124 <span style="font-size:18px;font-weight:600;letter-spacing:0;color:#c9c9ce;">tok/s</span></div>
    <div style="margin-top:8px;font-size:13px;line-height:1.45;color:#a4a4a9;">v1.8.4: 1,505 tok/s</div>
  </div>
  <div style="background:linear-gradient(160deg,#1d1d20 0%,#0e0e10 100%);border:1px solid rgba(255,255,255,0.09);border-radius:12px;padding:18px 20px 16px;box-shadow:2px 2px 0 #a6d2f2,inset 0 1px 0 rgba(255,255,255,0.05);">
    <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9a9a9f;">Four at once, end to end</div>
    <div style="margin-top:10px;font-size:36px;line-height:1;font-weight:800;letter-spacing:-0.02em;color:#a6d2f2;">113.4 <span style="font-size:18px;font-weight:600;letter-spacing:0;color:#c9c9ce;">tok/s</span></div>
    <div style="margin-top:8px;font-size:13px;line-height:1.45;color:#a4a4a9;">short code, end-to-end, 256-token cap per agent · v1.8.4: 86.6 tok/s</div>
  </div>
</div>
<p style="font-size:12.5px;line-height:1.5;color:#8b8b90;margin:0 0 18px;">RigMark, v2.0.1 with base weights + draft model · base profile (tok/s, higher is better), with the v1.8.4 figure under each one.</p>

Appliance comparison: different model IDs, not a same-weights claim. v1.8.4 ran with reasoning off, its default; v2.0.1 ran at reasoning effort low. Cold prefill and replay rows use raw token-ID completions, where reasoning effort does not apply. With base weights, v1.8.4 shows the first visible text sooner in two measured cases (about 2% on single-client short code replies, about 0.10 s on prose); both rows are in the RigMark section below.

> **v2.0.2 does not use the EXL3 files in this repository.** Those files are the v1.8.x weights, kept in the [`v1.8.x-exl3/`](https://huggingface.co/jakejharris/jspark3/tree/main/v1.8.x-exl3) folder for the v1.8.4 rollback. v2.0.2 downloads public weights from [`TensorFold/GLM-5.3-Flash-MLX-4bit-MTP`](https://huggingface.co/TensorFold/GLM-5.3-Flash-MLX-4bit-MTP/tree/76add2a341a1cd90ad0e86bb69839ea9c35827c6) (formerly `Vontra/GLM-5.3-Flash-MLX-4bit-MTP`, which redirects) at revision `76add2a341a1cd90ad0e86bb69839ea9c35827c6`, verifies every file against a pinned SHA-256 list, and splits them across your three Sparks on your own machine. Use the [v2.0.2 installation guide](https://github.com/jakejharris/jspark3/blob/v2.0.2/INSTALL.md), [known limitations](https://github.com/jakejharris/jspark3/blob/v2.0.2/LIMITATIONS.md), and [release on GitHub](https://github.com/jakejharris/jspark3/releases/tag/v2.0.2).

<p>
  <a href="https://github.com/jakejharris/jspark3/releases/tag/v2.0.2"><img src="https://img.shields.io/badge/release-v2.0.2-0a7c3f?style=for-the-badge&amp;logo=github&amp;logoColor=white" alt="Release v2.0.2"></a>
  <a href="https://github.com/jakejharris/jspark3"><img src="https://img.shields.io/badge/hardware-3%C3%97_DGX_Spark-76b900?style=for-the-badge&amp;logo=nvidia&amp;logoColor=white" alt="three NVIDIA DGX Sparks"></a>
  <a href="https://github.com/ashhart/TensorFold"><img src="https://img.shields.io/badge/engine-fork_of_TensorFold_0.3.6.2_%28MIT%29-8250df?style=for-the-badge" alt="Engine: a fork of TensorFold 0.3.6.2 (MIT)"></a>
  <a href="https://github.com/jakejharris/jspark3"><img src="https://img.shields.io/badge/recipe_license-Apache--2.0-d73a49?style=for-the-badge" alt="Recipe license Apache-2.0"></a>
  <a href="https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2"><img src="https://img.shields.io/badge/draft_model-DFlash2_non--commercial-6e7781?style=for-the-badge" alt="DFlash2 draft model, non-commercial license"></a>
</p>

- **80.9 tok/s** decode on one stream on our own benchmark (short code replies).
- **97.5 tok/s** aggregate decode on our own benchmark (short prompts, 41-62 tokens, 4 concurrent).
- **1.5 s** to first visible text when you continue a conversation of at least 100,000 tokens that had been pushed out of memory, against 49.9 s to read it from scratch (see known issue 17).
- **Images in chat:** up to 16 per request, as inline `data:` URLs.
- **Two weight variants, one switch:** MIT base weights by default, or opt-in refusal-removed (ablit) weights.

All performance figures above are historical v2.0.1 measurements: base weights + draft model unless marked. Every figure, with its conditions, is in the RigMark and Results sections below.

JSpark3 serves GLM-5.3 Flash on three NVIDIA DGX Sparks, connected by a direct high-speed (RDMA) link, as one OpenAI-compatible endpoint. v2.0.1 replaces the vLLM engine of v1.8.x with [a fork of TensorFold 0.3.6.2 (MIT)](https://github.com/ashhart/TensorFold). v2.0.1 also saves conversation state to each Spark's disk by default: with base weights and the draft model, a conversation of at least 100,000 tokens that had been pushed out of memory showed its first visible text 1.5 s after it was continued, against 49.9 s to read it from scratch (see known issue 17).

<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:18px 0 8px;">
  <div style="background:linear-gradient(160deg,#1d1d20 0%,#0e0e10 100%);border:1px solid rgba(255,255,255,0.09);border-radius:12px;padding:18px 20px 16px;box-shadow:2px 2px 0 #f0a8a8,inset 0 1px 0 rgba(255,255,255,0.05);">
    <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9a9a9f;">Decode per stream, code / prose</div>
    <div style="margin-top:10px;font-size:40px;line-height:1;font-weight:800;letter-spacing:-0.02em;color:#f0a8a8;">80.9 / 59.6 <span style="font-size:18px;font-weight:600;letter-spacing:0;color:#c9c9ce;">tok/s</span></div>
    <div style="margin-top:8px;font-size:13px;line-height:1.45;color:#a4a4a9;">base weights + draft model, short replies · base profile</div>
  </div>
  <div style="background:linear-gradient(160deg,#1d1d20 0%,#0e0e10 100%);border:1px solid rgba(255,255,255,0.09);border-radius:12px;padding:18px 20px 16px;box-shadow:2px 2px 0 #f2c6a6,inset 0 1px 0 rgba(255,255,255,0.05);">
    <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9a9a9f;">Decode per stream, code / prose</div>
    <div style="margin-top:10px;font-size:40px;line-height:1;font-weight:800;letter-spacing:-0.02em;color:#f2c6a6;">73.3 / 63.6 <span style="font-size:18px;font-weight:600;letter-spacing:0;color:#c9c9ce;">tok/s</span></div>
    <div style="margin-top:8px;font-size:13px;line-height:1.45;color:#a4a4a9;">refusal-removed weights + draft model, short replies · ablit profile</div>
  </div>
  <div style="background:linear-gradient(160deg,#1d1d20 0%,#0e0e10 100%);border:1px solid rgba(255,255,255,0.09);border-radius:12px;padding:18px 20px 16px;box-shadow:2px 2px 0 #f2e3a6,inset 0 1px 0 rgba(255,255,255,0.05);">
    <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9a9a9f;">Decode per stream, code / prose</div>
    <div style="margin-top:10px;font-size:40px;line-height:1;font-weight:800;letter-spacing:-0.02em;color:#f2e3a6;">63.6 / 57.7 <span style="font-size:18px;font-weight:600;letter-spacing:0;color:#c9c9ce;">tok/s</span></div>
    <div style="margin-top:8px;font-size:13px;line-height:1.45;color:#a4a4a9;">base weights, no draft model, short replies · base profile</div>
  </div>
  <div style="background:linear-gradient(160deg,#1d1d20 0%,#0e0e10 100%);border:1px solid rgba(255,255,255,0.09);border-radius:12px;padding:18px 20px 16px;box-shadow:2px 2px 0 #a8e6c7,inset 0 1px 0 rgba(255,255,255,0.05);">
    <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9a9a9f;">Aggregate decode</div>
    <div style="margin-top:10px;font-size:40px;line-height:1;font-weight:800;letter-spacing:-0.02em;color:#a8e6c7;">126.0 <span style="font-size:18px;font-weight:600;letter-spacing:0;color:#c9c9ce;">tok/s</span></div>
    <div style="margin-top:8px;font-size:13px;line-height:1.45;color:#a4a4a9;">short prompts, 41-62 tokens, 8 concurrent, default configuration · base profile</div>
  </div>
  <div style="background:linear-gradient(160deg,#1d1d20 0%,#0e0e10 100%);border:1px solid rgba(255,255,255,0.09);border-radius:12px;padding:18px 20px 16px;box-shadow:2px 2px 0 #a6d2f2,inset 0 1px 0 rgba(255,255,255,0.05);">
    <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9a9a9f;">Cold time to first token</div>
    <div style="margin-top:10px;font-size:40px;line-height:1;font-weight:800;letter-spacing:-0.02em;color:#a6d2f2;">63.6 <span style="font-size:18px;font-weight:600;letter-spacing:0;color:#c9c9ce;">s</span></div>
    <div style="margin-top:8px;font-size:13px;line-height:1.45;color:#a4a4a9;">128K-token prompt, default configuration · base profile</div>
  </div>
  <div style="background:linear-gradient(160deg,#1d1d20 0%,#0e0e10 100%);border:1px solid rgba(255,255,255,0.09);border-radius:12px;padding:18px 20px 16px;box-shadow:2px 2px 0 #c8b1f1,inset 0 1px 0 rgba(255,255,255,0.05);">
    <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9a9a9f;">Context window</div>
    <div style="margin-top:10px;font-size:40px;line-height:1;font-weight:800;letter-spacing:-0.02em;color:#c8b1f1;">262,144</div>
    <div style="margin-top:8px;font-size:13px;line-height:1.45;color:#a4a4a9;">tokens per request · base profile</div>
  </div>
</div>

## RigMark

v2.0.1 on RigMark, against our own v1.8.4 as the baseline, with one column per v2.0.1 weight variant.

Appliance comparison: different model IDs, not a same-weights claim. v1.8.4 ran with reasoning off, its default; v2.0.1 ran at reasoning effort low. Cold prefill and replay rows use raw token-ID completions, where reasoning effort does not apply.

<div style="background:linear-gradient(160deg,#1d1d20 0%,#0e0e10 100%);border:1px solid rgba(255,255,255,0.09);border-radius:12px;padding:0;box-shadow:2px 2px 0 #f0a8a8,inset 0 1px 0 rgba(255,255,255,0.05);overflow:hidden;margin:14px 0 6px;">
<table style="width:100%;border-collapse:separate;border-spacing:0;margin:0;background:transparent;">
  <thead><tr>
    <th style="text-align:left;padding:11px 14px;font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9a9a9f;border-bottom:1px solid rgba(255,255,255,0.10);background:rgba(255,255,255,0.03);">RigMark row</th>
    <th style="text-align:center;padding:11px 14px;font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9a9a9f;border-bottom:1px solid rgba(255,255,255,0.10);background:rgba(255,255,255,0.03);">v1.8.4</th>
    <th style="text-align:center;padding:11px 14px;font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9a9a9f;border-bottom:1px solid rgba(255,255,255,0.10);background:rgba(255,255,255,0.03);">base weights + draft model · base profile</th>
    <th style="text-align:center;padding:11px 14px;font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9a9a9f;border-bottom:1px solid rgba(255,255,255,0.10);background:rgba(255,255,255,0.03);">refusal-removed (ablit) weights + draft model · ablit profile</th>
  </tr></thead>
  <tbody>
  <tr>
    <td style="text-align:left;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">Code, decode estimate (tok/s, higher is better)</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">61.1</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);color:#f0a8a8;font-weight:700;">91.3</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">90.1</td>
  </tr>
  <tr>
    <td style="text-align:left;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">Prose, decode estimate (tok/s, higher is better)</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">31.5</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);color:#f0a8a8;font-weight:700;">51.6</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">50.8</td>
  </tr>
  <tr>
    <td style="text-align:left;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">Structured output ceiling* (tok/s, higher is better)</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">95.4</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);color:#f0a8a8;font-weight:700;">127.9</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">127.4</td>
  </tr>
  <tr>
    <td style="text-align:left;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">Cold prefill, 64K prompt (tok/s, higher is better)</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">1,505</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);color:#f0a8a8;font-weight:700;">2,124</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">2,129</td>
  </tr>
  <tr>
    <td style="text-align:left;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">Four at once, end to end (tok/s, higher is better): short code, end-to-end, 256-token cap per agent</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">86.6</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);color:#f0a8a8;font-weight:700;">113.4</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">128.4</td>
  </tr>
  <tr>
    <td style="text-align:left;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">C1 per-stream time to first token, seconds (lower is better)</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">0.396</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);color:#f0a8a8;font-weight:700;">0.405</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">0.361</td>
  </tr>
  <tr>
    <td style="text-align:left;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">Prose time to first visible text, seconds (lower is better)</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">0.380</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);color:#f0a8a8;font-weight:700;">0.484</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">0.497</td>
  </tr>
  </tbody>
</table>
</div>

\* predictable-output ceiling; not a proxy for agent speed

*C1 per-stream time to first token:* With base weights, v1.8.4 is about 2% faster on this row. The first token is visible text in every reply in each column.

*Prose time to first visible text:* With base weights, v1.8.4 shows prose text about 0.10 s sooner; v2.0.1 takes about 1.27x as long. RigMark's own prose time to first token marks the first reasoning token, not visible text, so it is not shown.

Prose row: at reasoning effort low, v2.0.1 writes a short reasoning passage before the visible text (with base weights, 11 to 12 tokens, about 1% of each reply of about 1,000 tokens); v1.8.4, with reasoning off, wrote none. RigMark counts those tokens in the prose decode rate and in last output time.

Both raw RigMark blocks, one per weight variant, are in the [v2.0.1 release notes](https://github.com/jakejharris/jspark3/releases/tag/v2.0.1).

## Results

Three result sets from the same build: base weights with the draft model, refusal-removed weights with the draft model, and base weights without it. Each weight variant ships its own measured settings profile. Each column heading names the profile behind its figures.

<div style="background:linear-gradient(160deg,#1d1d20 0%,#0e0e10 100%);border:1px solid rgba(255,255,255,0.09);border-radius:12px;padding:0;box-shadow:2px 2px 0 #f0a8a8,inset 0 1px 0 rgba(255,255,255,0.05);overflow:hidden;margin:14px 0 6px;">
<table style="width:100%;border-collapse:separate;border-spacing:0;margin:0;background:transparent;">
  <thead><tr>
    <th style="text-align:left;padding:11px 14px;font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9a9a9f;border-bottom:1px solid rgba(255,255,255,0.10);background:rgba(255,255,255,0.03);">Measurement</th>
    <th style="text-align:center;padding:11px 14px;font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9a9a9f;border-bottom:1px solid rgba(255,255,255,0.10);background:rgba(255,255,255,0.03);">base weights + draft model · base profile (default)</th>
    <th style="text-align:center;padding:11px 14px;font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9a9a9f;border-bottom:1px solid rgba(255,255,255,0.10);background:rgba(255,255,255,0.03);">refusal-removed (ablit) weights + draft model · ablit profile</th>
    <th style="text-align:center;padding:11px 14px;font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9a9a9f;border-bottom:1px solid rgba(255,255,255,0.10);background:rgba(255,255,255,0.03);">base weights, no draft model (commercial use) · base profile</th>
  </tr></thead>
  <tbody>
  <tr>
    <td style="text-align:left;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">Decode per stream, short code replies <span style="color:#a4a4a9;">(tok/s)</span></td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);color:#f0a8a8;font-weight:700;">80.9</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">73.3</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">63.6</td>
  </tr>
  <tr>
    <td style="text-align:left;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">Decode per stream, short prose replies <span style="color:#a4a4a9;">(tok/s)</span></td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);color:#f0a8a8;font-weight:700;">59.6</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">63.6</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">57.7</td>
  </tr>
  <tr>
    <td style="text-align:left;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">Decode, one stream after a 32K-token prompt <span style="color:#a4a4a9;">(tok/s)</span></td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);color:#f0a8a8;font-weight:700;">83.6</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">72.3</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">not measured</td>
  </tr>
  <tr>
    <td style="text-align:left;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">Aggregate decode <span style="color:#a4a4a9;">(tok/s)</span>: short prompts, 41-62 tokens, 1 concurrent</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);color:#f0a8a8;font-weight:700;">59.8</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">69.8</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">59.0</td>
  </tr>
  <tr>
    <td style="text-align:left;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">Aggregate decode <span style="color:#a4a4a9;">(tok/s)</span>: short prompts, 41-62 tokens, 2 concurrent</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);color:#f0a8a8;font-weight:700;">77.6</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">75.8</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">not measured</td>
  </tr>
  <tr>
    <td style="text-align:left;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">Aggregate decode <span style="color:#a4a4a9;">(tok/s)</span>: short prompts, 41-62 tokens, 4 concurrent</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);color:#f0a8a8;font-weight:700;">97.5</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">96.4</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">not measured</td>
  </tr>
  <tr>
    <td style="text-align:left;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">Aggregate decode <span style="color:#a4a4a9;">(tok/s)</span>: short prompts, 41-62 tokens, 8 concurrent</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);color:#f0a8a8;font-weight:700;">126.0</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">121.7</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">81.2</td>
  </tr>
  <tr>
    <td style="text-align:left;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">Aggregate decode <span style="color:#a4a4a9;">(tok/s)</span>: short prompts, 41-62 tokens, 16 concurrent</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);color:#f0a8a8;font-weight:700;">106.8</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">110.0</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">not measured</td>
  </tr>
  <tr>
    <td style="text-align:left;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">Time to first token, p50 <span style="color:#a4a4a9;">(s)</span>: short prompts, 41-62 tokens, 8 concurrent</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);color:#f0a8a8;font-weight:700;">0.54 (first visible text 1.5; 3 of 24 showed no text)</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">0.54 (first visible text 1.2; 1 of 24 showed no text)</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">1.9 (visible text 0.41 s after the first token, median over the replies with text; 3 of 24 showed no text)</td>
  </tr>
  <tr>
    <td style="text-align:left;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">First visible text, p50 <span style="color:#a4a4a9;">(s)</span>: short prompts, 41-62 tokens, 8 concurrent</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);color:#f0a8a8;font-weight:700;">1.5</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">1.2</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">see the note below the table</td>
  </tr>
  <tr>
    <td style="text-align:left;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">Short replies that showed no text: short prompts, 41-62 tokens, 8 concurrent</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);color:#f0a8a8;font-weight:700;">3 of 24</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">1 of 24</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">3 of 24</td>
  </tr>
  <tr>
    <td style="text-align:left;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">Longest token gap, median <span style="color:#a4a4a9;">(s)</span>: short prompts, 41-62 tokens, 8 concurrent, while a prompt of about 8,000 or 36,000 tokens joins</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);color:#f0a8a8;font-weight:700;">0.52</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">0.49</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">not measured</td>
  </tr>
  <tr>
    <td style="text-align:left;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">Longest token gap, max <span style="color:#a4a4a9;">(s)</span>: short prompts, 41-62 tokens, 8 concurrent, while a prompt of about 8,000 or 36,000 tokens joins</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);color:#f0a8a8;font-weight:700;">0.75</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">0.57</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">not measured</td>
  </tr>
  <tr>
    <td style="text-align:left;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">Cold time to first token, 8K-token prompt <span style="color:#a4a4a9;">(s)</span></td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);color:#f0a8a8;font-weight:700;">3.8</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">3.8</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">3.9</td>
  </tr>
  <tr>
    <td style="text-align:left;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">Cold time to first token, 32K-token prompt <span style="color:#a4a4a9;">(s)</span></td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);color:#f0a8a8;font-weight:700;">15.0</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">15.0</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">15.8</td>
  </tr>
  <tr>
    <td style="text-align:left;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">Cold time to first token, 64K-token prompt <span style="color:#a4a4a9;">(s)</span></td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);color:#f0a8a8;font-weight:700;">30.4</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">30.4</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">not measured</td>
  </tr>
  <tr>
    <td style="text-align:left;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">Cold time to first token, 128K-token prompt <span style="color:#a4a4a9;">(s)</span></td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);color:#f0a8a8;font-weight:700;">63.6</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">63.8</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">not measured</td>
  </tr>
  <tr>
    <td style="text-align:left;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">Draft tokens accepted per verify step: short prompts, 41-62 tokens, 8 concurrent</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);color:#f0a8a8;font-weight:700;">2.75</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">2.55</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">not measured</td>
  </tr>
  <tr>
    <td style="text-align:left;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">Draft tokens accepted / proposed: short prompts, 41-62 tokens, 8 concurrent</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);color:#f0a8a8;font-weight:700;">0.67</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">0.69</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">not measured</td>
  </tr>
  <tr>
    <td style="text-align:left;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">Draft tokens accepted per verify step, one repeated prompt</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);color:#f0a8a8;font-weight:700;">2.11</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">2.04</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:1px solid rgba(255,255,255,0.06);">not measured</td>
  </tr>
  <tr>
    <td style="text-align:left;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:0;">Draft tokens accepted / proposed, one repeated prompt</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:0;color:#f0a8a8;font-weight:700;">0.61</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:0;">0.55</td>
    <td style="text-align:center;padding:10px 14px;font-size:14px;line-height:1.45;color:#e6e6ea;border-bottom:0;">not measured</td>
  </tr>
  </tbody>
</table>
</div>
<p style="font-size:12.5px;line-height:1.5;color:#c9c9ce;margin:6px 0 10px;"><em>Time to first token, p50, base weights + draft model · base profile:</em> First visible text at about 1.5 s (median); 3 of 24 short replies spent their 96-token limit on reasoning and showed no text. Measured at reasoning effort low.<br><em>Time to first token, p50, refusal-removed (ablit) weights + draft model · ablit profile:</em> First visible text at about 1.2 s (median); 1 of 24 short replies spent its 96-token limit on reasoning and showed no text. Measured at reasoning effort low.<br><em>Time to first token, p50, base weights, no draft model (commercial use) · base profile:</em> Visible text 0.41 s after the first token (median, p95 1.6 s) in the 21 replies that showed text; 3 of 24 short replies spent their 96-token limit on reasoning and showed no text. Measured at reasoning effort low.<br>The first-token notes use two measures: with the draft model, the median time to first visible text; without it, the median gap from first token to first visible text in the replies that showed text, because a median taken only over replies that showed text would come out below the first-token median taken over all replies.</p>

<p style="font-size:12.5px;line-height:1.5;color:#8b8b90;margin:0 0 18px;">Every set was measured on the same build, each on its weights variant's shipped settings, and no figure is a best run. Rates and times are medians, with the number of runs given below; the token gap is given as both a median and a maximum, and the context window is a setting, not a measurement. Short-reply decode is reported separately for code and for prose: the per-stream rate of replies capped at 256 tokens (median of 3 each). Long decode is one greedy code stream of up to 96 tokens after a 32K-token prompt (median of 3). Aggregate decode is the wall-clock rate of concurrent greedy replies, capped at 96 tokens, to a fixed mix of varied short prompts (41 to 62 tokens each) that is the same for every set (median of 3). Draft acceptance depends on the prompt, so it is also reported for a single repeated prompt. Cold time to first token uses exactly the stated number of prompt tokens with nothing cached (median of 2). The 8-request time to first token is the median wait for the first token when 8 short prompts from the mix (41 to 62 tokens) are sent at once; without the draft model, prompts that arrive together are not read together in one batch, and first tokens arrive later (see known issues). The 8-stream token gap is the longest pause seen by running streams while a prompt of about 8,000 or 36,000 tokens joins, reported as the median and the maximum of that pause across runs. Draft acceptance is the number of draft tokens accepted per verify step, not counting the token the model adds itself, with accepted over proposed tokens alongside, both measured at 8 concurrent requests. The no-draft-model set is a reduced run. The benchmark client runs on a separate machine on the same local network, so client-side times include one network hop. The benchmark figures in the result tables, including the stall bounds, come from chat requests at low reasoning effort. Installation, disk and startup figures are not chat measurements. Low effort still reasons before it answers, and decode and aggregate rates count reasoning tokens. A chat request that sets no reasoning effort runs at high effort, so its replies are longer and its rates can differ from these. Time to first token is measured to the first streamed token, reasoning or text. In the cold-prompt, newcomer and saved-session tests, that first token was visible text in all but two replies: one 64K cold-prompt reply with base weights and the saved-session return with refusal-removed (ablit) weights reached their eight-token limit on reasoning and showed no text. In the eight-client short-prompt test, most replies began with a short reasoning passage, so visible text arrives later than the first token, and some short replies spent their 96-token limit on reasoning and showed no text; each set's first-token figure is shown with its own visible-text note. Full record: <a href="https://github.com/jakejharris/jspark3/blob/v2.0.1/release/MEASUREMENTS-v2.0.1.md">measurements</a> and <a href="https://github.com/jakejharris/jspark3/blob/v2.0.1/release/results-v2.0.1.json">machine-readable results</a>.</p>

## Weights: two variants, one switch

The default is `base` (base weights). Choose the variant at install with `WEIGHTS=ablit` in `cluster.env`, or `--weights ablit` on fetch-weights.sh, split.sh and serve.sh (default `base`).

| Variant | What it is | Pinned source | License |
|---|---|---|---|
| `base` (default) | GLM-5.3 Flash in 4-bit MLX format, with the model's own multi-token prediction head | [`TensorFold/GLM-5.3-Flash-MLX-4bit-MTP`](https://huggingface.co/TensorFold/GLM-5.3-Flash-MLX-4bit-MTP/tree/76add2a341a1cd90ad0e86bb69839ea9c35827c6) at `76add2a341a1cd90ad0e86bb69839ea9c35827c6` | MIT |
| `ablit` | refusal-removed (abliterated) weights, opt-in | [`orcarouter/GLM-5.3-Flash-Uncensored-MLX`](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-MLX/tree/c02a5f6fa06f0aa444877b44d19fd5c96390329f) at `c02a5f6fa06f0aa444877b44d19fd5c96390329f` (gated) | MIT, plus the source card's use conditions |

**`base`.** The installer downloads them at the pinned revision, verifies every file against a pinned SHA-256 list, and splits them into three per-host parts, checked against a shipped manifest. This project hosts none of the v2.0.1 weights. Download size: 181,741,759,037 bytes (181.7 GB; 54 files) for the weights; with the 2,342,460,697-byte draft model the download is 184,084,219,734 bytes (184.1 GB).

**`ablit`: refusal-removed (abliterated) weights from [`orcarouter/GLM-5.3-Flash-Uncensored-MLX`](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-MLX).** This is an opt-in install for ablit development, red-teaming and refusal research. The installer downloads them with your own token at the pinned revision and converts them on your machine. Ready in this release: `scripts/fetch-weights.sh --weights ablit` downloads the source with your own token, `scripts/convert-ablit.sh` converts it and writes the three per-host parts, and `manifests/ablit/` checks a third you already have. We ran the shipped conversion on one DGX Spark, and its output matched these manifests file for file. On your own machine, `scripts/convert-ablit.sh` runs the pinned conversion scripts in `scripts/ablit/` inside the release's pinned container image, with no network and no GPU, then writes and checks the three per-host parts. The pinned base weights, TensorFold/GLM-5.3-Flash-MLX-4bit-MTP at revision 76add2a341a1cd90ad0e86bb69839ea9c35827c6 (MIT), supply the four-bit tensor layout and the native prediction layer that the conversion restores. The chat template is the base checkpoint's MIT template plus six lines added by this recipe, also under MIT; its reconstructed bytes are hash-checked. Measured on one DGX Spark: the 200.1 GB source download took about an hour on our connection. Converting, splitting and checking then took about 26 minutes of processing, used no GPU and under 4 GiB of process memory, and needed about 371 GB of free disk beyond the downloaded source (about 571 GB in all), on top of the base weights you already installed. `scripts/convert-ablit.sh` checks for about 400 GB free before it starts. To use it:

1. Sign in to a Hugging Face account.
2. Open [the source page](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-MLX) and accept its terms yourself.
3. Give the installer your own token in `HF_TOKEN`.

The source's model card sets these conditions, quoted verbatim:

> - It is released **strictly for legitimate research** — interpretability, AI-safety and refusal-mechanism study, red-teaming, robustness evaluation, and controlled experiments.
> - **You assume full responsibility and liability** for how you use it and for everything it generates. Do not deploy it to end users or in production without adding your own safety, moderation, and abuse-prevention layers.
>
> By downloading or using this model you acknowledge and accept the above.

You are responsible for complying with the source model's terms and for how you use this model and what it generates. JSpark3 provides conversion tooling only; it hosts none of these weights and does not endorse any use of them.

## Draft model and commercial use

The default draft model is Inco AI's [DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) at revision `bf582e4eacc1810f76656d1811693ff6c6737d2a`, licensed CC BY-NC-ND 4.0: non-commercial research and evaluation use only. The installer downloads it unmodified from Inco AI; this project never redistributes it. For commercial use of the draft model itself, contact Inco AI through its model card.

**Commercial path.** For commercial use, run the base weights without the draft model: start with `scripts/serve.sh --drafter none` on all three hosts (or set `DRAFTER=none` in `cluster.env`), and the model drafts with its own multi-token prediction head. That path runs MIT weights on a permissively licensed engine (MIT, with some Apache-2.0 code) and an Apache-2.0 recipe, inside NVIDIA's container under NVIDIA's terms. On the base weights, the draft model is the only non-commercial component.

For installation and the required `--drafter none` serving preflight, follow the maintained
[Running without the draft model](https://github.com/jakejharris/jspark3/blob/v2.0.2/INSTALL.md#running-without-the-draft-model) instructions.

## Install

The live image-cache acceptance passed on the prepared runtime. A clean installation of the final v2.0.2 public recipe has not been demonstrated; the installation and performance receipts remain v2.0.1 evidence. Follow the [v2.0.2 installation guide](https://github.com/jakejharris/jspark3/blob/v2.0.2/INSTALL.md) and [upgrade and rollback guide](https://github.com/jakejharris/jspark3/blob/v2.0.2/UPGRADING.md). All three ranks must run the same release.

**Historical v2.0.1 notes:** read [known issues and hotfixes](https://github.com/jakejharris/jspark3/blob/main/docs/TROUBLESHOOTING.md#v201-known-issues-and-hotfixes) for kernel rebuilds, possible compile locks, unavailable downloads and no-drafter preflight. The optional kernel rebuild hotfix is **validated on 3x DGX Spark (2026-10-04): retained boot reused the kernel cache; undo restored the original files**.

**Security.** The server listens on loopback (127.0.0.1) only, with no authentication and no CORS. Reach it through an SSH tunnel or a reverse proxy that adds authentication; don't expose the port.

**Images.** Both weight variants accept images in chat messages as inline `data:` URLs (base64). Remote image URLs are refused. Each request takes up to 16 images, at most 32 MB per image and 32 MB in total, and at most 32 megapixels per image. This release does not measure how well the model understands images.

Conversation state, including the prompt's token ids, is cached on each host's own disk (up to 64 GiB per host) so returning to a long conversation is fast. It never leaves your machines. The maintained [session-tier instructions](https://github.com/jakejharris/jspark3/blob/v2.0.2/INSTALL.md#session-tier) explain where it lives, how to clear it and how to turn it off (SESSION_TIER=off).

## Known issues

The v2.0.2 update adds two image fixes only. The live image-cache acceptance passed on the prepared runtime. A clean installation of the final v2.0.2 public recipe has not been demonstrated; the installation and performance receipts remain v2.0.1 evidence.

For installation and restart corrections, see the maintained [v2.0.1 known issues and hotfixes](https://github.com/jakejharris/jspark3/blob/main/docs/TROUBLESHOOTING.md#v201-known-issues-and-hotfixes). The API issue numbers below are unchanged.

1. `stop` is ignored. A reply ends at the model's end of turn or at `max_tokens`.
2. `response_format` is ignored. A request that sets it to `json_schema` or `json_object` (JSON mode) is accepted without an error, and neither JSON nor the schema is enforced, so the reply is free text. Forcing a tool call with `tool_choice` (`required` or a named function) works, but the call's arguments are not held to the tool's schema. This fix is outside v2.0.2; no target version is assigned.
3. Identical prompts without a `seed` return identical outputs, even above temperature 0: a chat app's regenerate returns the same reply, and two users who send the same prompt get the same answer. Send a different `seed` with each request when you want a different sample.
4. `n`, `logprobs`, presence and frequency penalties and `logit_bias` are ignored.
5. A wrongly typed field, such as a string `temperature`, may return HTTP 500 instead of 400.
6. Non-streaming requests send nothing until the reply is complete. Behind a proxy with an idle timeout, use `stream: true`.
7. The `model` field is not validated; every request is served by GLM-5.3 Flash.
8. Without the draft model, a long conversation that includes images may not be saved to the disk session cache, and each saved state takes more memory, so fewer long conversations stay cached. Returning to such a conversation after it has left the memory cache can take as long as its first prompt. Text-only conversations of about 40,000 tokens are saved; longer text-only conversations were not tested.
9. Reasoning is always on, and no setting turns it fully off; v1.8.4 had it off by default. A request that sets no reasoning effort runs at High, and the lowest effort is low; a top-level `reasoning_effort: "none"` and `chat_template_kwargs: {"enable_thinking": false}` are both treated as low, so a reply can still begin with a short reasoning passage. Even at low effort, a small `max_tokens` can be used up by reasoning and return no visible text; allow a few hundred tokens or more.
10. With the draft model on, a short text request that arrives while no reply is streaming may wait up to 25 ms for a second request before its prompt is read. Without the draft model, prompts that arrive together are not read together in one batch, and first tokens arrive later: with the base weights and 8 requests at once, the median first token arrives after 1.9 s, with visible text 0.41 s later (median over the replies that showed text), against 0.54 s with the draft model, where the median first visible text arrives at 1.5 s; in both sets, 3 of 24 short replies spent their 96-token limit on reasoning and showed no text. This fix is outside v2.0.2; no target version is assigned. With 16 requests at once, twice the server's 8 reply slots, total output with the draft model on is about 10 to 15% lower than with 8 (about 15% with the base weights, about 10% with the refusal-removed (ablit) weights), because the second eight prompts are read in small steps while the first eight replies stream. A streaming reply can occasionally pause between updates, and the pauses are longest while a long new prompt is being read: the longest measured pause was about 0.75 seconds, with a 36,180-token prompt. While a prompt of that length is being read, one step can pause every streaming reply at once for up to about 0.53 seconds.
11. JSpark3 v2.0.1 saves the state at the end of each prompt it reads, whichever client sent it, and reuses it when a later prompt starts with that entire earlier prompt, such as the next turn of a conversation; it then reads only the rest. Sharing only a system prompt is not enough: no state is saved where a system prompt ends, so a prompt with the same system prompt but a different first message is read in full. With the draft model on, it also skips reading a prompt that exactly repeats the latest prompt of a conversation, such as regenerating the latest reply, while that state is still in memory. With the draft model on, only the latest state of each conversation stays in memory, so regenerating or resending an earlier turn after later turns have been sent does not get this shortcut. Such a request resumes only from a shorter state that the disk session store has finished saving. The store saves in the background while the server is idle and may not yet hold a given turn, or may have skipped it; in testing, these regenerations read the whole prompt again. Without the draft model, an exact repeat is never skipped, but earlier turns' states can stay in memory until evicted, so regenerating a later turn can resume from the previous turn's state. Regenerating the first reply of a conversation after later turns, or resending it without the draft model, reads the whole prompt again, because saved state is reused only when it is shorter than the new prompt.
12. Conversations that share only a system prompt do not share cached work. Saved state is matched by prompt content, not by conversation: a prompt reuses an earlier prompt's state only when it starts with that entire earlier prompt, whichever conversation sent it. No state is saved at the end of a system prompt, so a new conversation that starts with the same system prompt as an earlier one, but has a different first message, reads its whole prompt again. This fix is outside v2.0.2; no target version is assigned.
13. A client that disconnects while its connection's socket number is 1024 or higher is not detected, so its generation runs to completion and holds its slot. Normal connection counts do not reach this; very many idle keep-alive clients could. This fix is outside v2.0.2; no target version is assigned.
14. If `max_tokens` cuts off a tool call, `finish_reason` is `length` (or `tool_calls` if an earlier call in the same reply was complete), the cut-off call is left out of the final `tool_calls`, and its raw text is returned in `content`. When streaming, its name and partial `arguments` (incomplete JSON) have already been sent. Raise `max_tokens` for tool use.
15. The `usage` block in replies does not include `prompt_tokens_details.cached_tokens`. The number of prompt tokens the server reused from saved state is reported in the reply's `tensorfold.cached` field instead (in the final chunk when streaming). For a request that forces a tool call, this count can be too high, even above the prompt's length. v1.8.4 returned this field, so a client that reads it must switch to `tensorfold.cached` when upgrading. This fix is outside v2.0.2; no target version is assigned.
16. v2.0.1 does not support per-request cache isolation. It ignores the `cache_salt` request field, and all clients of one server share its saved prompt state. A request whose prompt starts with another client's entire earlier prompt reuses that state, which shows in the reply's cached-token count and in a faster first token. v1.8.4's engine honored `cache_salt`, so a deployment that relied on it to keep clients apart is no longer isolated after upgrading. If clients must not learn about each other's prompts, give each one its own server with its own session folder. This fix is outside v2.0.2; no target version is assigned.
17. Saving a conversation to the disk session store is best-effort. The store saves in the background while the server is idle, so back-to-back requests from other long conversations can keep it from saving a conversation. A later return to that conversation, after it has left the memory cache, then reads its whole prompt again. In testing, returns sent after a pause of several seconds resumed from the disk store.

Image decode allocation failures, including `MemoryError`, can surface as HTTP 400. Pillow is not separately version-pinned. Boundary and full-prompt checkpoints share one optional save batch; backpressure can drop both together.

## Licenses

Recipe files are Apache-2.0. The engine (`engine/`) is MIT: a fork of TensorFold 0.3.6.2 (MIT). It includes third-party code under its own permissive licenses (MIT and Apache-2.0), listed in its THIRD_PARTY_NOTICES, and two lines ported from upstream TensorFold that stay Apache-2.0, credited in its NOTICE. The base weights are MIT. The refusal-removed weights are MIT plus their source card's use conditions. The chat template is Z.AI's MIT template plus six lines added by this recipe, also under MIT. The draft model is CC BY-NC-ND 4.0 (non-commercial); it is downloaded at install and never redistributed here. The NVIDIA container image is pulled from NGC under NVIDIA's terms.

This repository's metadata says `license: other` because it still hosts the v1.8.x EXL3 weights, which the ShapleyMcg License v1.0 in LICENSE covers. That license applies to those files only. v2.0.1 uses none of them.

## Upgrade and rollback

From v2.0.1, use a separate checkout and session namespace. All three ranks must run the same release. See the [v2.0.2 upgrade guide](https://github.com/jakejharris/jspark3/blob/v2.0.2/UPGRADING.md); keep v2.0.1 for rollback.

### Historical v2.0.1 upgrade notes

v2.0.1 is a new installation, not an in-place upgrade. The engine changes from vLLM to a fork of TensorFold 0.3.6.2 (MIT), and the weights change from the v1.8.x EXL3 files to public 4-bit MLX-format weights split across the three hosts. Stop v1.8.x before you start v2.0.1, and keep your v1.8.4 checkout and weights if you might roll back.

Reasoning is now always on. v1.8.4 had it off by default and honoured requests to turn it off; v2.0.1 runs a request with no reasoning setting at High and treats a request to turn it off as low (known issue 9). Replies begin with a reasoning passage, returned as `reasoning_content`, before the visible text, and it uses part of `max_tokens`.

v1.8.4 stays available; see rolling back. With base weights, two measured cases favour it (three DGX Sparks; v1.8.4 at its default, reasoning off, and v2.0.1 at reasoning effort low; an appliance comparison with different model IDs, not a same-weights claim). On prose replies, v1.8.4 shows the first visible text about 0.1 s sooner, because v2.0.1 writes a short reasoning passage first (known issue 9). With base weights and a single client on short code replies, the first visible text arrives in about the same time, with v1.8.4 about 2% faster. If you keep very many idle keep-alive clients connected, read known issue 13 first; that fix is outside v2.0.2; no target version is assigned.

To roll back, stop v2.0.1 and start v1.8.4 from its tag, following its own installation guide. v1.8.4 is the documented rollback.

In your v1.8.4 checkout:

```bash
git fetch --tags && git checkout v1.8.4
```

Or from a fresh clone:

```bash
git clone --branch v1.8.4 https://github.com/jakejharris/jspark3.git jspark3-v1.8.4
```

v1.8.4's installation guide downloads its weights from `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`, which is no longer public. If you kept your v1.8.4 weights, use them. Otherwise, fetch the same weight and config files, with the hashes v1.8.4 checks, from this repository at revision [`e6cb0b09`](https://huggingface.co/jakejharris/jspark3/tree/e6cb0b09b3bf9f2ce35721426c570c8e714c5fc4), which carries v1.8.4's own checksum list. In step 4 of the v1.8.4 guide, run this in place of the `hf download Mia-AiLab/...` command. Keep the directory name: v1.8.4's scripts expect it.

```sh
export HF_HUB_DISABLE_XET=1
hf download jakejharris/jspark3 \
  --revision e6cb0b09b3bf9f2ce35721426c570c8e714c5fc4 \
  --local-dir "$JSPARK_MODEL_ROOT/Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw-25a44fdb"
sha256sum "$JSPARK_MODEL_ROOT/Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw-25a44fdb/SHA256SUMS"
# expected: cb0da1f97a53aebc3fbc5478f19c82b25586b0bf8533c99fb4ed5321a48f5342
```

Use that revision, not `main`: later revisions changed the card and the checksum list. Then continue with the guide as written: its `validate_checkpoint.py` hashes every file against that list and must report `"serving_checkpoint_pass": true`. The v1.8.4 tag, recipe and guide stay unchanged.

See the maintained [upgrade and rollback guide](https://github.com/jakejharris/jspark3/blob/main/UPGRADING.md#going-back-to-v184) for the complete procedure.

## The EXL3 files in this repository

This repository also hosts an exact mirror of Brandon M. Music's
[ShapleyMcg](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) EXL3/TR3 4-bpw
quantization of GLM-5.3 Flash, as re-hosted by Mia-AiLab on Hugging Face. JSpark3 v1.8.x
served those files; v2.0.1 does not. They stay here, in the [`v1.8.x-exl3/`](https://huggingface.co/jakejharris/jspark3/tree/main/v1.8.x-exl3) folder, under the ShapleyMcg
License v1.0 in [`LICENSE`](LICENSE), for the [v1.8.4 rollback](https://github.com/jakejharris/jspark3/blob/main/UPGRADING.md#going-back-to-v184).

## Credits

v2.0.1 builds on work by:

- Z.AI (GLM-5.3 Flash, the base model, and its chat template)
- Hugging Face and the transformers contributors (the GLM-5.3 Flash model code the engine's CUDA path implements)
- Vontra, now TensorFold on Hugging Face (the 4-bit MLX base weights)
- Ash Hart and the TensorFold contributors (TensorFold 0.3.6.2, the engine release this project forks; DFlash ring-snapshot follow-up 47bf822)
- Taus Soe (GLM multi-stream foundation 20dbaba, disk-chain foundation b8a555a/bb16122 and CUDA image input 19680d9, via taussoe/TensorFold)
- FlyCockpit (zero-padding dimensions for three-way splitting, credited since v1.0.0)
- BTCXoomer (reporting the three-Spark NCCL subnet-routing requirement, credited in v1.1.0)
- Inco AI (the DFlash2 draft model)
- orcarouter (the refusal-removed source weights)
- z-lab (DFlash)
- MiaAI-Lab (upstream TensorFold: follower doorbell 358875c, DFlash ring 7c088eb, prefill row-blocking b23c10a and typed-parser hunk fe2b514)
- mikolaj92 (sparse-attention pool optimizations: skipping invisible pool tiles and bounding radix selection to visible pools, via upstream TensorFold commits b3b8a39 and f119334)
- Dorian (an upstream TensorFold server fix, ported)
- turboderp (ExLlamaV3's EXL3 format, which the engine's own decoders read)
- QTIP and QuIP# authors (trellis and incoherence-processing foundations used through ExLlamaV3's EXL3 format)
- Apple (MLX)
- Google DeepMind (Gemma 4, supported by the vendored engine's MLX backend)
- Prince Canuma and the mlx-vlm contributors (GLM-5.3 Flash code the engine follows and ports)

The [third-party notices](https://github.com/jakejharris/jspark3/blob/main/THIRD_PARTY_NOTICES.md) are the complete record: every third-party component of v2.0.1, its authors and its license (credits corrected 2026-10-03; the v2.0.1 tag's copy predates this).

v1.8.x credits are in the [v1.8.4 third-party notices](https://github.com/jakejharris/jspark3/blob/v1.8.4/THIRD_PARTY_NOTICES.md).

The [ShapleyMcg](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) attribution is reproduced below.

This work includes or was produced using ShapleyMcg, created by Brandon M. Music (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under the ShapleyMcg License v1.0, an attribution-required license that grants no rights to the person known as "0xSero." Use of ShapleyMcg without this attribution is unlicensed.

Schedule B citation:

```bibtex
@misc{music2026shapleymcg,
  author = {Music, Brandon M.},
  title  = {ShapleyMCG: An Auditable Calibration-to-Encoding Pipeline for
            Low-Bit Mixture-of-Experts Models},
  year   = {2026},
  url    = {https://github.com/brandonmmusic-max/shapleymcg},
  note   = {Licensed under the ShapleyMcg License v1.0}
}
```
