# JSpark3 v2.0.2 public release facts

Current copy and tag pointers, with historical v2.0.1 result literals preserved.
The release-owner approved publication with the clean-install limitation stated below.

```json
{
  "status": "final",
  "project": "JSpark3",
  "model": "GLM-5.3 Flash",
  "version": "v2.0.2",
  "published": "2026-10-05",
  "version_label_public": "v2.0.2",
  "history": [
    {
      "version": "v2.0.0",
      "note": "internal build, not published"
    }
  ],
  "tag": "v2.0.2",
  "install_tag": "v2.0.2",
  "results_tag": "v2.0.1",
  "engine": {
    "name": "TensorFold 0.3.6.2 (fork)",
    "upstream": "https://github.com/ashhart/TensorFold",
    "base_version": "TensorFold 0.3.6.2 (MIT, before upstream relicensed to Apache-2.0 at 0.6.0)",
    "license": "MIT, plus one Apache-2.0 file (src/tensorfold/cuda/server.py, two ported lines; NOTICE)",
    "previous_engine": "vLLM (JSpark3 v1.8.x)",
    "provenance_line_public": "a fork of TensorFold 0.3.6.2 (MIT)"
  },
  "weights": {
    "variants": {
      "base": {
        "source": "TensorFold/GLM-5.3-Flash-MLX-4bit-MTP@76add2a341a1cd90ad0e86bb69839ea9c35827c6",
        "redirects_to": "TensorFold/GLM-5.3-Flash-MLX-4bit-MTP",
        "license": "MIT",
        "download_bytes": "181,741,759,037 bytes (181.7 GB; 54 files) for the weights; with the 2,342,460,697-byte draft model the download is 184,084,219,734 bytes (184.1 GB)",
        "stranger_split_matches": "A fresh download and split on a DGX Spark that had never run this project matched the published per-host manifests file for file.",
        "label": "base weights",
        "description": "GLM-5.3 Flash in 4-bit MLX format, with the model's own multi-token prediction head",
        "how": "The installer downloads them at the pinned revision, verifies every file against a pinned SHA-256 list, and splits them into three per-host parts, checked against a shipped manifest. This project hosts none of the v2.0.1 weights."
      },
      "ablit": {
        "description": "refusal-removed (abliterated) weights",
        "source": "orcarouter/GLM-5.3-Flash-Uncensored-MLX@c02a5f6fa06f0aa444877b44d19fd5c96390329f",
        "source_access": "gated (auto-approve on the repo page): Hugging Face account, user accepts the source's terms themselves, user's own HF_TOKEN",
        "other_inputs": "The pinned base weights, TensorFold/GLM-5.3-Flash-MLX-4bit-MTP at revision 76add2a341a1cd90ad0e86bb69839ea9c35827c6 (MIT), supply the four-bit tensor layout and the native prediction layer that the conversion restores. The chat template is the base checkpoint's MIT template plus six lines added by this recipe, also under MIT; its reconstructed bytes are hash-checked.",
        "license": "MIT (Copyright (c) 2026 Z.AI Co., Ltd), plus the use conditions on the source model card, quoted below.",
        "ablit_source_terms_verbatim": "- It is released **strictly for legitimate research** \u2014 interpretability, AI-safety and refusal-mechanism study, red-teaming, robustness evaluation, and controlled experiments.\n- **You assume full responsibility and liability** for how you use it and for everything it generates. Do not deploy it to end users or in production without adding your own safety, moderation, and abuse-prevention layers.\n\nBy downloading or using this model you acknowledge and accept the above.",
        "conversion": "On your own machine, `scripts/convert-ablit.sh` runs the pinned conversion scripts in `scripts/ablit/` inside the release's pinned container image, with no network and no GPU, then writes and checks the three per-host parts.",
        "conversion_cost_one_spark": "Measured on one DGX Spark: the 200.1 GB source download took about an hour on our connection. Converting, splitting and checking then took about 26 minutes of processing, used no GPU and under 4 GiB of process memory, and needed about 371 GB of free disk beyond the downloaded source (about 571 GB in all), on top of the base weights you already installed. `scripts/convert-ablit.sh` checks for about 400 GB free before it starts.",
        "user_responsibility_line": "You are responsible for complying with the source model's terms and for how you use this model and what it generates. JSpark3 provides conversion tooling only; it hosts none of these weights and does not endorse any use of them.",
        "label": "refusal-removed (abliterated) weights",
        "how": "The installer downloads them with your own token at the pinned revision and converts them on your machine.",
        "framing_public": "an opt-in install for ablit development, red-teaming and refusal research",
        "measured_vs_fresh_public": "A fresh conversion matches the measured data file for file, except two bookkeeping files on each host that record which conversion produced them.",
        "status_public": "Ready in this release: `scripts/fetch-weights.sh --weights ablit` downloads the source with your own token, `scripts/convert-ablit.sh` converts it and writes the three per-host parts, and `manifests/ablit/` checks a third you already have. We ran the shipped conversion on one DGX Spark, and its output matched these manifests file for file."
      }
    },
    "default_variant": "base"
  },
  "draft_model": {
    "name": "incoai/GLM-5.3-Flash-DFlash2@bf582e4eacc1810f76656d1811693ff6c6737d2a",
    "license": "CC BY-NC-ND 4.0 (non-commercial)",
    "distribution": "downloaded unmodified at install; never redistributed; requantized in memory at load only",
    "commercial_alternative": "`scripts/serve.sh --drafter none` (the weights' own multi-token prediction head)",
    "commercial_contact": "For commercial use of the draft model itself, contact Inco AI through its model card."
  },
  "headline": {
    "sentences": [
      "JSpark3 v2.0.2 fixes image-history checkpoint reuse and adds GIF frame-zero support.",
      "Live 8/9/10-image resumes were 0 -> 68 -> 174 tokens, both nonzero hits from disk; GIF passed and smoke passed 6/6."
    ],
    "cites": [],
    "rule": "Small CUDA acceptance sequence with idle persistence opportunities; no new performance benchmark."
  },
  "result_sets": {
    "V-D": {
      "label_public": "base weights + draft model",
      "metrics": {
        "decode_short_tok_s": {
          "code": "80.9",
          "prose": "59.6"
        },
        "decode_long_tok_s": "83.6",
        "cold_ttft_s": {
          "8k": "3.8",
          "32k": "15.0",
          "64k": "30.4",
          "128k": "63.6"
        },
        "concurrency_aggregate_tok_s": {
          "c1": "59.8",
          "c2": "77.6",
          "c4": "97.5",
          "c8": "126.0",
          "c16": "106.8"
        },
        "c8_ttft_p50_s": "0.54",
        "c8_visible_text_p50_s": "1.5",
        "c8_no_text_replies": "3 of 24",
        "c8_ttft_partner_public": "First visible text at about 1.5 s (median); 3 of 24 short replies spent their 96-token limit on reasoning and showed no text. Measured at reasoning effort low.",
        "c8_stall_s": {
          "median": "0.52",
          "max": "0.75"
        },
        "draft_acceptance": {
          "mix": {
            "accepted_per_verify_step": "2.75",
            "accepted_over_proposed": "0.67"
          },
          "single_prompt": {
            "accepted_per_verify_step": "2.11",
            "accepted_over_proposed": "0.61"
          }
        },
        "max_context_tokens": "262,144"
      }
    },
    "O-D": {
      "label_public": "refusal-removed (ablit) weights + draft model",
      "metrics": {
        "decode_short_tok_s": {
          "code": "73.3",
          "prose": "63.6"
        },
        "decode_long_tok_s": "72.3",
        "cold_ttft_s": {
          "8k": "3.8",
          "32k": "15.0",
          "64k": "30.4",
          "128k": "63.8"
        },
        "concurrency_aggregate_tok_s": {
          "c1": "69.8",
          "c2": "75.8",
          "c4": "96.4",
          "c8": "121.7",
          "c16": "110.0"
        },
        "c8_ttft_p50_s": "0.54",
        "c8_visible_text_p50_s": "1.2",
        "c8_no_text_replies": "1 of 24",
        "c8_ttft_partner_public": "First visible text at about 1.2 s (median); 1 of 24 short replies spent its 96-token limit on reasoning and showed no text. Measured at reasoning effort low.",
        "c8_stall_s": {
          "median": "0.49",
          "max": "0.57"
        },
        "draft_acceptance": {
          "mix": {
            "accepted_per_verify_step": "2.55",
            "accepted_over_proposed": "0.69"
          },
          "single_prompt": {
            "accepted_per_verify_step": "2.04",
            "accepted_over_proposed": "0.55"
          }
        },
        "max_context_tokens": "262,144"
      }
    },
    "V-N": {
      "label_public": "base weights, no draft model (commercial use)",
      "metrics": {
        "decode_short_tok_s": {
          "code": "63.6",
          "prose": "57.7"
        },
        "cold_ttft_s": {
          "8k": "3.9",
          "32k": "15.8"
        },
        "concurrency_aggregate_tok_s": {
          "c1": "59.0",
          "c8": "81.2"
        },
        "c8_ttft_p50_s": "1.9",
        "c8_no_text_replies": "3 of 24",
        "c8_ttft_partner_public": "Visible text 0.41 s after the first token (median, p95 1.6 s) in the 21 replies that showed text; 3 of 24 short replies spent their 96-token limit on reasoning and showed no text. Measured at reasoning effort low.",
        "c8_visible_gap_p50_s": "0.41",
        "c8_visible_gap_p95_s": "1.6"
      }
    }
  },
  "metrics_template": {
    "decode_short_tok_s": {
      "code": true,
      "prose": true
    },
    "decode_long_tok_s": true,
    "cold_ttft_s": {
      "8k": true,
      "32k": true,
      "64k": true,
      "128k": true
    },
    "concurrency_aggregate_tok_s": {
      "c1": true,
      "c2": true,
      "c4": true,
      "c8": true,
      "c16": true
    },
    "c8_ttft_p50_s": true,
    "c8_visible_text_p50_s": true,
    "c8_no_text_replies": true,
    "c8_ttft_partner_public": true,
    "c8_stall_s": {
      "median": true,
      "max": true
    },
    "draft_acceptance": {
      "mix": {
        "accepted_per_verify_step": true,
        "accepted_over_proposed": true
      },
      "single_prompt": {
        "accepted_per_verify_step": true,
        "accepted_over_proposed": true
      }
    },
    "max_context_tokens": "262,144",
    "concurrency_label_template": "short prompts, 41-62 tokens, {N} concurrent",
    "stall_condition_template": "short prompts, 41-62 tokens, {N} concurrent, while a prompt of about 8,000 or 36,000 tokens joins",
    "c8_partner_estimators_public": "The first-token notes use two measures: with the draft model, the median time to first visible text; without it, the median gap from first token to first visible text in the replies that showed text, because a median taken only over replies that showed text would come out below the first-token median taken over all replies.",
    "prompt_mix": {
      "public": "The 16 prompts were written for this benchmark and contain no private data. Their full texts ship with the results as PROMPT-MIX.jsonl."
    },
    "_first_token_partners": [
      {
        "figure": "result_sets.V-D.metrics.c8_ttft_p50_s",
        "partner_public": "result_sets.V-D.metrics.c8_ttft_partner_public",
        "partner_keys": [
          "result_sets.V-D.metrics.c8_visible_text_p50_s",
          "result_sets.V-D.metrics.c8_no_text_replies",
          "result_sets.V-D.metrics.c8_visible_gap_p50_s"
        ]
      }
    ]
  },
  "rigmark": {
    "v1_8_4_block": "MODEL      glm-5.3-flash\nAPPLIANCE  3x NVIDIA DGX Spark (GB10), 128 GB unified memory each\nRUN        reasoning=low  \u2022  protocol=1.1.0\nSOURCE     git:c5a0db01b054  \u2022  clean\nWORKLOAD       DECODE EST.      LAST OUTPUT          RANGE          BASIC GATE\nCODE              61.1 tok/s      37.6s last     58.2\u201364.7     \u2713 5/5\nPROSE             31.5 tok/s      30.7s last     31.1\u201332.3     \u2713 5/5\nSTRUCTURED*       95.4 tok/s       5.0s last     94.0\u201395.8     \u2713 5/5\n* predictable-output ceiling; not a proxy for agent speed\n64K PREFILL   cold 1,505 tok/s  \u2022  immediate replay 132,921 tok/s\nAGGREGATE   C1 42.1  \u2022  C2 61.9  \u2022  C4 86.6 tok/s   (short code, end-to-end, 256-token cap per agent)\nC4 OUTPUT STATE   normal stop 0/12  \u2022  visible 12/12  \u2022  reasoning may be included\nJSON       sha256:a3a5adf1b11a5a3e\u2026",
    "V-D": "MODEL      glm53\nAPPLIANCE  3x NVIDIA DGX Spark (GB10), 128 GB unified memory each\nRUN        reasoning=low  \u2022  protocol=1.1.0\nSOURCE     git:c5a0db01b054  \u2022  clean\nWORKLOAD       DECODE EST.      LAST OUTPUT          RANGE          BASIC GATE\nCODE              91.3 tok/s      21.3s last     88.2\u201392.8     \u2713 5/5\nPROSE             51.6 tok/s      19.3s last     50.3\u201353.1     \u2713 5/5\nSTRUCTURED*      127.9 tok/s       3.8s last    127.4\u2013128.3    \u2713 5/5\n* predictable-output ceiling; not a proxy for agent speed\n64K PREFILL   cold 2,124 tok/s  \u2022  immediate replay 821,037 tok/s\nAGGREGATE   C1 64.8  \u2022  C2 84.6  \u2022  C4 113.4 tok/s   (short code, end-to-end, 256-token cap per agent)\nC4 OUTPUT STATE   normal stop 0/12  \u2022  visible 12/12  \u2022  reasoning may be included\nJSON       sha256:05051f88e880384c\u2026",
    "O-D": "MODEL      glm53\nAPPLIANCE  3x NVIDIA DGX Spark (GB10), 128 GB unified memory each\nRUN        reasoning=low  \u2022  protocol=1.1.0\nSOURCE     git:c5a0db01b054  \u2022  clean\nWORKLOAD       DECODE EST.      LAST OUTPUT          RANGE          BASIC GATE\nCODE              90.1 tok/s      23.8s last     88.4\u201392.0     \u2713 5/5\nPROSE             50.8 tok/s      19.3s last     50.6\u201351.6     \u2713 5/5\nSTRUCTURED*      127.4 tok/s       3.8s last    121.0\u2013127.9    \u2713 5/5\n* predictable-output ceiling; not a proxy for agent speed\n64K PREFILL   cold 2,129 tok/s  \u2022  immediate replay 812,054 tok/s\nAGGREGATE   C1 67.9  \u2022  C2 93.8  \u2022  C4 128.4 tok/s   (short code, end-to-end, 256-token cap per agent)\nC4 OUTPUT STATE   normal stop 0/12  \u2022  visible 12/12  \u2022  reasoning may be included\nJSON       sha256:1b42ea87215719ef\u2026",
    "publish": true,
    "comparison_public": {
      "scope": "Appliance comparison: different model IDs, not a same-weights claim. v1.8.4 ran with reasoning off, its default; v2.0.1 ran at reasoning effort low. Cold prefill and replay rows use raw token-ID completions, where reasoning effort does not apply.",
      "c1_ttft_row": {
        "label": "C1 per-stream time to first token, seconds (lower is better)",
        "v1_8_4": "0.396",
        "V-D": "0.405",
        "O-D": "0.361",
        "note": "With base weights, v1.8.4 is about 2% faster on this row. The first token is visible text in every reply in each column."
      },
      "prose_footnote": "Prose row: at reasoning effort low, v2.0.1 writes a short reasoning passage before the visible text (with base weights, 11 to 12 tokens, about 1% of each reply of about 1,000 tokens); v1.8.4, with reasoning off, wrote none. RigMark counts those tokens in the prose decode rate and in last output time.",
      "prose_visible_ttft_row": {
        "label": "Prose time to first visible text, seconds (lower is better)",
        "v1_8_4": "0.380",
        "V-D": "0.484",
        "O-D": "0.497",
        "note": "With base weights, v1.8.4 shows prose text about 0.10 s sooner; v2.0.1 takes about 1.27x as long. RigMark's own prose time to first token marks the first reasoning token, not visible text, so it is not shown."
      }
    }
  },
  "rigmark_rows": [
    {
      "id": "code",
      "label": "Code, decode estimate",
      "unit": "tok/s",
      "better": "higher",
      "v1_8_4": "61.1",
      "V-D": "91.3",
      "O-D": "90.1"
    },
    {
      "id": "prose",
      "label": "Prose, decode estimate",
      "unit": "tok/s",
      "better": "higher",
      "v1_8_4": "31.5",
      "V-D": "51.6",
      "O-D": "50.8"
    },
    {
      "id": "structured",
      "label": "Structured output ceiling",
      "unit": "tok/s",
      "better": "higher",
      "v1_8_4": "95.4",
      "V-D": "127.9",
      "O-D": "127.4"
    },
    {
      "id": "prefill_64k",
      "label": "Cold prefill, 64K prompt",
      "unit": "tok/s",
      "better": "higher",
      "v1_8_4": "1,505",
      "V-D": "2,124",
      "O-D": "2,129"
    },
    {
      "id": "c4",
      "label": "Four at once, end to end",
      "unit": "tok/s",
      "better": "higher",
      "v1_8_4": "86.6",
      "V-D": "113.4",
      "O-D": "128.4"
    }
  ],
  "runnability": {
    "level_reached": "live-acceptance"
  },
  "compatibility": {
    "openai_chat_completions": "Works: `/v1/chat/completions` returns the reply with `finish_reason` `stop`.",
    "streaming": "Works: `stream: true` returns server-sent events ending in `[DONE]`, with the same text as the non-streaming reply.",
    "tool_calls": "Works: a request with `tools` returns a `tool_calls` reply with JSON arguments and `finish_reason` `tool_calls`.",
    "response_format_json_schema": "Ignored: `json_schema` and `json_object` requests are accepted without an error and not enforced (known issue 2). Forced `tool_choice` works. A clear HTTP 400 error is outside v2.0.2; no target version is assigned.",
    "thinking_on_off": "Always on; the lowest reasoning effort is low. v1.8.4 had it off by default.",
    "bind": "loopback only, no auth, no CORS"
  },
  "license": {
    "components": {
      "recipe": "Apache-2.0",
      "engine": "MIT (fork of TensorFold 0.3.6.2, MIT), with inherited third-party code under its own permissive licenses (MIT and Apache-2.0; engine THIRD_PARTY_NOTICES) and two Apache-2.0 lines ported from upstream TensorFold into cuda/server.py (engine NOTICE)",
      "base_weights": "MIT",
      "draft_model": "CC BY-NC-ND 4.0",
      "ablit_weights": "MIT + source card use conditions; gated; fetched with the user's token, converted locally, not redistributed",
      "container_image": "NVIDIA terms (pulled by digest, not redistributed)",
      "fabric_and_chat_template": "No unlicensed inputs: the fabric launcher is this project's own recipe script (Apache-2.0); the chat template is the base model's stock MIT template (Z.AI notice kept) plus six lines added by this recipe, also under MIT, shipped in the recipe."
    },
    "line": "Recipe files are Apache-2.0. The engine (`engine/`) is MIT: a fork of TensorFold 0.3.6.2 (MIT). It includes third-party code under its own permissive licenses (MIT and Apache-2.0), listed in its THIRD_PARTY_NOTICES, and two lines ported from upstream TensorFold that stay Apache-2.0, credited in its NOTICE. The base weights are MIT. The refusal-removed weights are MIT plus their source card's use conditions. The chat template is Z.AI's MIT template plus six lines added by this recipe, also under MIT. The draft model is CC BY-NC-ND 4.0 (non-commercial); it is downloaded at install and never redistributed here. The NVIDIA container image is pulled from NGC under NVIDIA's terms.",
    "commercial_path": "For commercial use, run the base weights without the draft model: start with `scripts/serve.sh --drafter none` on all three hosts (or set `DRAFTER=none` in `cluster.env`), and the model drafts with its own multi-token prediction head. That path runs MIT weights on a permissively licensed engine (MIT, with some Apache-2.0 code) and an Apache-2.0 recipe, inside NVIDIA's container under NVIDIA's terms. On the base weights, the draft model is the only non-commercial component."
  },
  "links": {
    "release": "https://github.com/jakejharris/jspark3/releases/tag/v2.0.2",
    "install": "https://github.com/jakejharris/jspark3/blob/v2.0.2/INSTALL.md",
    "results": "https://github.com/jakejharris/jspark3/blob/v2.0.1/release/results-v2.0.1.json",
    "site_hub": "https://jakejh.com/jspark3/",
    "site_model_page": "https://jakejh.com/jspark3/glm/",
    "repo": "https://github.com/jakejharris/jspark3",
    "hf_repo": "https://huggingface.co/jakejharris/jspark3",
    "source": "https://github.com/jakejharris/jspark3/tree/v2.0.2",
    "measurements": "https://github.com/jakejharris/jspark3/blob/v2.0.1/release/MEASUREMENTS-v2.0.1.md",
    "limitations": "https://github.com/jakejharris/jspark3/blob/v2.0.2/LIMITATIONS.md",
    "licensing": "https://github.com/jakejharris/jspark3/blob/v2.0.2/NOTICE",
    "notices": "https://github.com/jakejharris/jspark3/blob/v2.0.2/THIRD_PARTY_NOTICES.md",
    "upgrade": "https://github.com/jakejharris/jspark3/blob/v2.0.2/UPGRADING.md",
    "evidence": "https://github.com/jakejharris/jspark3/blob/v2.0.2/release/v2.0.2/ROOTCAUSE.md"
  },
  "upgrade_from_v1_8": "v2.0.2 is a new installation, not an in-place upgrade. The engine changes from vLLM to a fork of TensorFold 0.3.6.2 (MIT), and the weights change from the v1.8.x EXL3 files to public 4-bit MLX-format weights split across the three hosts. Stop v1.8.x before you start v2.0.2, and keep your v1.8.4 checkout and weights if you might roll back.",
  "supersedes": "v2.0.1",
  "release": {
    "title": "JSpark3 v2.0.2"
  },
  "hardware": {
    "summary": "three NVIDIA DGX Sparks",
    "link": "connected by a direct high-speed (RDMA) link",
    "network": "Cable the boxes' ConnectX-7 ports in a ring (rank 0 to rank 1, rank 1 to rank 2, rank 2 to rank 0), with each port up, RDMA working and MTU 9000; tensors travel over these cables. Every box also needs a shared LAN on which it can reach rank 0. The engine uses that LAN only to coordinate startup; tensor traffic stays on the cables.",
    "disk_per_host": {
      "image": "about 25 GB",
      "wheels_and_engine_build": "under 50 MB",
      "download": "181,741,759,037 bytes (181.7 GB; 54 files) for the weights; with the 2,342,460,697-byte draft model the download is 184,084,219,734 bytes (184.1 GB)",
      "thirds": "63.9 GB for the first third, 62.8 GB for each of the other two",
      "kernel_cache": "about 45 MB per host",
      "session_tier": "up to 64 GiB, written only while 150 GiB stays free",
      "after_split": "the full download ($DATA/base/weights) is not needed to serve, so you may delete it."
    }
  },
  "previous_release": {
    "tag": "v1.8.4",
    "engine": "vLLM",
    "url": "https://github.com/jakejharris/jspark3/releases/tag/v1.8.4",
    "install": "https://github.com/jakejharris/jspark3/blob/v1.8.4/docs/INSTALL.md",
    "notices": "https://github.com/jakejharris/jspark3/blob/v1.8.4/THIRD_PARTY_NOTICES.md",
    "disk_per_host_public": "roughly 164 GiB of weights, 2.34 GB of draft weights and a 21 GB image per Spark, plus build layers and caches (v1.8.4 INSTALL)"
  },
  "related": {
    "tempo": {
      "name": "JSpark3 Tempo",
      "model": "DeepSeek-V4.1 Flash",
      "repo": "https://github.com/jakejharris/jspark3-deepseek"
    }
  },
  "what_changed": "JSpark3 v2.0.2 fixes image-history checkpoint reuse and adds GIF frame-zero support. Boundary and full-prompt checkpoints were durable on all three ranks. Optional disk saves can still be skipped under continuous traffic.",
  "install_claim": "The live image-cache acceptance passed on the prepared runtime. A clean installation of the final v2.0.2 public recipe has not been demonstrated; the installation and performance receipts remain v2.0.1 evidence.",
  "measurement_conditions": "Historical v2.0.1 measurements; v2.0.2 has no new performance results. Every set was measured on the same build, each on its weights variant's shipped settings, and no figure is a best run. Rates and times are medians, with the number of runs given below; the token gap is given as both a median and a maximum, and the context window is a setting, not a measurement. Short-reply decode is reported separately for code and for prose: the per-stream rate of replies capped at 256 tokens (median of 3 each). Long decode is one greedy code stream of up to 96 tokens after a 32K-token prompt (median of 3). Aggregate decode is the wall-clock rate of concurrent greedy replies, capped at 96 tokens, to a fixed mix of varied short prompts (41 to 62 tokens each) that is the same for every set (median of 3). Draft acceptance depends on the prompt, so it is also reported for a single repeated prompt. Cold time to first token uses exactly the stated number of prompt tokens with nothing cached (median of 2). The 8-request time to first token is the median wait for the first token when 8 short prompts from the mix (41 to 62 tokens) are sent at once; without the draft model, prompts that arrive together are not read together in one batch, and first tokens arrive later (see known issues). The 8-stream token gap is the longest pause seen by running streams while a prompt of about 8,000 or 36,000 tokens joins, reported as the median and the maximum of that pause across runs. Draft acceptance is the number of draft tokens accepted per verify step, not counting the token the model adds itself, with accepted over proposed tokens alongside, both measured at 8 concurrent requests. The no-draft-model set is a reduced run. The benchmark client runs on a separate machine on the same local network, so client-side times include one network hop. The benchmark figures in the result tables, including the stall bounds, come from chat requests at low reasoning effort. Installation, disk and startup figures are not chat measurements. Low effort still reasons before it answers, and decode and aggregate rates count reasoning tokens. A chat request that sets no reasoning effort runs at high effort, so its replies are longer and its rates can differ from these. Time to first token is measured to the first streamed token, reasoning or text. In the cold-prompt, newcomer and saved-session tests, that first token was visible text in all but two replies: one 64K cold-prompt reply with base weights and the saved-session return with refusal-removed (ablit) weights reached their eight-token limit on reasoning and showed no text. In the eight-client short-prompt test, most replies began with a short reasoning passage, so visible text arrives later than the first token, and some short replies spent their 96-token limit on reasoning and showed no text; each set's first-token figure is shown with its own visible-text note.",
  "switch": {
    "weights": "`WEIGHTS=ablit` in `cluster.env`, or `--weights ablit` on fetch-weights.sh, split.sh and serve.sh (default `base`)",
    "drafter": "`scripts/serve.sh --drafter none` on all three hosts, or `DRAFTER=none` in `cluster.env` (default `dflash2`)"
  },
  "known_issues_public": [
    "`stop` is ignored. A reply ends at the model's end of turn or at `max_tokens`.",
    "`response_format` is ignored. A request that sets it to `json_schema` or `json_object` (JSON mode) is accepted without an error, and neither JSON nor the schema is enforced, so the reply is free text. Forcing a tool call with `tool_choice` (`required` or a named function) works, but the call's arguments are not held to the tool's schema. This fix is outside v2.0.2; no target version is assigned.",
    "Identical prompts without a `seed` return identical outputs, even above temperature 0: a chat app's regenerate returns the same reply, and two users who send the same prompt get the same answer. Send a different `seed` with each request when you want a different sample.",
    "`n`, `logprobs`, presence and frequency penalties and `logit_bias` are ignored.",
    "A wrongly typed field, such as a string `temperature`, may return HTTP 500 instead of 400.",
    "Non-streaming requests send nothing until the reply is complete. Behind a proxy with an idle timeout, use `stream: true`.",
    "The `model` field is not validated; every request is served by GLM-5.3 Flash.",
    "Without the draft model, a long conversation that includes images may not be saved to the disk session cache, and each saved state takes more memory, so fewer long conversations stay cached. Returning to such a conversation after it has left the memory cache can take as long as its first prompt. Text-only conversations of about 40,000 tokens are saved; longer text-only conversations were not tested.",
    "Reasoning is always on, and no setting turns it fully off; v1.8.4 had it off by default. A request that sets no reasoning effort runs at High, and the lowest effort is low; a top-level `reasoning_effort: \"none\"` and `chat_template_kwargs: {\"enable_thinking\": false}` are both treated as low, so a reply can still begin with a short reasoning passage. Even at low effort, a small `max_tokens` can be used up by reasoning and return no visible text; allow a few hundred tokens or more.",
    "With the draft model on, a short text request that arrives while no reply is streaming may wait up to 25 ms for a second request before its prompt is read. Without the draft model, prompts that arrive together are not read together in one batch, and first tokens arrive later: with the base weights and 8 requests at once, the median first token arrives after 1.9 s, with visible text 0.41 s later (median over the replies that showed text), against 0.54 s with the draft model, where the median first visible text arrives at 1.5 s; in both sets, 3 of 24 short replies spent their 96-token limit on reasoning and showed no text. This fix is outside v2.0.2; no target version is assigned. With 16 requests at once, twice the server's 8 reply slots, total output with the draft model on is about 10 to 15% lower than with 8 (about 15% with the base weights, about 10% with the refusal-removed (ablit) weights), because the second eight prompts are read in small steps while the first eight replies stream. A streaming reply can occasionally pause between updates, and the pauses are longest while a long new prompt is being read: the longest measured pause was about 0.75 seconds, with a 36,180-token prompt. While a prompt of that length is being read, one step can pause every streaming reply at once for up to about 0.53 seconds.",
    "JSpark3 v2.0.1 saves the state at the end of each prompt it reads, whichever client sent it, and reuses it when a later prompt starts with that entire earlier prompt, such as the next turn of a conversation; it then reads only the rest. Sharing only a system prompt is not enough: no state is saved where a system prompt ends, so a prompt with the same system prompt but a different first message is read in full. With the draft model on, it also skips reading a prompt that exactly repeats the latest prompt of a conversation, such as regenerating the latest reply, while that state is still in memory. With the draft model on, only the latest state of each conversation stays in memory, so regenerating or resending an earlier turn after later turns have been sent does not get this shortcut. Such a request resumes only from a shorter state that the disk session store has finished saving. The store saves in the background while the server is idle and may not yet hold a given turn, or may have skipped it; in testing, these regenerations read the whole prompt again. Without the draft model, an exact repeat is never skipped, but earlier turns' states can stay in memory until evicted, so regenerating a later turn can resume from the previous turn's state. Regenerating the first reply of a conversation after later turns, or resending it without the draft model, reads the whole prompt again, because saved state is reused only when it is shorter than the new prompt.",
    "Conversations that share only a system prompt do not share cached work. Saved state is matched by prompt content, not by conversation: a prompt reuses an earlier prompt's state only when it starts with that entire earlier prompt, whichever conversation sent it. No state is saved at the end of a system prompt, so a new conversation that starts with the same system prompt as an earlier one, but has a different first message, reads its whole prompt again. This fix is outside v2.0.2; no target version is assigned.",
    "A client that disconnects while its connection's socket number is 1024 or higher is not detected, so its generation runs to completion and holds its slot. Normal connection counts do not reach this; very many idle keep-alive clients could. This fix is outside v2.0.2; no target version is assigned.",
    "If `max_tokens` cuts off a tool call, `finish_reason` is `length` (or `tool_calls` if an earlier call in the same reply was complete), the cut-off call is left out of the final `tool_calls`, and its raw text is returned in `content`. When streaming, its name and partial `arguments` (incomplete JSON) have already been sent. Raise `max_tokens` for tool use.",
    "The `usage` block in replies does not include `prompt_tokens_details.cached_tokens`. The number of prompt tokens the server reused from saved state is reported in the reply's `tensorfold.cached` field instead (in the final chunk when streaming). For a request that forces a tool call, this count can be too high, even above the prompt's length. v1.8.4 returned this field, so a client that reads it must switch to `tensorfold.cached` when upgrading. This fix is outside v2.0.2; no target version is assigned.",
    "v2.0.1 does not support per-request cache isolation. It ignores the `cache_salt` request field, and all clients of one server share its saved prompt state. A request whose prompt starts with another client's entire earlier prompt reuses that state, which shows in the reply's cached-token count and in a faster first token. v1.8.4's engine honored `cache_salt`, so a deployment that relied on it to keep clients apart is no longer isolated after upgrading. If clients must not learn about each other's prompts, give each one its own server with its own session folder. This fix is outside v2.0.2; no target version is assigned.",
    "Saving a conversation to the disk session store is best-effort. The store saves in the background while the server is idle, so back-to-back requests from other long conversations can keep it from saving a conversation. A later return to that conversation, after it has left the memory cache, then reads its whole prompt again. In testing, returns sent after a pause of several seconds resumed from the disk store.",
    "Image decode allocation failures, including MemoryError, can surface as HTTP 400.",
    "Pillow is not separately version-pinned by this recipe.",
    "Boundary and full-prompt checkpoints share one optional save batch; backpressure can drop both together. All three ranks must run the same release."
  ],
  "security_note": "The server listens on loopback (127.0.0.1) only, with no authentication and no CORS. Reach it through an SSH tunnel or a reverse proxy that adds authentication; don't expose the port.",
  "session_cache_note": "Conversation state, including the prompt's token ids, is cached on each host's own disk (up to 64 GiB per host) so returning to a long conversation is fast. It never leaves your machines. OPERATIONS explains where it lives, how to clear it and how to turn it off (SESSION_TIER=off).",
  "rollback": "To roll back, stop v2.0.2 and start v1.8.4 from its tag, following its own installation guide. v1.8.4 is the documented rollback.",
  "rollback_commands": {
    "existing_checkout": "git fetch --tags && git checkout v1.8.4",
    "existing_checkout_where": "in your v1.8.4 checkout",
    "fresh": "git clone --branch v1.8.4 https://github.com/jakejharris/jspark3.git jspark3-v1.8.4"
  },
  "upgrade_who_should_stay": "v1.8.4 stays available; see rolling back. With base weights, two measured cases favour it (three DGX Sparks; v1.8.4 at its default, reasoning off, and v2.0.1 at reasoning effort low; an appliance comparison with different model IDs, not a same-weights claim). On prose replies, v1.8.4 shows the first visible text about 0.1 s sooner, because v2.0.1 writes a short reasoning passage first (known issue 9). With base weights and a single client on short code replies, the first visible text arrives in about the same time, with v1.8.4 about 2% faster. If you keep very many idle keep-alive clients connected, read known issue 13 first; that fix is outside this release; no target version is assigned.",
  "credits": [
    "Z.AI (GLM-5.3 Flash, the base model, and its chat template)",
    "Hugging Face and the transformers contributors (the GLM-5.3 Flash model code the engine's CUDA path implements)",
    "Vontra, now TensorFold on Hugging Face (the 4-bit MLX base weights)",
    "Ash Hart and the TensorFold contributors (TensorFold 0.3.6.2, the engine release this project forks; DFlash ring-snapshot follow-up 47bf822)",
    "Taus Soe (GLM multi-stream foundation 20dbaba, disk-chain foundation b8a555a/bb16122 and CUDA image input 19680d9, via taussoe/TensorFold)",
    "FlyCockpit (zero-padding dimensions for three-way splitting, credited since v1.0.0)",
    "BTCXoomer (reporting the three-Spark NCCL subnet-routing requirement, credited in v1.1.0)",
    "Inco AI (the DFlash2 draft model)",
    "orcarouter (the refusal-removed source weights)",
    "z-lab (DFlash)",
    "MiaAI-Lab (upstream TensorFold: follower doorbell 358875c, DFlash ring 7c088eb, prefill row-blocking b23c10a and typed-parser hunk fe2b514)",
    "mikolaj92 (sparse-attention pool optimizations: skipping invisible pool tiles and bounding radix selection to visible pools, via upstream TensorFold commits b3b8a39 and f119334)",
    "Dorian (an upstream TensorFold server fix, ported)",
    "turboderp (ExLlamaV3's EXL3 format, which the engine's own decoders read)",
    "QTIP and QuIP# authors (trellis and incoherence-processing foundations used through ExLlamaV3's EXL3 format)",
    "Apple (MLX)",
    "Google DeepMind (Gemma 4, supported by the vendored engine's MLX backend)",
    "Prince Canuma and the mlx-vlm contributors (GLM-5.3 Flash code the engine follows and ports)"
  ],
  "profiles": {
    "public_line": "Each weight variant ships its own measured settings profile."
  },
  "results_conditions_public": "Every set ran on the same build with the same benchmark harness and protocol, each on its weights variant's shipped settings profile. Each set's receipt records its label, the combined digest of the three hosts' data manifests, the draft model, the settings profile and the engine commit.",
  "capabilities": {
    "image_input": true,
    "image_input_public": "Both weight variants accept images in chat messages as inline `data:` URLs (base64). Remote image URLs are refused. Each request takes up to 16 images, at most 32 MB per image and 32 MB in total, and at most 32 megapixels per image. This release does not measure how well the model understands images."
  },
  "site_tiles": [
    "decode_short_tok_s.code",
    "decode_short_tok_s.prose",
    "rigmark.prefill_64k",
    "rigmark.c4"
  ],
  "site_tile_labels": {
    "decode_short_tok_s.code": "One stream code",
    "decode_short_tok_s.prose": "One stream prose",
    "rigmark.prefill_64k": "Cold prefill, 64K prompt (RigMark)",
    "rigmark.c4": "Four at once, end to end (RigMark)"
  },
  "site_tile_sets": {
    "decode_short_tok_s.prose": {
      "set": "O-D",
      "label": "ablit weights"
    }
  },
  "site_tile_sources": {
    "rigmark.prefill_64k": {
      "rigmark_row": "prefill_64k",
      "set": "V-D"
    },
    "rigmark.c4": {
      "rigmark_row": "c4",
      "set": "V-D"
    }
  },
  "site_tile_captions": {
    "rigmark.c4": "short code, end-to-end, 256-token cap per agent"
  },
  "site_card_footer_public": "base weights + draft model unless marked. Historical v2.0.1 measurements.",
  "credits_site": [
    {
      "name": "Z.AI",
      "role": "GLM-5.3 Flash, the base model, and its chat template",
      "url": "https://huggingface.co/zai-org/GLM-5.3-Flash"
    },
    {
      "name": "Hugging Face and the transformers contributors",
      "role": "the GLM-5.3 Flash model code the engine's CUDA path implements",
      "url": "https://github.com/huggingface/transformers"
    },
    {
      "name": "Vontra, now TensorFold on Hugging Face",
      "role": "the 4-bit MLX base weights",
      "url": "https://huggingface.co/TensorFold/GLM-5.3-Flash-MLX-4bit-MTP"
    },
    {
      "name": "Ash Hart and the TensorFold contributors",
      "role": "TensorFold 0.3.6.2, the engine release this project forks; DFlash ring-snapshot follow-up 47bf822",
      "url": "https://github.com/ashhart/TensorFold"
    },
    {
      "name": "Taus Soe",
      "role": "GLM multi-stream foundation 20dbaba, disk-chain foundation b8a555a/bb16122 and CUDA image input 19680d9, via taussoe/TensorFold"
    },
    {
      "name": "FlyCockpit",
      "role": "zero-padding dimensions for three-way splitting, credited since v1.0.0"
    },
    {
      "name": "BTCXoomer",
      "role": "reporting the three-Spark NCCL subnet-routing requirement, credited in v1.1.0"
    },
    {
      "name": "Inco AI",
      "role": "the DFlash2 draft model",
      "url": "https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2"
    },
    {
      "name": "orcarouter",
      "role": "the refusal-removed source weights",
      "url": "https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-MLX"
    },
    {
      "name": "z-lab",
      "role": "DFlash",
      "url": "https://github.com/z-lab/dflash"
    },
    {
      "name": "MiaAI-Lab",
      "role": "upstream TensorFold: follower doorbell 358875c, DFlash ring 7c088eb, prefill row-blocking b23c10a and typed-parser hunk fe2b514",
      "url": "https://github.com/ashhart/TensorFold/commit/358875c15506f5f0f47ecbab16e6fe42585cea65"
    },
    {
      "name": "mikolaj92",
      "role": "sparse-attention pool optimizations: skipping invisible pool tiles and bounding radix selection to visible pools, via upstream TensorFold commits b3b8a39 and f119334"
    },
    {
      "name": "Dorian",
      "role": "an upstream TensorFold server fix, ported",
      "url": "https://github.com/ashhart/TensorFold/commit/50dfe38aaee41edf49657d57d2378180c9553713"
    },
    {
      "name": "turboderp",
      "role": "ExLlamaV3's EXL3 format, which the engine's own decoders read",
      "url": "https://github.com/turboderp-org/exllamav3"
    },
    {
      "name": "QTIP and QuIP# authors",
      "role": "trellis and incoherence-processing foundations used through ExLlamaV3's EXL3 format"
    },
    {
      "name": "Apple",
      "role": "MLX",
      "url": "https://github.com/ml-explore/mlx"
    },
    {
      "name": "Google DeepMind",
      "role": "Gemma 4, supported by the vendored engine's MLX backend"
    },
    {
      "name": "Prince Canuma and the mlx-vlm contributors",
      "role": "GLM-5.3 Flash code the engine follows and ports",
      "url": "https://github.com/Blaizzy/mlx-vlm"
    }
  ],
  "install_costs": {
    "conditions": "Download, build, conversion and disk figures were measured on one DGX Spark over our connection while another job shared its network and disk. Start times and the kernel cache were measured on the three-Spark cluster. Pull and download times depend on your connection.",
    "image_disk": "about 25 GB",
    "image_pull_time": "about 35 minutes",
    "wheels_disk": "under 50 MB",
    "build_time": "about 7 seconds",
    "download_time": "about 39 minutes, including the checksum check",
    "drafter_time": "about 31 seconds",
    "one_third_disk": "63.9 GB for the first third, 62.8 GB for each of the other two",
    "split_time": "about 3.5 minutes per third",
    "split_ram": "about 4 GiB",
    "kernel_cache_disk": "about 45 MB per host",
    "first_start_time": "about 5 minutes for all three hosts, including compiling the kernels",
    "warm_start_time": "about 5 minutes"
  },
  "errata": {
    "engine_docs_public": "Two lines in the engine's own files are out of date. Correcting them is outside v2.0.2; no target version is assigned. `engine/NOTICE` says to select `--drafter none --draft-policy c7:0.3`. Don't add `--draft-policy` yourself: `scripts/serve.sh --drafter none` (or `DRAFTER=none` in `cluster.env`) applies the value from the weights' own settings profile, and the default base weights use `c7:0.45`. `engine/README.md` has an outdated relative facts link. Current public facts are in `release/RELEASE-FACTS.md`; historical v2.0.1 results are in the README's Results section, `docs/BENCHMARKS.md` and `release/MEASUREMENTS-v2.0.1.md`."
  },
  "upgrade_thinking_public": "Reasoning is now always on. v1.8.4 had it off by default and honoured requests to turn it off; v2.0.1 runs a request with no reasoning setting at High and treats a request to turn it off as low (known issue 9). Replies begin with a reasoning passage, returned as `reasoning_content`, before the visible text, and it uses part of `max_tokens`.",
  "metric_conditions": {
    "all": "Historical v2.0.1 measurements; v2.0.2 has no new performance results. Every set was measured on the same build, each on its weights variant's shipped settings, and no figure is a best run. Rates and times are medians, with the number of runs given below; the token gap is given as both a median and a maximum, and the context window is a setting, not a measurement. Short-reply decode is reported separately for code and for prose: the per-stream rate of replies capped at 256 tokens (median of 3 each). Long decode is one greedy code stream of up to 96 tokens after a 32K-token prompt (median of 3). Aggregate decode is the wall-clock rate of concurrent greedy replies, capped at 96 tokens, to a fixed mix of varied short prompts (41 to 62 tokens each) that is the same for every set (median of 3). Draft acceptance depends on the prompt, so it is also reported for a single repeated prompt. Cold time to first token uses exactly the stated number of prompt tokens with nothing cached (median of 2). The 8-request time to first token is the median wait for the first token when 8 short prompts from the mix (41 to 62 tokens) are sent at once; without the draft model, prompts that arrive together are not read together in one batch, and first tokens arrive later (see known issues). The 8-stream token gap is the longest pause seen by running streams while a prompt of about 8,000 or 36,000 tokens joins, reported as the median and the maximum of that pause across runs. Draft acceptance is the number of draft tokens accepted per verify step, not counting the token the model adds itself, with accepted over proposed tokens alongside, both measured at 8 concurrent requests. The no-draft-model set is a reduced run. The benchmark client runs on a separate machine on the same local network, so client-side times include one network hop. The benchmark figures in the result tables, including the stall bounds, come from chat requests at low reasoning effort. Installation, disk and startup figures are not chat measurements. Low effort still reasons before it answers, and decode and aggregate rates count reasoning tokens. A chat request that sets no reasoning effort runs at high effort, so its replies are longer and its rates can differ from these. Time to first token is measured to the first streamed token, reasoning or text. In the cold-prompt, newcomer and saved-session tests, that first token was visible text in all but two replies: one 64K cold-prompt reply with base weights and the saved-session return with refusal-removed (ablit) weights reached their eight-token limit on reasoning and showed no text. In the eight-client short-prompt test, most replies began with a short reasoning passage, so visible text arrives later than the first token, and some short replies spent their 96-token limit on reasoning and showed no text; each set's first-token figure is shown with its own visible-text note.",
    "first_token_partner_estimators": "The first-token notes use two measures: with the draft model, the median time to first visible text; without it, the median gap from first token to first visible text in the replies that showed text, because a median taken only over replies that showed text would come out below the first-token median taken over all replies."
  },
  "_historical_source": {
    "file": "RELEASE-FACTS.md",
    "sha256": "70490d33185ec2a084a709127f31417b0ea7ce83a1db5949cd070428118c3167",
    "status": "final",
    "results": {
      "file": "results-v2.0.1.facts-b64268e2.json",
      "sha256": "1a78dd4e57494433dbb1cf1ce66813b8783c4f5b9fe88807efb03d42541b2923",
      "status": "measured"
    }
  },
  "_evidence": {
    "release_identity": "manifests/release.json",
    "live_receipt": "release/v2.0.2/live-retry.json",
    "live_receipt_sha256": "4e0f945bd820ae2dcc9fc15b877da6f07a549dd742862ce98145d3c86141b260",
    "performance_measured_version": "v2.0.1",
    "engine_package_sha256": "04f2a77806a8f0379d8e0aa329e290adc72b239230a4fda2bfeca41641bbfff1"
  }
}
```
