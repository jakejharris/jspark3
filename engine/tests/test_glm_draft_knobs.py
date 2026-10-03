"""GLM-5.3-Flash on CUDA: --draft-policy and --drafter-bits reach the engine (host side, no GPU)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tensorfold import cli

pytest.importorskip("torch")


def _serve(tmp_path, monkeypatch, *flags):
    import tensorfold.cuda.server as server

    made = []
    family = SimpleNamespace(title="Test family", model_type="test",
                             package=SimpleNamespace(cuda_engine=lambda *a, **k: made.append(k) or
                                                     SimpleNamespace(max_len=8192)))
    monkeypatch.setattr(server, "App", lambda *a, **k: SimpleNamespace(effective_context_window=8185))
    monkeypatch.setattr(server, "serve", lambda *a: None)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--no-drafts", *flags])
    assert cli._serve_cuda(args, family, tmp_path, 262144) == 0
    return made[0]


def test_the_cli_hands_the_policy_and_a_non_default_drafter_width_to_the_engine(tmp_path, monkeypatch):
    plain = _serve(tmp_path, monkeypatch)
    assert "draft_policy" not in plain and "drafter_bits" not in plain      # other families never see them
    knobs = _serve(tmp_path, monkeypatch, "--draft-policy", "fc7:0.3", "--drafter-bits", "0")
    assert knobs["draft_policy"] == "fc7:0.3" and knobs["drafter_bits"] == 0


def test_cuda_engine_takes_the_policy_and_the_drafters_width(tmp_path, monkeypatch):
    from tensorfold.families import glm5_next
    from tensorfold.families.glm5_next.cuda import engine

    built = []
    monkeypatch.setattr(engine, "GlmEngine", lambda *a, **k: built.append(k) or k)
    common = dict(drafter=str(tmp_path), tp=3, rank=0, master="192.0.2.1")
    glm5_next.cuda_engine(tmp_path, **common)
    assert built[-1]["policy"] == "auto" and built[-1]["drafter_bits"] == 4
    glm5_next.cuda_engine(tmp_path, **common, draft_policy="fc7:0.3")
    assert built[-1]["policy"] == "fc7:0.3"
    glm5_next.cuda_engine(tmp_path, **common, draft_policy="f7", drafter_bits=0)
    assert built[-1]["policy"] == "f7" and built[-1]["drafter_bits"] == 16
    with pytest.raises(ValueError, match="both choose"):
        glm5_next.cuda_engine(tmp_path, **common, draft_policy="f7", mtp_drafts=3)
    with pytest.raises(ValueError, match="needs --drafter"):
        glm5_next.cuda_engine(tmp_path, **{**common, "drafter": ""}, draft_policy="fc7:0.3")
    with pytest.raises(ValueError, match="4-bit or bf16"):
        glm5_next.cuda_engine(tmp_path, **common, drafter_bits=8)


def test_depth_seven_policies_decode_to_seven_drafts():
    from tensorfold.families.glm5_next.cuda.engine import MAX_ROWS, decode_policy, encode_policy

    assert MAX_ROWS - 1 == 7                        # a pending token and up to 7 drafts: DFlash2's block of 8
    fixed = decode_policy(encode_policy("f7"))
    assert encode_policy("f7") == [11, 7, 0, 0] and fixed.fixed and fixed.most == 7 and fixed.confidence == 0
    conf = decode_policy(encode_policy("fc7:0.3"))
    assert encode_policy("fc7:0.3") == [13, 7, 300000, 0] and conf.fixed and conf.most == 7 and conf.confidence == 0.3
    with pytest.raises(ValueError):
        encode_policy("f8")                         # past the widest verify window
