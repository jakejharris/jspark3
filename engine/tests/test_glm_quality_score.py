"""CPU checks for the default-off, TP-aware GLM quality scoring hook."""
from __future__ import annotations

import hashlib
import io
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cuda import server
from tensorfold.cuda.streams import Stream
from tensorfold.families.glm5_next import cuda as glm_cuda
from tensorfold.families.glm5_next.cuda import quality_score
from tensorfold.families.glm5_next.cuda.batched import BatchedDecoder, FILL_SLICE, SCORE
from tensorfold.families.glm5_next.cuda.engine import GlmEngine


def test_validate_score_request_refuses_bad_boundary_and_ids():
    valid = {"model": "glm53", "token_ids": [1, 2, 3], "score_start": 2}
    assert quality_score.validate_request(valid, "glm53", 10, 16) == ([1, 2, 3], 2)
    for change in ({"score_start": 0}, {"score_start": 3}, {"score_start": True},
                   {"token_ids": [1, True, 3]}, {"token_ids": [1, 10, 3]},
                   {"model": "other"}, {"extra": 1}):
        with pytest.raises(ValueError):
            quality_score.validate_request({**valid, **change}, "glm53", 10, 16)
    with pytest.raises(ValueError, match="2 to 2"):
        quality_score.validate_request(valid, "glm53", 10, 2)
    with pytest.raises(ValueError, match="at most 256"):
        quality_score.validate_request({"model": "glm53", "token_ids": [1] * 258,
                                        "score_start": 1}, "glm53", 10, 300)


def test_distributed_nll_uses_all_vocab_shards():
    shards = [torch.tensor([0.0, 2.0]), torch.tensor([-1.0, 0.5]), torch.tensor([3.0])]
    offset, target = 2, 3
    class Comm:
        def all_gather(self, packed, out):
            rows = []
            for i, logits in enumerate(shards):
                hit = logits[target - sum(len(s) for s in shards[:i])] if i == 1 else torch.tensor(float("-inf"))
                rows.extend([torch.logsumexp(logits, 0), hit])
            out.copy_(torch.stack(rows))
    got = quality_score.distributed_nll(shards[1], target, offset, Comm(), 3)
    full = torch.cat(shards)
    assert got == pytest.approx((torch.logsumexp(full, 0) - full[target]).item(), abs=1e-6)


def test_prefill_scores_prior_rows_and_shifted_start_changes_hash(monkeypatch):
    class State:
        pos = 0
    class Engine:
        def __init__(self):
            self.st = State()
            self.prefill_rows = 2
            self.w = NS(norm=None, cfg=NS(eps=0), head=None, vocab_offset=0, world=1,
                        comm=NS(all_gather=lambda send, recv: recv.copy_(send)))
            self.pbuf = NS(hidden=torch.zeros((2, 3)), fnormed=torch.zeros((2, 3)),
                           fxs=torch.zeros((2, 1)), logits=torch.zeros((1, 3)), ids=[])
        def reset(self):
            self.st.pos = 0
    def stage(w, st, b, ids):
        b.ids = list(ids)
        return len(ids)
    def compute(w, st, b, rows, **kwargs):
        assert kwargs["logits"] is False and kwargs["host_pos"] == st.pos
        for row in range(rows):
            b.hidden[row] = torch.tensor([float(b.ids[row]), 0, 0])
    def mm(b, x, head, scales, out):
        out[0] = torch.tensor([x[0, 0], 0, -x[0, 0]])
        return out
    def commit(w, st, b, rows, keep):
        assert rows == keep
        st.pos += keep
    forward = NS(stage=stage, compute=compute, mm=mm, commit=commit, chunks_for=lambda st, rows: 1)
    glue = NS(rmsnorm=lambda source, norm, eps, dest, scales: dest.copy_(source))
    monkeypatch.setitem(sys.modules, "tensorfold.families.glm5_next.cuda.forward", forward)
    monkeypatch.setitem(sys.modules, "tensorfold.families.glm5_next.cuda.glue", glue)
    monkeypatch.setattr(glm_cuda, "glue", glue, raising=False)
    ids = [1, 2, 0, 2, 1]
    scores = quality_score.score_prefill(Engine(), ids, 2)
    expected = []
    for j in range(2, len(ids)):
        row = torch.tensor([float(ids[j - 1]), 0, -float(ids[j - 1])])
        expected.append((torch.logsumexp(row, 0) - row[ids[j]]).item())
    assert scores == pytest.approx(expected, abs=1e-6)
    shifted = quality_score.score_prefill(Engine(), ids, 3)
    digest = lambda values: hashlib.sha256(json.dumps(values).encode()).hexdigest()
    assert shifted == pytest.approx(expected[1:], abs=1e-6)
    assert digest(scores) != digest(shifted)  # negative control: wrong boundary cannot silently pass


def test_flag_off_refuses_without_gpu_dispatch():
    engine = object.__new__(GlmEngine)
    engine.quality_score_enabled = False
    with pytest.raises(ValueError, match="off"):
        engine.score([1, 2], 1)


def test_batched_score_sends_one_named_command_and_skips_finish(monkeypatch):
    decoder = object.__new__(BatchedDecoder)
    decoder.streams, decoder.sessions = {}, None
    sent = []
    decoder.broken = None
    decoder.fill_owner = None
    decoder.express_owner = None
    decoder._send = lambda values, **kwargs: sent.append(values)
    decoder.owner = NS(e=object(), quality_score_enabled=True)
    monkeypatch.setattr(quality_score, "score_prefill", lambda e, ids, start: [float(start), float(len(ids))])
    stream = Stream([4, 5, 6], 1)
    stream.quality_score_start = 1
    stream.quality_result = []
    decoder._begin_admit(stream, stepped=True)
    assert sent == [[SCORE, 1], [4, 5, 6]]
    assert stream.done and stream.quality_result == [1.0, 3.0]
    decoder.finish([stream])
    assert sent == [[SCORE, 1], [4, 5, 6]]


@pytest.mark.parametrize("lane", ["fill_owner", "express_owner"])
def test_score_opcode_is_distinct_and_waits_for_sliced_prefill(lane):
    assert FILL_SLICE == 5 and SCORE == 6
    decoder = object.__new__(BatchedDecoder)
    decoder.fill_owner = decoder.express_owner = None
    setattr(decoder, lane, 7)
    decoder.filling = {}
    decoder.live = lambda: 0
    stream = Stream([4, 5], 1)
    stream.quality_score_start = 1
    assert decoder.can_admit(stream) is False
    with pytest.raises(ValueError, match="sliced GLM chunk"):
        decoder._begin_admit(stream, stepped=True)


def test_batched_follower_runs_same_score_command(monkeypatch):
    decoder = object.__new__(BatchedDecoder)
    decoder.streams, decoder.sessions = {}, None
    decoder.broken = None
    decoder.owner = NS(e=object(), quality_score_enabled=True)
    messages = iter(([SCORE, 2], [4, 5, 6], []))
    decoder.share = lambda _: next(messages)
    seen = []
    monkeypatch.setattr(quality_score, "score_prefill", lambda e, ids, start: seen.append((ids, start)))
    with pytest.raises(RuntimeError, match="empty GLM worker message"):
        decoder.follow()
    assert seen == [([4, 5, 6], 2)]


def test_single_stream_score_sends_header_before_tokens(monkeypatch):
    engine = object.__new__(GlmEngine)
    engine.quality_score_enabled = True
    engine.concurrent = False
    engine.e = object()
    engine.live = [1]
    engine._take_over = lambda keep: None
    sent = []
    engine._share = lambda values: sent.append(values)
    monkeypatch.setattr(quality_score, "score_prefill", lambda e, ids, start: [0.5])
    assert engine.score([4, 5], 1) == [0.5]
    assert sent == [[-1, 1], [4, 5]]
    assert engine.live == []


def test_single_stream_follower_saves_live_rows_before_score(monkeypatch):
    engine = object.__new__(GlmEngine)
    engine.concurrent = False
    engine.e = object()
    engine.live = [1, 2, 3]
    messages = iter(([-1, 1], [4, 5]))
    engine._share = lambda _: next(messages)
    seen = []
    engine._take_over = lambda keep: seen.append((keep, list(engine.live)))
    monkeypatch.setattr(quality_score, "score_prefill", lambda e, ids, start: seen.append((ids, start, list(engine.live))))
    with pytest.raises(StopIteration):
        engine.follow()
    assert seen == [([], [1, 2, 3]), ([4, 5], 1, [])]


def test_concurrent_score_returns_worker_result():
    engine = object.__new__(GlmEngine)
    engine.quality_score_enabled = True
    engine.concurrent = True
    def submit(ids, count, sampling, draft, callback, **request):
        assert (ids, count, sampling, draft) == ([4, 5], 1, None, False)
        assert request["quality_score_start"] == 1
        request["quality_result"].append(0.75)
        return {}
    engine.scheduler = NS(submit=submit)
    assert engine.score([4, 5], 1) == [0.75]


def _http(app, body):
    handler = server.make_handler(app)
    h = object.__new__(handler)
    h.command, h.path = "POST", "/v1/quality/score"
    h.request_version, h.requestline = "HTTP/1.1", "POST /v1/quality/score HTTP/1.1"
    h.date_time_string = lambda: "Thu, 01 Jan 1970 00:00:00 GMT"
    data = json.dumps(body).encode()
    h.headers = {"Content-Length": str(len(data))}
    h.rfile, h.wfile, h.connection = io.BytesIO(data), io.BytesIO(), object()
    h.client_address = ("127.0.0.1", 2000)
    h.close_connection = False
    h.do_POST()
    head, payload = h.wfile.getvalue().split(b"\r\n\r\n", 1)
    return head, json.loads(payload)


def test_http_quality_flag_off_and_on_contract():
    sent = []
    engine = NS(quality_score_enabled=False, limit=10, w=NS(cfg=NS(vocab=100)),
                score=lambda ids, start: sent.append((ids, start)) or [0.25] * (len(ids) - start))
    app = NS(engine=engine, served="glm53", tok=NS(get_vocab_size=lambda **kw: 100),
             _serving_lock=lambda *args: nullcontext())
    body = {"model": "glm53", "token_ids": [1, 2, 3], "score_start": 2}
    head, payload = _http(app, body)
    assert b" 404 " in head and not sent
    engine.quality_score_enabled = True
    head, payload = _http(app, body)
    assert b" 200 " in head and payload == {"token_ids": [1, 2, 3], "score_start": 2, "nll": [0.25]}
    assert sent == [([1, 2, 3], 2)]
    head, _ = _http(app, {**body, "score_start": 3})
    assert b" 400 " in head and sent == [([1, 2, 3], 2)]


def test_scoring_flag_does_not_change_normal_completion_bytes(monkeypatch):
    from test_cuda_http_hardening import app_stub, request

    monkeypatch.setattr(server.uuid, "uuid4", lambda: NS(hex="fixedqualityid"))
    monkeypatch.setattr(server.time, "time", lambda: 1)
    body = {"messages": [{"role": "user", "content": "ordinary"}], "max_tokens": 1}
    off = request(app_stub(engine=NS(quality_score_enabled=False)), body)
    on = request(app_stub(engine=NS(quality_score_enabled=True)), body)
    assert off == on
