"""The real pooled constructor serves overlapping caller threads on a tiny TP3 shape."""

import threading

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('CUDA only', allow_module_level=True)

from tensorfold.families.glm5_next.cuda.engine import GlmEngine
from test_glm_tp3 import _Copies, _checkpoint
from test_glm_engine import _drafter


def test_real_constructor_routes_concurrent_dflash_requests(tmp_path, monkeypatch):
    target, draft = tmp_path / 'target', tmp_path / 'draft'
    _checkpoint(target)
    _drafter(draft)
    monkeypatch.setenv('TF_GLM_VISION', '0')
    monkeypatch.setenv('TF_GLM_CACHE_GIB', '0.03')
    engine = GlmEngine(target, rank=0, world=3, master='', port=0, comm=_Copies(0),
                       drafter=draft, context=4096, prefill_rows=64, parallel=2, policy='f7')
    assert engine.concurrent and engine.multi.owner is engine
    assert engine.e.st.capacity == engine.capacity_plan['cache_slots'] + 64
    prompts = [[11] * 23, [17] * 31]
    wanted = []
    for prompt in prompts:
        out = []
        engine.request.policy = '0'
        engine.request.stop_eos = False
        engine.generate(prompt, 15, None, lambda new: out.extend(new), draft=False)
        wanted.append(out)
    barrier = threading.Barrier(2)
    got, errors = [[], []], []

    def ask(i):
        try:
            engine.request.policy = 'f7'
            engine.request.stop_eos = False
            barrier.wait(timeout=20)
            stats = engine.generate(prompts[i], 15, None, lambda new: got[i].extend(new))
            assert stats['drafts']
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=ask, args=(i,), daemon=True) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert not any(t.is_alive() for t in threads), 'constructor scheduler stranded a request'
    assert not errors
    assert got == wanted
    assert not engine.multi.streams and not engine.multi.pool.spans
