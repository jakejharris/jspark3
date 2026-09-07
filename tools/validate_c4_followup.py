"""Validate the separate post-release C4 receipt export without altering sealed evidence."""
import hashlib
import json
from pathlib import Path
from statistics import median

EVIDENCE = 'results/evidence/candidate/cadence-v11-c4'


def validate(root: Path) -> dict:
    """Return claims only after all receipt, estimator and display checks pass."""
    directory = root / EVIDENCE
    claims = json.loads((directory / 'CLAIMS.json').read_text())
    assert claims['release'] == '1.1.0'
    assert claims['release_commit'] == 'bde15cf769203cf9c387720810f0cc09dea55c66'
    assert claims['author_commit'] == 'e93fc87d54c8699e98b63a764ab260bf9d446c52'
    assert set(claims['series']) == {'original', 'instrumented'}
    display = {}
    for name, series in claims['series'].items():
        assert series['receipt'] == name.upper() + '-RESULTS.json'
        payload = (directory / series['receipt']).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == series['sha256'], 'receipt hash drift'
        data = json.loads(payload)
        assert data['status'] == 'PASS' and data['retries'] == 0
        assert data['requests'] == len(data['request_results']) == 24
        assert data['completion_tokens'] == sum(r['completion_tokens'] for r in data['request_results']) == 8496
        assert [r['repetition'] for r in data['repetitions']] == [1, 2, 3]
        if name == 'original':
            assert data['author_sources_unchanged'] is True
            assert data['all_payloads_byte_identical_to_historical'] is True
            assert data['no_return_token_ids'] is True
        values = {'aggregate_decode_tok_s': [], 'mean_per_stream_decode_tok_s': []}
        for repetition in data['repetitions']:
            assert [w['concurrency'] for w in repetition['waves']] == [1, 2, 4]
            wave = repetition['waves'][2]
            streams = wave['streams']
            assert len(streams) == 4 and all(s['completion_tokens'] == 400 for s in streams)
            aggregate = sum(s['completion_tokens'] - 1 for s in streams) / (
                max(s['last_visible_s'] for s in streams) - min(s['first_visible_s'] for s in streams))
            per_stream = sum((s['completion_tokens'] - 1) / (s['last_visible_s'] - s['first_visible_s']) for s in streams) / 4
            for metric, observed in [('aggregate_decode_tok_s', aggregate), ('mean_per_stream_decode_tok_s', per_stream)]:
                author = wave['original_' + metric]
                assert abs(observed / author - 1) <= 0.002, 'author/proxy discrepancy exceeds declared tolerance'
                values[metric].append(author)
        for metric, rates in values.items():
            assert series[metric] == rates, 'wave export drift'
            assert series['median_' + metric] == median(rates), 'median drift'
            for i, rate in enumerate(rates, 1):
                display[f'v11.c4.{name}.{metric}.r{i}'] = f'{rate:.2f}'
            display[f'v11.c4.{name}.{metric}.median'] = f'{median(rates):.2f}'
    historical = json.loads((root / 'results/results.json').read_text())
    assert claims['historical_prefill_source'] == 'results/results.json#/prefill_matched/candidate_prefill_tok_s'
    display['historical.prefill.rounded'] = f"{historical['prefill_matched']['candidate_prefill_tok_s']:,.0f}"
    assert claims['display'] == display, 'display drift'
    classes = {k: 'candidate_v11_descriptive' for k in display}
    classes['historical.prefill.rounded'] = 'internal_ablation'
    assert claims['display_class'] == classes
    assert 'no causal release comparison' in claims['limits']
    return claims
