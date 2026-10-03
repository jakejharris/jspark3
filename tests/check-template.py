#!/usr/bin/env python3
"""Check template/chat-template.jinja: the stock file plus our six lines, pinned, and its Reasoning Effort line.

python3 tests/check-template.py   (the render cases need jinja2 and print SKIP without it)
"""
import hashlib
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
STOCK_SHA256 = '34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891'  # weights.sha256: chat_template.jinja
OURS = b'''{#- JSpark3: a request that switches thinking off (enable_thinking or thinking set false) thinks at Low, not Max; this template has no no-think mode. -#}
{%- if effective_reasoning_effort == 'max' and ((enable_thinking is defined and not enable_thinking) or (thinking is defined and not thinking)) -%}
{%- set effective_reasoning_effort = 'low' -%}
{%- endif -%}
{#- JSpark3 (effort-high-default): with no reasoning_effort given and thinking on, answer at High rather than Max. -#}
{%- if reasoning_effort is not defined and effective_reasoning_effort == 'max' -%}{%- set effective_reasoning_effort = 'high' -%}{%- endif -%}
'''


def main():
    data = (ROOT / 'template/chat-template.jinja').read_bytes()
    lines = data.splitlines(keepends=True)
    assert b''.join(lines[2:8]) == OURS, 'lines 3-8 are not our block'
    stock = b''.join(lines[:2] + lines[8:])
    assert hashlib.sha256(stock).hexdigest() == STOCK_SHA256, 'removing our six lines does not give the stock template'
    print(f'PASS removing lines 3-8 gives the stock template (sha256 {STOCK_SHA256})')
    pins = dict(re.findall(r'^(\w+)=(\S*)', (ROOT / 'pins.env').read_text(), re.M))
    assert hashlib.sha256(data).hexdigest() == pins['TEMPLATE_SHA256'], 'template does not match TEMPLATE_SHA256'
    print(f'PASS template matches pins.env TEMPLATE_SHA256={pins["TEMPLATE_SHA256"]}')
    try:
        import jinja2
        import jinja2.ext
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError:
        print('SKIP render cases: jinja2 is not importable')
        return
    # The environment the engine renders chat templates with.
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    template = env.from_string(data.decode())
    messages = [{'role': 'user', 'content': 'hi'}]
    cases = [({}, 'High'), ({'enable_thinking': True}, 'High'), ({'reasoning_effort': 'max'}, 'Max'),
             ({'reasoning_effort': 'medium'}, 'Max'), ({'reasoning_effort': 'garbage'}, 'Max'),
             ({'reasoning_effort': None}, 'Max'), ({'enable_thinking': False}, 'Low'), ({'thinking': False}, 'Low'),
             ({'reasoning_effort': 'high'}, 'High'), ({'reasoning_effort': 'low'}, 'Low'),
             ({'reasoning_effort': 'high', 'enable_thinking': False}, 'High'),
             ({'reasoning_effort': 'max', 'enable_thinking': False}, 'Low')]
    for kwargs, expected in cases:
        text = template.render(messages=messages, add_generation_prompt=True, **kwargs)
        found = re.findall(r'Reasoning Effort: (\w+)', text)
        assert found == [expected], (kwargs, found)
        print(f'PASS render {kwargs or "unset"}: Reasoning Effort: {expected}')
    print(f'PASS template render cases (jinja2 {jinja2.__version__}, engine sandbox settings)')


if __name__ == '__main__':
    main()
