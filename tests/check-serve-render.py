#!/usr/bin/env python3
"""Render scripts/serve.sh --dry-run for each rank, weights profile and mode, and compare with the measured containers.

    python3 tests/check-serve-render.py

For each weights profile (config/profiles/base.env and ablit.env) and ranks 0, 1 and 2: the serve command equals
tests/measured-serve.json with that profile's draft policy, the mount destinations and modes equal the measured ones
(less the idle /capture mount), the container options and per-box variables are present, the engine environment is
config/serve.env with nothing from the profile, and the jspark3.profile label is the profile's sha256. The ablit
profile must hold the measured values. Then:
  --drafter none          --drafter none --draft-policy <the profile's NO_DRAFTER_POLICY> and no /drafter mount
  --session-tier off      TF_GLM_SESSION_DISK=0 and an empty TF_GLM_DISK_DIR after the env file, no /sessions mount
  a changed profile       anything but SERVE_DRAFT_POLICY and NO_DRAFTER_POLICY in their ranges stops serve.sh before
                          docker; allowed changes render with a warning and their own label
Needs no docker and no data: dry runs only, on a copy of the tree for the changed profiles. Exit 0 when every check
passes.
"""
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
MEASURED = json.loads((HERE / 'tests/measured-serve.json').read_text())
failures = []


def check(ok, what):
    print(f'{"PASS" if ok else "FAIL"}  {what}')
    if not ok:
        failures.append(what)


def settings(path):
    return dict(line.split('=', 1) for line in path.read_text().splitlines() if line and not line.startswith('#'))


def serve(rank, *flags, tree=HERE):
    example = (tree / 'cluster.env.example').read_text()
    with tempfile.TemporaryDirectory() as tmp:
        env_file = Path(tmp) / 'cluster.env'
        env_file.write_text(example.replace('\nRANK=0 ', f'\nRANK={rank} '))
        env = dict(os.environ, CLUSTER_ENV=str(env_file))
        return subprocess.run([str(tree / 'scripts/serve.sh'), '--dry-run', '--no-wait', *flags],
                              capture_output=True, text=True, env=env)


def render(rank, *flags, tree=HERE):
    p = serve(rank, *flags, tree=tree)
    if p.returncode != 0:
        sys.exit(f'serve.sh --dry-run failed for rank {rank} {flags}: {p.stderr.strip()}')
    line = next(x for x in p.stdout.splitlines() if x.startswith('+ docker run') or x.startswith('+docker run'))
    words = shlex.split(line[1:])
    image = next(i for i, w in enumerate(words) if w.startswith('nvcr.io/'))
    opts, tail = words[2:image], words[image + 1:]
    mounts, envs, options, labels, i = {}, [], [], {}, 0
    while i < len(opts):
        w = opts[i]
        if w == '-v':
            parts = opts[i + 1].split(':')
            mounts[parts[1]] = not (len(parts) > 2 and parts[2] == 'ro')
            i += 2
        elif w == '-e':
            envs.append(opts[i + 1])
            i += 2
        elif w == '--label':
            key, _, value = opts[i + 1].partition('=')
            labels[key] = value
            i += 2
        elif w in ('--env-file', '--name', '--gpus', '--network', '--ulimit', '--device', '--cap-add'):
            options.append(f'{w} {opts[i + 1]}')
            i += 2
        else:
            options.append(w)
            i += 1
    return {'cmd': tail[2], 'mounts': mounts, 'env': envs, 'options': options, 'labels': labels, 'bash': tail[:2],
            'stderr': p.stderr}


def with_policy(cmd, policy):
    return re.sub(r'--draft-policy \S+', f'--draft-policy {policy}', cmd)


shared = settings(HERE / 'config/serve.env')
for weights, ref in sorted(MEASURED['profiles'].items()):
    path = HERE / ref['file']
    values = settings(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    check(sha == ref['sha256'] and values == ref['settings'],
          f'{weights}: {ref["file"]} matches its recorded profile checksum')
    check(not set(values) & set(shared), f'{weights}: no profile key is also set in config/serve.env')
    check(sorted(values) == ['NO_DRAFTER_POLICY', 'SERVE_DRAFT_POLICY'],
          f'{weights}: the profile holds exactly SERVE_DRAFT_POLICY and NO_DRAFTER_POLICY')
    for rank in ('0', '1', '2'):
        want = MEASURED['ranks'][rank]
        got = render(rank, '--weights', weights)
        check(got['bash'] == ['bash', '-c'] and got['cmd'] == with_policy(want['cmd'], values['SERVE_DRAFT_POLICY']),
              f'{weights} rank {rank}: serve command is the measured one with the profile draft policy '
              f'{values["SERVE_DRAFT_POLICY"]}')
        want_mounts = {k: v for k, v in want['mounts'].items() if k not in MEASURED['omitted_mounts']}
        check(got['mounts'] == want_mounts, f'{weights} rank {rank}: mounts and modes are the measured ones (less /capture)')
        check(all(o in got['options'] for o in MEASURED['container_options']),
              f'{weights} rank {rank}: container options present')
        check(any(o.startswith('--env-file ') and o.endswith('/config/serve.env') for o in got['options']),
              f'{weights} rank {rank}: shared engine environment comes from config/serve.env')
        names = sorted(e.split('=', 1)[0] for e in got['env'])
        check(names == sorted(MEASURED['per_box_env']), f'{weights} rank {rank}: only the per-box variables are added')
        check(got['labels'].get('jspark3.profile') == sha and got['labels'].get('jspark3.weights') == weights,
              f'{weights} rank {rank}: jspark3.profile label is the profile sha256')
        check('differs from the file this release shipped' not in got['stderr'] or not (HERE / 'SHA256SUMS').exists(),
              f'{weights} rank {rank}: no changed-profile warning for the shipped file')
    none = render('1', '--weights', weights, '--drafter', 'none')
    check(f' --drafter none --draft-policy {values["NO_DRAFTER_POLICY"]} ' in none['cmd'] and '/drafter' not in none['mounts'],
          f'{weights} --drafter none: no drafter, prediction-head policy {values["NO_DRAFTER_POLICY"]}, no /drafter mount')
ablit = settings(HERE / MEASURED['profiles']['ablit']['file'])
check(all(ablit.get(k) == v for k, v in MEASURED['measured_settings'].items()),
      'ablit profile holds the measured values ' + ' '.join(f'{k}={v}' for k, v in MEASURED['measured_settings'].items()))

off = render('0', '--session-tier', 'off')
check('TF_GLM_SESSION_DISK=0' in off['env'] and 'TF_GLM_DISK_DIR=' in off['env'],
      '--session-tier off: TF_GLM_SESSION_DISK=0 and an empty TF_GLM_DISK_DIR')
check('/sessions' not in off['mounts'], '--session-tier off: no /sessions mount')
check(off['cmd'] == render('0')['cmd'], '--session-tier off: serve command unchanged')
check(shared.get('TF_GLM_SESSION_DISK') == '1' and shared.get('TF_GLM_DISK_DIR') == '/sessions',
      'config/serve.env keeps the measured session tier (the switch overrides it per start)')

# Changed profiles, on a copy of the tree: what serve.sh refuses, and what it renders with a warning.
GOOD = 'SERVE_DRAFT_POLICY=fq7:0.3\nNO_DRAFTER_POLICY=c7:0.3\n'
REFUSED = [
    ('SERVE_DRAFT_POLICY=c7:0.3\nNO_DRAFTER_POLICY=c7:0.3\n', 'must be fq7:<c> or fc7:<c>'),
    ('SERVE_DRAFT_POLICY=fq7:0.95\nNO_DRAFTER_POLICY=c7:0.3\n', 'confidence must be from 0.1 to 0.9'),
    ('SERVE_DRAFT_POLICY=fc7:0.05\nNO_DRAFTER_POLICY=c7:0.3\n', 'confidence must be from 0.1 to 0.9'),
    ('SERVE_DRAFT_POLICY=fq7:0.3\nNO_DRAFTER_POLICY=fq7:0.3\n', 'must be c7:<c>'),
    (GOOD + 'SERVE_DRAFT_POLICY=fq7:0.4\n', 'sets SERVE_DRAFT_POLICY twice'),
    ('SERVE_DRAFT_POLICY=fq7:0.3\n', 'does not set NO_DRAFTER_POLICY'),
    (GOOD + 'NCCL_ALGO=Tree\n', 'SERVE_DRAFT_POLICY and NO_DRAFTER_POLICY only'),
    (GOOD + 'TF_GLM_NEW_KNOB=3\n', 'SERVE_DRAFT_POLICY and NO_DRAFTER_POLICY only'),
    (GOOD + 'TF_GLM_PREFILL_ROWS=2048\n', 'SERVE_DRAFT_POLICY and NO_DRAFTER_POLICY only'),
    (GOOD + 'TF_GLM_SESSION_DISK=0\n', 'SERVE_DRAFT_POLICY and NO_DRAFTER_POLICY only'),
    (GOOD + 'TF_GLM_DECODE_OBSERVE_DIR=capture\n', 'SERVE_DRAFT_POLICY and NO_DRAFTER_POLICY only'),
    (GOOD + 'TF_GLM_DECODE_OBSERVE_DIR=/tmp/x\n', 'is not KEY=VALUE'),
    ('SERVE_DRAFT_POLICY = fq7:0.3\nNO_DRAFTER_POLICY=c7:0.3\n', 'is not KEY=VALUE'),
    ('SERVE_DRAFT_POLICY=fq7:0.3;id\nNO_DRAFTER_POLICY=c7:0.3\n', 'is not KEY=VALUE'),
]
with tempfile.TemporaryDirectory() as tmp:
    tree = Path(tmp) / 'tree'
    for part in ('scripts', 'config', 'manifests', 'template'):
        shutil.copytree(HERE / part, tree / part, symlinks=True)
    for part in ('pins.env', 'cluster.env.example', 'SHA256SUMS'):
        if (HERE / part).exists():
            shutil.copy2(HERE / part, tree / part)
    profile = tree / 'config/profiles/base.env'
    for text, expect in REFUSED:
        profile.write_text(text)
        p = serve('0', tree=tree)
        first = next((x for x in text.splitlines() if x != 'SERVE_DRAFT_POLICY=fq7:0.3' and x != 'NO_DRAFTER_POLICY=c7:0.3'),
                     text.strip().splitlines()[-1])
        check(p.returncode == 2 and expect in p.stderr and 'docker run' not in p.stdout,
              f'changed profile refused before docker: {first!r} ({expect})')
    profile.write_text('# a changed profile\nSERVE_DRAFT_POLICY=fc7:0.9\nNO_DRAFTER_POLICY=c7:0.1\n')
    got = render('0', tree=tree)
    sha = hashlib.sha256(profile.read_bytes()).hexdigest()
    check(' --draft-policy fc7:0.9 ' in got['cmd'] and got['labels'].get('jspark3.profile') == sha
          and 'differs from the file this release shipped' in got['stderr'],
          'changed profile within the rules: renders fc7:0.9, labels its own sha256, warns')
    none = render('0', '--drafter', 'none', tree=tree)
    check(' --drafter none --draft-policy c7:0.1 ' in none['cmd'], 'changed profile: --drafter none renders c7:0.1')
    profile.write_text(GOOD)
    serve_env = tree / 'config/serve.env'
    serve_env.write_text(serve_env.read_text().replace('\nTF_GLM_FAIR_SCHED=1\n', '\nTF_GLM_FAIR_SCHED=0\n'))
    p = serve('0', tree=tree)
    check(p.returncode == 2 and 'needs TF_GLM_FAIR_SCHED=1' in p.stderr,
          'reply prefill without the fair scheduler is refused before docker')
print(f'check-serve-render: {len(failures)} failed')
sys.exit(1 if failures else 0)
