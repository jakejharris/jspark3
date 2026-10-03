#!/usr/bin/env python3
"""Check this box before a step: tools, disk, the GPU, the RDMA cables, locked memory, ports and peers.

    python3 scripts/preflight.py [--for fetch|split|serve] [--weights base|ablit] [--drafter dflash2|none]
                                 [--session-tier disk|off]

--for serve (the default) checks everything serve.sh needs; fetch and split check less. Run it on every box.
Each line is PASS, WARN or FAIL with what to do. Output names settings, never their values (no addresses, paths or
user names), so it can be shared as is. Exit 0 when nothing failed, 1 otherwise. Read-only, except that the serve
check starts two short containers from the pinned image (never pulling it), which docker removes when they finish:
one reads the locked-memory limit, one lists the GPU the way serve.sh requests it (--gpus all).
"""
import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
GIB = 1 << 30
SETTINGS = ('DATA RANK MASTER_ADDR MASTER_PORT LAN_IFACE PREV_IFACE NEXT_IFACE API_HOST API_PORT WEIGHTS DRAFTER '
            'SESSION_TIER PREV_PEER NEXT_PEER IMAGE IMAGE_ID TEMPLATE_SHA256 SERVE_SESSION_GIB SERVE_SESSION_NAMESPACE '
            'ENGINE_WHEEL WHEEL_CONTENT_SHA256 CONTAINER_PREFIX HF_HUB_VERSION').split()
results = []


def report(state, what, todo=''):
    results.append(state)
    print(f'{state:4}  {what}' + (f': {todo}' if todo and state != 'PASS' else ''))


def sh(*cmd, timeout=60):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return 127, ''


def settings(argv):
    """cluster.env, the defaults and the command line, resolved by the same code the scripts use."""
    script = ('source "$1/scripts/lib.sh"; shift; parse_common "$@"; load_cluster >/dev/null; '
              + ''.join(f'printf "%s\\0" "${{{n}:-}}"; ' for n in SETTINGS))
    p = subprocess.run(['bash', '-c', script, 'preflight', str(HERE), *argv], capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stderr.strip(), file=sys.stderr)
        sys.exit(2)
    return dict(zip(SETTINGS, p.stdout.split('\0')))


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verified(folder, manifest):
    marker = Path(str(folder) + '.verified')
    return marker.is_file() and manifest.is_file() and marker.read_text().strip() == sha256(manifest)


def free_bytes(path):
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def check_common(s):
    version = '.'.join(str(n) for n in sys.version_info[:3])
    report('PASS' if sys.version_info >= (3, 10) else 'FAIL', 'python3 is 3.10 or newer',
           f'this python3 is {version}; install Python 3.10 or newer (INSTALL.md, "What you need")')
    rc, _ = sh('docker', 'info', '--format', '{{.ServerVersion}}')
    report('PASS' if rc == 0 else 'FAIL', 'docker runs for this user',
           'install docker and add your user to the docker group (INSTALL.md, "What you need")')
    data = Path(s['DATA'])
    ok = data.is_dir() and os.access(data, os.W_OK)
    report('PASS' if ok else 'FAIL', 'DATA exists and is writable', 'create it on the local NVMe disk and own it')


def check_fetch(s):
    install = (f'install it in a virtual environment: python3 -m venv ~/hf-cli && ~/hf-cli/bin/pip install '
               f'"huggingface_hub=={s["HF_HUB_VERSION"]}", then export PATH="$HOME/hf-cli/bin:$PATH" (INSTALL.md)')
    version = ''
    if shutil.which('hf'):
        version = sh(sys.executable, '-I', '-S', str(HERE / 'scripts/hf-version.py'), timeout=35)[1] or 'unknown'
    report('PASS' if version == s['HF_HUB_VERSION'] else 'FAIL',
           f"the Hugging Face CLI ('hf') is the pinned version {s['HF_HUB_VERSION']}",
           (f'it is {version}: ' if version else 'not installed: ') + install)
    if s['WEIGHTS'] == 'ablit':
        report('PASS' if os.environ.get('HF_TOKEN') else 'FAIL', 'HF_TOKEN is set for the gated ablit source',
               "accept the source's terms on its Hugging Face page, then export HF_TOKEN=<your token>")
    if os.environ.get('HF_HUB_DISABLE_XET', '1') != '1':
        report('WARN', 'HF_HUB_DISABLE_XET is not 1', 'downloads default to plain HTTPS; unset it unless you need Xet')


def check_split(s):
    rc, image_id = sh('docker', 'image', 'inspect', '--format', '{{.Id}}', s['IMAGE'])
    report('PASS' if rc == 0 and image_id == s['IMAGE_ID'] else 'FAIL', 'the pinned image is here, ID as pinned',
           'scripts/pull-image.sh')
    wheel = HERE / 'wheels' / s['ENGINE_WHEEL']
    ok = False
    if wheel.is_file() and s['WHEEL_CONTENT_SHA256'] != 'PENDING':
        rc, _ = sh(sys.executable, str(HERE / 'scripts/wheel-content.py'), '--expect', s['WHEEL_CONTENT_SHA256'], str(wheel))
        ok = rc == 0
    report('PASS' if ok else 'FAIL', 'the engine wheel is the pinned build', 'scripts/build-wheel.sh')
    lock_ok = all((HERE / 'wheels' / line.split()[0]).is_file()
                  for line in (HERE / 'wheels.lock').read_text().splitlines() if line.strip() and not line.startswith('#'))
    report('PASS' if lock_ok else 'FAIL', 'the dependency wheels are in wheels/', 'scripts/fetch-wheels.sh')


def check_third(s):
    w, rank = s['WEIGHTS'], s['RANK']
    third = Path(s['DATA']) / w / f'rank{rank}'
    report('PASS' if verified(third, HERE / 'manifests' / w / f'rank{rank}.sha256') else 'FAIL',
           f'the {w} rank {rank} third is split and verified', 'scripts/split.sh (INSTALL.md)')
    if s['DRAFTER'] == 'dflash2':
        report('PASS' if verified(Path(s['DATA']) / 'drafter', HERE / 'manifests/inputs/drafter.sha256') else 'FAIL',
               'the drafter is downloaded and verified', 'scripts/fetch-weights.sh, or --drafter none on every box')
    template = HERE / 'template/chat-template.jinja'
    report('PASS' if template.is_file() and sha256(template) == s['TEMPLATE_SHA256'] else 'FAIL',
           'the chat template matches pins.env', 'git checkout template/')


def check_serve(s):
    rc, gpus = sh('nvidia-smi', '-L')
    report('PASS' if rc == 0 and 'GPU 0' in gpus else 'FAIL', 'the GPU is visible (nvidia-smi)', 'install the NVIDIA driver')
    # The way serve.sh asks for the GPU (--gpus all), not a runtime name: stock DGX OS docker has only runc.
    rc, listed = sh('docker', 'run', '--rm', '--pull', 'never', '--network', 'none', '--gpus', 'all',
                    '--entrypoint', 'nvidia-smi', s['IMAGE'], '-L', timeout=120)
    report('PASS' if rc == 0 and 'GPU 0' in listed else 'FAIL', 'a container sees the GPU (docker run --gpus all)',
           'pull the image first (scripts/pull-image.sh); if it is here, install the NVIDIA Container Toolkit and '
           'restart docker')
    ib = Path('/sys/class/infiniband')
    ports = []
    if ib.is_dir():
        ports = [p for p in ib.glob('*/ports/*/state') if 'ACTIVE' in p.read_text()]
    report('PASS' if Path('/dev/infiniband').is_dir() else 'FAIL', 'RDMA devices exist (/dev/infiniband)',
           'install the ConnectX-7 (mlx5) drivers and RDMA core')
    report('PASS' if len(ports) >= 2 else 'FAIL', 'at least two RDMA ports are ACTIVE',
           'cable both ConnectX-7 ports in the ring (rank 0 -> 1 -> 2 -> 0) and bring the links up')
    for name in ('LAN_IFACE', 'PREV_IFACE', 'NEXT_IFACE'):
        net = Path('/sys/class/net') / s[name]
        if not net.is_dir():
            report('FAIL', f'{name} names an interface on this box', 'fix cluster.env (ip -br link lists them)')
            continue
        up = (net / 'operstate').read_text().strip() == 'up'
        report('PASS' if up else 'FAIL', f'{name} is up', 'bring the link up')
        if name != 'LAN_IFACE':
            mtu = int((net / 'mtu').read_text())
            report('PASS' if mtu == 9000 else 'WARN', f'{name} has MTU 9000', 'the measured ring ran at MTU 9000')
    for name, iface in (('PREV_PEER', 'PREV_IFACE'), ('NEXT_PEER', 'NEXT_IFACE')):
        if not s[name]:
            report('WARN', f'{name} is not set, so the cable to it is not tested', 'set it in cluster.env to check')
            continue
        rc, _ = sh('ping', '-c', '2', '-W', '2', '-M', 'do', '-s', '8972', '-I', s[iface], s[name], timeout=15)
        report('PASS' if rc == 0 else 'FAIL', f'{name} answers a 9000-byte ping over {iface}',
               'check the cable, the addresses and the MTU on both ends')
    # --entrypoint bash skips the image's banner; the last line is the limit either way.
    rc, limit = sh('docker', 'run', '--rm', '--pull', 'never', '--network', 'none', '--ulimit', 'memlock=-1',
                   '--entrypoint', 'bash', s['IMAGE'], '-c', 'ulimit -l', timeout=120)
    last = limit.splitlines()[-1].strip() if limit else ''
    report('PASS' if rc == 0 and last == 'unlimited' else 'FAIL', 'containers get unlimited locked memory',
           "allow docker's --ulimit memlock=-1 (default on a stock docker daemon)")
    with open('/proc/meminfo') as f:
        mem = {k: int(v.split()[0]) * 1024 for k, v in (line.split(':', 1) for line in f)}
    report('PASS' if mem.get('SwapTotal', 0) == 0 or mem.get('SwapFree', 0) == mem.get('SwapTotal', 0) else 'WARN',
           'swap is off or unused', 'optional: scripts/host-prep.sh matches the measured host state')
    rank = s['RANK']
    if rank == '0':
        for name, host, port in (('API_PORT', s['API_HOST'], s['API_PORT']), ('MASTER_PORT', '0.0.0.0', s['MASTER_PORT'])):
            sock = socket.socket()
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((host, int(port)))
                report('PASS', f'{name} is free on this box')
            except OSError:
                report('FAIL', f'{name} is free on this box', 'stop whatever listens there (an earlier server?)')
            finally:
                sock.close()
        report('PASS' if s['API_HOST'] in ('127.0.0.1', 'localhost', '::1') else 'WARN',
               'the API listens on loopback only',
               'the API has no authentication: expose it only through an authenticating proxy (INSTALL.md)')
    else:
        rc, _ = sh('ping', '-c', '2', '-W', '2', '-I', s['LAN_IFACE'], s['MASTER_ADDR'], timeout=15)
        report('PASS' if rc == 0 else 'WARN', 'MASTER_ADDR is reachable over the LAN',
               'rank 0 must be reachable at MASTER_ADDR from every box')
    name = f'{s["CONTAINER_PREFIX"]}-rank{rank}'
    rc, line = sh('docker', 'inspect', '--format', '{{.State.Status}} {{index .Config.Labels "jspark3.release"}}', name)
    if rc != 0:
        report('PASS', f'the container name {name} is free')
    elif len(line.split()) < 2:
        report('FAIL', f'the container name {name} is free',
               'a container this recipe did not start has that name; it is left alone. Set CONTAINER_PREFIX in '
               'cluster.env to another name, the same on all three boxes')
    else:
        report('FAIL', f'the container name {name} is free', 'a container from an earlier start has it: scripts/stop.sh')
    rc, others = sh('docker', 'ps', '--format', '{{.Names}}')
    others = [n for n in others.split() if n and n != name] if rc == 0 else []
    report('PASS' if not others else 'WARN', 'no other container is running on this box',
           f'{len(others)} running; stop any that use the GPU, its memory or the ports (another release, a benchmark)')
    rc, apps = sh('nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader')
    report('PASS' if rc == 0 and not apps.strip() else 'WARN', 'no other process is using the GPU',
           'stop it before serving: the server sizes itself from the memory that is free when it starts')
    try:
        proactive = int(Path('/proc/sys/vm/compaction_proactiveness').read_text())
    except (OSError, ValueError):
        proactive = None
    report('PASS' if proactive == 0 else 'WARN', 'vm.compaction_proactiveness is 0',
           'background memory compaction can stall serving after a build or split on the same box. Run: '
           'sudo sysctl -w vm.compaction_proactiveness=0 (or scripts/host-prep.sh); a reboot resets it')
    if s['SESSION_TIER'] == 'disk':
        need = (150 + int(s['SERVE_SESSION_GIB'] or 0)) * GIB
        report('PASS' if free_bytes(s['DATA']) >= need else 'WARN',
               f'DATA has room for the session tier ({s["SERVE_SESSION_GIB"]} GiB, with 150 GiB left free)',
               'free space on DATA, or serve with --session-tier off; otherwise conversations are not kept on disk')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split('\n', 2)[2])
    ap.add_argument('--for', dest='step', choices=('fetch', 'split', 'serve'), default='serve')
    a, rest = ap.parse_known_args()
    s = settings(rest)
    check_common(s)
    if a.step == 'fetch':
        check_fetch(s)
    if a.step in ('split', 'serve'):
        check_split(s)
    if a.step == 'serve':
        check_third(s)
        check_serve(s)
    failed = results.count('FAIL')
    print(f'preflight --for {a.step}: {results.count("PASS")} passed, {results.count("WARN")} warnings, {failed} failed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
