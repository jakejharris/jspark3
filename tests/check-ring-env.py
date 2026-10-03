#!/usr/bin/env python3
"""Check scripts/fabric/launch.py against a synthetic sysfs tree laid out like the measured boxes.

python3 tests/check-ring-env.py   (CPU only, standard library; no RDMA, NCCL or Docker)
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / 'scripts/fabric/launch.py'
PREV, NEXT = 'enp1s0f0np0', 'enp1s0f1np1'
# As on the real boxes: every ConnectX-7 port is listed in PCI domain 0000 and again in 0002 at the same
# bus:device.function, and both copies carry an RDMA device.
BOX = {'0000:01:00.0': (PREV, ['rocep1s0f0']), '0000:01:00.1': (NEXT, ['rocep1s0f1']),
       '0002:01:00.0': (None, ['roceP2p1s0f0']), '0002:01:00.1': (None, ['roceP2p1s0f1']),
       '0000:00:00.0': (None, []), '0002:00:00.0': (None, []),  # root ports
       '0007:01:00.0': ('enP7s7', [])}  # the LAN NIC: same bus:device.function as a ConnectX function, no RDMA

# Expected environment for the synthetic PCI topology above, written out
# independently of the way launch.py computes it.
MEASURED = '''ARX_RING_NEXT_HCAS=rocep1s0f1,roceP2p1s0f1
ARX_RING_PREV_HCAS=rocep1s0f0,roceP2p1s0f0
NCCL_ALGO=Ring
NCCL_CROSS_NIC=1
NCCL_GRAPH_FILE=/tmp/nccl-ring-graph.{pid}.xml
NCCL_IB_ADDR_FAMILY=AF_INET
NCCL_IB_HCA==rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1
NCCL_IB_MERGE_NICS=0
NCCL_IB_ROCE_VERSION_NUM=2
NCCL_IB_SUBNET_AWARE_ROUTING=1
NCCL_NET_PLUGIN=none
VLLM_ARX_RING=1
'''
HEAD = ('<graphs version="1"><graph id="0" pattern="4" crossnic="1" nchannels="{n}" speedintra="0.24" speedinter="0.24"'
        ' latencyinter="0" typeintra="LOC" typeinter="P2C" samechannels="1">')
ROOT0 = '<channel><net dev="0"/><gpu dev="0"/><net dev="1"/></channel>'  # receive rocep1s0f0, send rocep1s0f1
ROOT2 = '<channel><net dev="2"/><gpu dev="0"/><net dev="3"/></channel>'  # receive roceP2p1s0f0, send roceP2p1s0f1
MEASURED_GRAPH = HEAD.format(n=8) + (ROOT0 + ROOT2) * 4 + '</graph></graphs>\n'
MEASURED_LOG = ('[fabric] NCCL_IB_HCA==rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1 '
                'NCCL_GRAPH_FILE=/tmp/nccl-ring-graph.{pid}.xml NCCL_ALGO=Ring NCCL_IB_SUBNET_AWARE_ROUTING=1 '
                'NCCL_NET_PLUGIN=none')
SET = {line.split('=', 1)[0] for line in MEASURED.splitlines()}
assert len(SET) == 12
CHILD = ('import json, os; print(json.dumps({"pid": os.getpid(), "env": dict(os.environ),'
         ' "graph": open(os.environ["NCCL_GRAPH_FILE"]).read()}))')


def make_sysfs(root: Path, box: dict) -> Path:
    """Build devices/, bus/pci/devices/, class/net/ and class/infiniband/ with symlinks as Linux lays them out."""
    sysfs = root / 'sys'
    for function in reversed(sorted(box)):  # creation order must not matter
        iface, rdma = box[function]
        domain = function.split(':')[0]
        real = sysfs / f'devices/pci{domain}:00' / f'{domain}:00:00.0' / function
        if function.endswith(':00:00.0'):
            real = sysfs / f'devices/pci{domain}:00' / function
        real.mkdir(parents=True, exist_ok=True)
        bus = sysfs / 'bus/pci/devices' / function
        bus.parent.mkdir(parents=True, exist_ok=True)
        bus.symlink_to(os.path.relpath(real, bus.parent))
        for name in rdma:
            (real / 'infiniband' / name).mkdir(parents=True)
            entry = sysfs / 'class/infiniband' / name
            entry.mkdir(parents=True)
            (entry / 'device').symlink_to(os.path.relpath(real, entry))
        if iface:
            entry = sysfs / 'class/net' / iface
            entry.mkdir(parents=True)
            (entry / 'device').symlink_to(os.path.relpath(real, entry))
    return sysfs


def run(sysfs, argv, prefix=(), **env):
    base = {k: v for k, v in os.environ.items() if not k.startswith(('NCCL_', 'RING_'))}
    environ = {**base, 'RING_SYSFS_ROOT': str(sysfs), 'RING_PREV_IFACE': PREV, 'RING_NEXT_IFACE': NEXT,
               'NCCL_MAX_NCHANNELS': '8', **env}
    environ = {k: v for k, v in environ.items() if v is not None}
    process = subprocess.Popen([*prefix, sys.executable, str(LAUNCH), *argv], env=environ, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate(timeout=30)
    return process.pid, process.returncode, out, err, environ


def fails(sysfs, why, expect, prefix=(), **env):
    """The launcher must stop before exec with a non-zero exit and a one-line message."""
    pid, code, out, err, _ = run(sysfs, ['sh', '-c', 'echo EXECUTED'], prefix, **env)
    assert code != 0 and 'EXECUTED' not in out and '[fabric]' not in out, (why, code, out, err)
    assert len(err.strip().splitlines()) == 1 and expect in err, (why, err)
    print(f'PASS {why}: exit {code}, {err.strip()}')
    return pid


def main():
    with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as tmp:
        t = Path(tmp)
        sysfs = make_sysfs(t / 'box', BOX)
        for label, channels in (('8', '8'), ('unset', None), ('empty', '')):
            pid, code, out, err, _ = run(sysfs, ['--print-env'], NCCL_MAX_NCHANNELS=channels)
            assert code == 0 and err == '', (code, err)
            assert out == MEASURED.format(pid=pid) + MEASURED_GRAPH, out
            assert not Path(f'/tmp/nccl-ring-graph.{pid}.xml').exists()
            print(f'PASS --print-env, NCCL_MAX_NCHANNELS {label}: the measured variables and graph bytes, nothing written')
        print(MEASURED_GRAPH, end='')

        # Exec path: the child keeps the PID, sees exactly the spec's changes, and reads the graph file.
        pid, code, out, err, parent = run(sysfs, [sys.executable, '-c', CHILD], NCCL_IB_GID_INDEX='3',
                                          RING_PASSTHROUGH='kept')
        graph_file = Path(f'/tmp/nccl-ring-graph.{pid}.xml')
        try:
            assert code == 0, err
            log, report = out.splitlines()
            assert log == MEASURED_LOG.format(pid=pid), log
            child = json.loads(report)
            assert child['pid'] == pid and child['graph'] == MEASURED_GRAPH
            assert graph_file.read_text() == MEASURED_GRAPH
            changed = {k for k in set(parent) | set(child['env']) if parent.get(k) != child['env'].get(k)}
            assert changed == SET | {'NCCL_IB_GID_INDEX'}, changed
            assert 'NCCL_IB_GID_INDEX' not in child['env'] and child['env']['RING_PASSTHROUGH'] == 'kept'
            assert all(f'{k}={child["env"][k]}' in MEASURED.format(pid=pid).splitlines() for k in SET)
        finally:
            graph_file.unlink(missing_ok=True)
        print('PASS exec: measured [fabric] line, PID kept, graph file bytes, the 12 variables set, NCCL_IB_GID_INDEX'
              ' removed, everything else passed through')
        print('in the container (PID 1): ' + MEASURED_LOG.format(pid=1))

        # A switch: both ports are the same, so each device appears once and each channel stays on one root.
        pid, code, out, err, _ = run(sysfs, ['--print-env'], RING_NEXT_IFACE=PREV)
        assert code == 0, err
        assert 'NCCL_IB_HCA==rocep1s0f0,roceP2p1s0f0\n' in out
        assert 'ARX_RING_PREV_HCAS=rocep1s0f0,roceP2p1s0f0\n' in out and 'ARX_RING_NEXT_HCAS=rocep1s0f0,roceP2p1s0f0\n' in out
        same = '<channel><net dev="{0}"/><gpu dev="0"/><net dev="{0}"/></channel>'
        assert out.endswith(HEAD.format(n=8) + (same.format(0) + same.format(1)) * 4 + '</graph></graphs>\n')
        pid, code, out, err, _ = run(sysfs, ['--print-env'], NCCL_MAX_NCHANNELS='2')
        assert code == 0 and out.endswith(HEAD.format(n=2) + ROOT0 + ROOT2 + '</graph></graphs>\n'), out
        print('PASS one port for both directions (switch) and NCCL_MAX_NCHANNELS=2')

        fails(sysfs, 'RING_PREV_IFACE unset', 'RING_PREV_IFACE is not set', RING_PREV_IFACE=None)
        fails(sysfs, 'RING_NEXT_IFACE empty', 'RING_NEXT_IFACE is not set', RING_NEXT_IFACE='')
        fails(sysfs, 'missing iface', 'enp9s0f0np0: no sysfs device', RING_NEXT_IFACE='enp9s0f0np0')
        (sysfs / 'class/net/lo').mkdir()  # a virtual netdev: no device link
        fails(sysfs, 'iface without a device', 'lo: no sysfs device', RING_PREV_IFACE='lo')
        one = make_sysfs(t / 'one', {**BOX, '0002:01:00.1': (None, [])})
        fails(one, '1 RDMA device on the next port', 'enp1s0f1np1: expected 2 RDMA devices')
        three = make_sysfs(t / 'three', {**BOX, '0004:01:00.0': (None, ['roceP4p1s0f0'])})
        fails(three, '3 RDMA devices on the prev port', 'enp1s0f0np0: expected 2 RDMA devices')
        fails(sysfs, 'bad NCCL_MAX_NCHANNELS', 'NCCL_MAX_NCHANNELS', NCCL_MAX_NCHANNELS='eight')

        # An unwritable graph path: bash puts a directory at its own PID's path, then execs the launcher (same PID).
        blocker = ['bash', '-c', 'mkdir /tmp/nccl-ring-graph.$$.xml && exec "$@"', 'bash']
        pid = None
        try:
            pid = fails(sysfs, 'unwritable graph path', 'cannot write the NCCL graph file /tmp/nccl-ring-graph.', blocker)
        finally:
            if pid is not None:
                os.rmdir(f'/tmp/nccl-ring-graph.{pid}.xml')
    print('PASS ring environment: synthetic sysfs, measured values; no RDMA, NCCL or Docker used')


if __name__ == '__main__':
    main()
