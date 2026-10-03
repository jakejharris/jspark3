#!/usr/bin/env python3
"""Point NCCL at the cabled TP=3 ring, then replace this process with the server.

    python3 /bundle/fabric/launch.py tensorfold serve ...   (from serve.sh, in the container)
    python3 launch.py --print-env                           (print what would be set; run nothing)

Topology. The three Sparks are wired straight to each other, no switch: rank r's RING_NEXT_IFACE cable lands on rank
r+1's RING_PREV_IFACE, closing the loop 0 -> 1 -> 2 -> 0. Each cable carries a private two-host subnet, so a given
port can talk to one neighbour and nobody else. GB10 exposes every ConnectX-7 port twice, as the same
bus:device.function under PCI domain 0000 and under domain 0002 (one per PCIe root complex), with an RDMA device on
each copy. That makes two RDMA devices per port and four per box.

Left alone, NCCL would treat any NIC as a route to any rank. What the launcher sets instead:
- a graph file whose channels take data in through a prev-port device and push it out through a next-port device,
  switching PCIe root on every other channel;
- NCCL_ALGO=Ring, so traffic only flows between adjacent ranks;
- NCCL_IB_SUBNET_AWARE_ROUTING=1. A rank with a single GPU uses the graph's first NIC for sending and receiving
  alike, so the IB transport has to redirect each send to whichever port shares a subnet with the destination.

Graph constants. The 0.24 speeds, the LOC/P2C path types, pattern 4, crossnic 1 and samechannels 1 are copied from
what NCCL 2.30.7's own topology search settled on in the measured boots. NCCL bases its protocol and chunk-size
choices on whatever graph it is handed. In the served configuration NCCL only carries data between ranks: the default
TF_TP3_REDUCE=gather adds the partials in TensorFold's own kernels, in rank order, and NCCL's all-reduce is used only
by the TF_TP3_REDUCE=nccl probe. So the constants are kept to reproduce the measured transport exactly; they do not
alter any arithmetic.

Inputs: RING_PREV_IFACE, RING_NEXT_IFACE (required), NCCL_MAX_NCHANNELS (default 8) and, for tests, RING_SYSFS_ROOT
(default /sys). Standard library only.
"""
import os
import sys
from pathlib import Path

DEFAULT_CHANNELS = 8
GRAPH = ('<graphs version="1"><graph id="0" pattern="4" crossnic="1" nchannels="{n}" speedintra="0.24"'
         ' speedinter="0.24" latencyinter="0" typeintra="LOC" typeinter="P2C" samechannels="1">{channels}'
         '</graph></graphs>\n')
CHANNEL = '<channel><net dev="{recv}"/><gpu dev="0"/><net dev="{send}"/></channel>'
LOGGED = ('NCCL_IB_HCA', 'NCCL_GRAPH_FILE', 'NCCL_ALGO', 'NCCL_IB_SUBNET_AWARE_ROUTING', 'NCCL_NET_PLUGIN')
REMOVED = 'NCCL_IB_GID_INDEX'  # NCCL_IB_ADDR_FAMILY picks each device's GID by IPv4; no single index suits all four


class RingError(Exception):
    """The ring cannot be set up; the launcher stops before exec."""


def port_rdma_devices(sysfs: Path, iface: str) -> list[str]:
    """Both RDMA devices behind one ConnectX-7 port, ordered by PCI address and then name."""
    link = sysfs / 'class/net' / iface / 'device'
    if not link.exists():
        raise RingError(f'{iface}: no sysfs device at {link}')
    function = Path(os.path.realpath(link)).name  # e.g. 0000:01:00.0
    _, sep, bdf = function.partition(':')
    if not sep:
        raise RingError(f'{iface}: device {function!r} is not a PCI address')
    pairs = []
    for twin in (sysfs / 'bus/pci/devices').iterdir():  # the same bus:device.function in every PCI domain
        if twin.name.partition(':')[2] == bdf and (twin / 'infiniband').is_dir():
            pairs += [(twin.name, name) for name in os.listdir(twin / 'infiniband')]
    names = [name for _, name in sorted(pairs)]
    if len(names) != 2:
        raise RingError(f'{iface}: expected 2 RDMA devices (one per PCIe root), found {len(names)}: {names}')
    return names


def pci_path(sysfs: Path, name: str) -> str:
    link = sysfs / 'class/infiniband' / name / 'device'
    if not link.exists():
        raise RingError(f'{name}: no sysfs device at {link}')
    return os.path.realpath(link)


def channel_count() -> int:
    value = os.environ.get('NCCL_MAX_NCHANNELS') or str(DEFAULT_CHANNELS)
    if not value.isdigit() or int(value) < 1:
        raise RingError(f'NCCL_MAX_NCHANNELS={value!r} is not a positive integer')
    return int(value)


def ring_setup(pid: int) -> tuple[dict[str, str], str, str]:
    """Return (variables to set, graph path, graph contents) for this box, without touching anything."""
    ifaces = {}
    for key in ('RING_PREV_IFACE', 'RING_NEXT_IFACE'):
        ifaces[key] = os.environ.get(key, '')
        if not ifaces[key]:
            raise RingError(f'{key} is not set')
    sysfs = Path(os.environ.get('RING_SYSFS_ROOT') or '/sys')
    prev = port_rdma_devices(sysfs, ifaces['RING_PREV_IFACE'])
    nxt = port_rdma_devices(sysfs, ifaces['RING_NEXT_IFACE'])
    # A graph <net dev=N> uses NCCL's own numbering of the devices it keeps, which follows its enumeration order. On
    # GB10 that is PCI order: every measured boot logs 'NET/IB : Using [0]rocep1s0f0 [1]rocep1s0f1 [2]roceP2p1s0f0
    # [3]roceP2p1s0f1'. Listing the HCAs once each, by PCI path, keeps this list in that same order.
    devs = sorted(set(prev + nxt), key=lambda name: (pci_path(sysfs, name), name))
    index = {name: i for i, name in enumerate(devs)}
    n = channel_count()
    channels = ''.join(CHANNEL.format(recv=index[prev[c % 2]], send=index[nxt[c % 2]]) for c in range(n))
    path = f'/tmp/nccl-ring-graph.{pid}.xml'
    variables = {
        'NCCL_IB_HCA': '=' + ','.join(devs),  # leading '=': whole device names, no prefix matching
        'NCCL_IB_MERGE_NICS': '0',
        'NCCL_IB_SUBNET_AWARE_ROUTING': '1',
        'NCCL_NET_PLUGIN': 'none',
        'NCCL_ALGO': 'Ring',
        'NCCL_CROSS_NIC': '1',
        'NCCL_GRAPH_FILE': path,
        'NCCL_IB_ADDR_FAMILY': 'AF_INET',
        'NCCL_IB_ROCE_VERSION_NUM': '2',
        # No reader in TensorFold's engine source or its locked wheels. Kept only so each rank's environment is the
        # measured boot's, byte for byte, until a search of every library the ranks map shows nothing reads them.
        'VLLM_ARX_RING': '1',
        'ARX_RING_PREV_HCAS': ','.join(prev),
        'ARX_RING_NEXT_HCAS': ','.join(nxt),
    }
    return variables, path, GRAPH.format(n=n, channels=channels)


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        sys.exit('usage: launch.py COMMAND [ARGS...] | launch.py --print-env')
    try:
        variables, path, graph = ring_setup(os.getpid())  # the server inherits this PID through exec
        if argv[1] == '--print-env':
            sys.stdout.write(''.join(f'{k}={v}\n' for k, v in sorted(variables.items())) + graph)
            return
        try:
            Path(path).write_text(graph)
        except OSError as error:
            raise RingError(f'cannot write the NCCL graph file {path}: {error}') from None
    except RingError as error:
        sys.exit(f'launch.py: {error}')
    os.environ.update(variables)
    os.environ.pop(REMOVED, None)
    print('[fabric] ' + ' '.join(f'{k}={variables[k]}' for k in LOGGED), flush=True)
    os.execvp(argv[1], argv[1:])


if __name__ == '__main__':
    main(sys.argv)
