"""Controller-only correction: persist the existing zero-swap contract in systemd.

Final launch admission requires the release qualification protocol.
The native resource gate is retained.
"""
ANNOTATION = 'org.systemd.property.MemorySwapMax'
VALUE = 'uint64 0'

SCOPE_READ = r'''
import json,pathlib,subprocess,sys
cid=sys.argv[1]
doc=json.loads(subprocess.check_output(['docker','container','inspect',cid],text=True))[0]
if doc['Id']!=cid or not doc['State']['Running']:raise SystemExit('bound running identity changed')
rows=(pathlib.Path('/proc')/str(doc['State']['Pid'])/'cgroup').read_text().splitlines()
matches=[line.split('::',1)[1] for line in rows if line.startswith('0::')]
if len(matches)!=1:raise SystemExit('ambiguous cgroup v2 path')
root=pathlib.Path('/sys/fs/cgroup').resolve()
cg=(root/matches[0].lstrip('/')).resolve(strict=True)
if root not in cg.parents or cg.name!='docker-'+cid+'.scope':raise SystemExit('unexpected systemd container scope')
prop=subprocess.check_output(['systemctl','show','--value','--property=MemorySwapMax',cg.name],text=True,timeout=15).strip()
print(json.dumps(dict(container_id=cid,pid=doc['State']['Pid'],started_at=doc['State']['StartedAt'],cgroup_path=str(cg),
    systemd_memory_swap_max=prop,memory_max=(cg/'memory.max').read_text().strip(),
    swap_max=(cg/'memory.swap.max').read_text().strip(),swap_current=int((cg/'memory.swap.current').read_text()),
    events=dict(line.split() for line in (cg/'memory.events').read_text().splitlines()))))
'''


def add_annotation(argv):
    if argv[:2] != ['docker', 'create'] or '--annotation' in argv:
        raise ValueError('unexpected native create command')
    for key in ('--memory', '--memory-swap'):
        if argv.count(key) != 1 or argv[argv.index(key) + 1] != '68719476736':
            raise ValueError('native 64 GiB / zero-swap request changed')
    return argv[:2] + ['--annotation', ANNOTATION + '=' + VALUE] + argv[2:]


def check_scope(doc, identity):
    if (doc.get('container_id') != identity or doc.get('systemd_memory_swap_max') != '0'
            or doc.get('memory_max') != '68719476736' or doc.get('swap_max') != '0'
            or doc.get('swap_current') != 0
            or any(int(doc.get('events', {}).get(k, -1)) != 0
                   for k in ('oom', 'oom_kill', 'oom_group_kill'))):
        raise ValueError('systemd/kernel zero-swap contract not established: ' + repr(doc))


def install(fleet):
    """Augment native create/contract/start, preserving its exact failure cleanup."""
    import json
    original_argv = fleet.container_argv
    original_contract = fleet.validate_container_contract
    original_remote = fleet.remote

    def container_argv(*args, **kwargs):
        return add_annotation(original_argv(*args, **kwargs))

    def contract(values, rank, identity, item, preflight_sha, recipe_sha):
        original_contract(values, rank, identity, item, preflight_sha, recipe_sha)
        if (item.get('HostConfig') or {}).get('Annotations') != {ANNOTATION: VALUE}:
            raise fleet.Refusal(f'rank{rank} persistent zero-swap annotation missing/changed')

    def remote(values, rank, argv, *, check=True):
        result = original_remote(values, rank, argv, check=check)
        if argv[:2] == ['docker', 'start'] and result.returncode == 0:
            if len(argv) != 3 or not fleet.SHA_RE.fullmatch(argv[2]):
                raise fleet.Refusal('zero-swap check requires one exact container ID')
            observed = original_remote(values, rank, ['python3', '-B', '-S', '-c', SCOPE_READ, argv[2]])
            doc = json.loads(observed.stdout)
            check_scope(doc, argv[2])
            # This host/controller observation is evidence, not a substitute for
            # any subsequent native verify or finalizer gate.
            fleet.diagnostics.retain(observed.stdout)
            print('PERSISTENT-ZERO-SWAP ' + json.dumps(dict(rank=rank, container_id=argv[2],
                  status='PASS', systemd_memory_swap_max=0, swap_max=0, swap_current=0), sort_keys=True), flush=True)
        return result

    fleet.container_argv = container_argv
    fleet.validate_container_contract = contract
    fleet.remote = remote
