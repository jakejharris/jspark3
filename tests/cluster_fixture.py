"""Point a check's scripts at its own cluster.env, and refuse to run them anywhere else.

The scripts read the cluster.env that CLUSTER_ENV names, so a value exported for a real install would send a check's
writes and deletes into that install. select() replaces it with the copied tree's file, resolves DATA with the
scripts' own code, and exits before any script runs unless DATA and the copied tree are both inside the folder the
check made for itself (symlinks resolved).
"""
import os
import subprocess
import sys

RESOLVE = 'source "$1/scripts/lib.sh" && load_cluster >/dev/null && printf "%s" "$DATA"'


def select(env, tree, root):
    """Set env's CLUSTER_ENV to tree/cluster.env; exit unless DATA and tree resolve inside root."""
    env['CLUSTER_ENV'] = str(tree / 'cluster.env')
    p = subprocess.run(['bash', '-c', RESOLVE, 'cluster-fixture', str(tree)], env=env, capture_output=True, text=True)
    top = os.path.realpath(root)
    for name, path in (('DATA', p.stdout if p.returncode == 0 else ''), ('the copied tree', str(tree))):
        real = os.path.realpath(path) if path.startswith('/') else ''
        if not real or os.path.commonpath([top, real]) != top:
            sys.exit(f"refusing to run: {name} is not inside this check's temporary folder")
    return env
