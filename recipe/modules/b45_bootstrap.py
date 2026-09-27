"""One import owner for the preserved B5 hooks and request-scoped B4 banks.

Copied into the candidate writable layer before its first start. No bind changes.
The read-only original KDA file remains present and unmodified for launcher checks;
only its Python import is redirected to the sealed original B4 implementation.
"""
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import sys


def install():
    if os.environ.get('B45_COMBINED') != '1':
        return
    import b5_prefix_verify as b5
    import b45_graphs as b4

    # b5 installs automatically. Replace only its own finder with the combined owner.
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, b5._Finder)]
    here = Path(__file__).resolve().parent
    target = 'vllm.model_executor.layers.quantization.kda_mixed_output_blocks'
    original = Path(b5._SITE) / (target.replace('.', '/') + '.py')
    shadow = here / 'kda_mixed_output_blocks.py'
    assert hashlib.sha256(original.read_bytes()).hexdigest() == '01aa249dd9ed35c96cc4339f85389d43a90085b9878a52827927974b93c58cd5'
    assert hashlib.sha256(shadow.read_bytes()).hexdigest() == '1628a3c1581c36f9d2dd1453b2169a0556c36938343cd0a0d0705c2f3e19160a'
    targets = set(b5.TARGETS) | set(b4.TARGETS)
    done = set()
    assert not (targets & set(sys.modules)), 'Combined hook installed after model imports'

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target_module=None):
            if name not in targets or name in done:
                return None
            done.add(name)
            spec = (importlib.util.spec_from_file_location(name, shadow) if name == target
                    else importlib.machinery.PathFinder.find_spec(name, path))
            if spec is None or spec.loader is None:
                raise RuntimeError('Combined import missing: ' + name)
            execute = spec.loader.exec_module

            def wrapped(module):
                execute(module)
                b5._gate_once()
                if name in b5.TARGETS:
                    b5.TARGETS[name](module)
                if name in b4.TARGETS:
                    b4._GATED = True
                    b4.TARGETS[name](module)
            spec.loader.exec_module = wrapped
            return spec

    sys.meta_path.insert(0, Finder())


install()
