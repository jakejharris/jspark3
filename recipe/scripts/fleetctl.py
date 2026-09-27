#!/usr/bin/env python3
"""Serving controller with the sealed systemd zero-swap augmentation.

The native controller remains in _fleetctl.py. All native resource, receipt
and correctness gates still apply; this wrapper adds the systemd annotation
and verifies its effective state immediately after container start.
"""
import _fleetctl as native
import resource_scope

resource_scope.install(native)


def __getattr__(name):
    return getattr(native, name)


if __name__ == '__main__':
    raise SystemExit(native.main())
