# Serving recipe

See [installation](../docs/INSTALL.md) and [operations](../docs/OPERATIONS.md).
Prepare a separate runtime copy with locally rebuilt, hash-verified native
artifacts. The example selects stock mode with the production-stock profile.
Its policy is tested offline; production admission remains closed until the
exact shipped configuration passes fresh hardware qualification. Historical
stock measurements under the validation profile do not qualify production admission.

The lifecycle controller is `scripts/fleetctl.py`. Its `--dry-run` option
renders commands without contacting hosts. Runtime transforms refuse unknown
source bytes and verify exact before/after contracts.
