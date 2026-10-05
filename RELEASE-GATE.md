# JSpark3 v2.0.2 release gate

**Draft candidate. The packaging gate is separate from live validation and permission to publish.**
The release identity is in [manifests/release.json](manifests/release.json). Only the two image fixes
are included; the benchmark records remain v2.0.1 measurements.

Run on a clean checkout or archive, with the built wheel in `wheels/`:

```bash
python3 scripts/check-release.py --wheel wheels/tensorfold-0.3.6.2-py3-none-any.whl
bash tests/run.sh
```

The first command checks release identity, citations, candidate install references, session namespace,
engine source and wheel pins, wheel-to-source agreement, historical result identity and every shipped
file against `SHA256SUMS`. The existing offline suite also checks identity, rendered serving commands,
download isolation, space guards, output hygiene and payload integrity. Neither command contacts a Spark.

The image regressions additionally run with the engine test dependencies:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -q -p no:cacheprovider \
  engine/tests/test_glm_image_rotation_cache.py engine/tests/test_glm_session_batched.py \
  engine/tests/test_glm_session_disk.py engine/tests/test_glm_session_writer.py \
  engine/tests/test_glm_identical_cache.py engine/tests/test_glm_animated_gif.py \
  engine/tests/test_glm_vision_inputs.py engine/tests/test_glm_image_hardening.py \
  engine/tests/test_glm_image_lifetime.py
```

## Live blocker

The GIF fix passed its first live request. The cache root cause is VERIFIED offline, but the first live
swap at 17:37Z on 2026-10-05 failed its check and was rolled back. The eight-image request resumed 0 tokens;
nine and ten images were not checked. [ROOTCAUSE.md](release/v2.0.2/ROOTCAUSE.md) preserves the failed
attempt, the offline evidence and the limits of the log estimates. The PR stays draft until a live retry
passes. No clean-install serving or v2.0.2 performance proof is claimed.

## Replacing the cache fix

The cache and GIF changes are separate commits directly above v2.0.1. If the live owner provides an
incremental correction, cherry-pick it. If it replaces the cache change, revert only the cache commit
and cherry-pick its replacement. Keep the GIF commit. Do not rewrite a published branch.

Build the resulting engine with `scripts/build-wheel.sh` in the pinned image, using an isolated checkout
and empty wheel output. It intentionally rejects the old pin after leaving the newly built wheel for
inspection. Then one command updates the wheel pin, source binding, session identity and payload hashes:

```bash
python3 tools/repin-release.py wheels/tensorfold-0.3.6.2-py3-none-any.whl --engine-commit HEAD
```

Rerun the gates and engine regressions, record the live evidence, and commit the packaging changes.
The helper refuses a wheel whose package files differ from `engine/src`, and it never changes live status
to passed. Tagging, GitHub release assets, Hugging Face and site publication remain with the release owner.
