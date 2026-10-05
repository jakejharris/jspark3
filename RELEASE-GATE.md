# JSpark3 v2.0.2 release gate

**Candidate ready for review. The live retry passed; publication still requires release-owner approval.**
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

## Live acceptance

The unchanged engine passed the retry at 18:37Z on 2026-10-05: 8/9/10-image resumes `0 -> 68 -> 174`,
both nonzero hits from disk, GIF pass and smoke 6/6. Boundary and full-prompt checkpoints were durable
on all three ranks. The gate requires a checksummed [live receipt](release/v2.0.2/live-retry.json) bound
to the same wheel content and engine fix commits before accepting `ready-for-review` status.

[ROOTCAUSE.md](release/v2.0.2/ROOTCAUSE.md) preserves the first attempt's fixture failure, rollback,
corrected fixture and log estimates. The acceptance runner provided idle persistence opportunities;
the bounded writer can still skip optional saves under continuous traffic. This proves the small CUDA
sequence on the prepared runtime, not lossless persistence under load, a long-context latency benchmark
or a clean installation of the final public recipe. No new speed result is claimed.

## Replacing the cache fix

The cache and GIF changes are separate commits directly above v2.0.1. No replacement was needed for the
passing retry. For a future engine change, first return the PR and metadata to draft, set
`live_validation.retry` to `pending`, and remove its prior receipt binding. New engine bytes require new
live acceptance; `tools/repin-release.py` deliberately refuses a candidate ready for review.

If the live owner provides an
incremental correction, cherry-pick it. If it replaces the cache change, revert only the cache commit
and cherry-pick its replacement. Keep the GIF commit. Do not rewrite a published branch.
Update `manifests/release.json`'s `fixes` entries to name the replacement or incremental source and applied
commits, and update the evidence narrative. The repin command below preserves that lineage record.

Build the resulting engine with `scripts/build-wheel.sh` in the pinned image, using an isolated checkout
and empty wheel output. It intentionally rejects the old pin after leaving the newly built wheel for
inspection. Then one command updates the wheel pin, source binding, session identity and payload hashes:

```bash
python3 tools/repin-release.py wheels/tensorfold-0.3.6.2-py3-none-any.whl --engine-commit HEAD
```

Rerun the gates and engine regressions, record the live evidence, and commit the packaging changes.
The helper refuses a wheel whose package files differ from `engine/src`, and it never changes live status
to passed. Tagging, GitHub release assets, Hugging Face and site publication remain with the release owner.
