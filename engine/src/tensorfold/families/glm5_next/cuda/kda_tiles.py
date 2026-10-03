"""KDA value ownership experiment; no changed recurrence or persistent storage."""

from functools import lru_cache
from pathlib import Path


# GB10 screen winner: 4 values/warp, 8 warps/CTA, 8 staged rows.
# Alternate native layouts remain available to the reproducible benchmark only.
VARIANT = 5


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    return load(name="tensorfold_glm_kda_tiles_v1",
                sources=[str(Path(__file__).with_suffix(".cu"))],
                extra_cuda_cflags=["-O3", "--fmad=false"], verbose=False)


def chain_wide(*args):
    _ext().chain_wide(*args, VARIANT)
