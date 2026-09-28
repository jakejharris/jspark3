#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Private compiler stage. Call from a fresh /w mount; never execute the probe.
set -euo pipefail
test "$PWD" = /w
export LC_ALL=C TZ=UTC SOURCE_DATE_EPOCH=1789774362
mkdir -p keep/probe
gcc -O2 -fPIC -shared -Wall -I/usr/local/cuda/include \
    -o libglm53_display_kv.so display_kv.c -L/usr/local/cuda/lib64/stubs -lcuda
# --keep gives the host stub a stable basename instead of a PID-derived tmpxft
# STT_FILE symbol. Distinct seeds cover both translation units; no ELF rewriting.
nvcc=/usr/local/cuda/bin/nvcc
"$nvcc" -O2 -arch=sm_121 --frandom-seed=jspark3-v14-display-probe \
    --keep --keep-dir /w/keep/probe -c probe_main.cu -o probe_main.o
"$nvcc" -O2 -arch=sm_121 --frandom-seed=jspark3-v14-display-host \
    -c display_kv.c -o display_kv.o
"$nvcc" -O2 -arch=sm_121 --link -o display_kv_probe probe_main.o display_kv.o -lcuda
{ gcc --version | head -1; "$nvcc" --version | tail -1; } > toolchain.txt
