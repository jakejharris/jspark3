// Isolated GLM-5.3-Flash TP3 specialization of the upstream two-stage MoE kernel
// (Turboderp exllamav3 58d4d732, MIT). Compile-time constants match this C ABI.
// K4 MCG is codebook cb=1 (mcg=1, mul1=2). Do not load the DS4.1 .so.
#include <cuda_runtime.h>
#include <cmath>
#define GLM53_COOP_NATIVE_ONLY 1
#define EXL3_MOE_COOP_DEFINE_ROT 1
#ifndef GLM53_ROWS_MAX
#error "Compile this translation unit with GLM53_ROWS_MAX=32 or 64"
#endif
#define COOP_JOIN_(a, b) a##b
#define COOP_JOIN(a, b) COOP_JOIN_(a, b)
#define GLM53_COOP_NAMESPACE COOP_JOIN(glm53_coop_ns_, GLM53_ROWS_MAX)
#define COOP_SYMBOL(name) COOP_JOIN(name, GLM53_ROWS_MAX)
#include "quant/glm53_coop_kernel.cuh"

namespace ns = GLM53_COOP_NAMESPACE;
constexpr int H = 4096, I = 2048, TOPK = 8, ROWS_MAX = GLM53_ROWS_MAX;
constexpr int SLOTS_MAX = TOPK * ROWS_MAX;
constexpr int CB_MCG = 1;
static bool prepared[3] = {};
// Both capacities use 9-bit slots so slot 256 cannot collide with expert 1.
static_assert(SLOTS_MAX == TOPK * ROWS_MAX, "slot capacity is rows * top-k");
static_assert(SLOTS_MAX <= 512, "run-builder rank key packs slot in 9 bits");
static_assert(sizeof(MoeCoopParams) == 344, "native parameter ABI size");
constexpr int CTR_LEN = SLOTS_MAX * (I / 128) + ROWS_MAX * (H / 128) + 2 + (SLOTS_MAX + 1) + SLOTS_MAX;
static_assert(CTR_LEN == (ROWS_MAX == 32 ? 5635 : 11267), "counter/run-table allocation");

struct Selected { void* a; void* b; int sa; int sb; bool wa; bool wb; };

static Selected select(int geometry) {
    // Geometry encoding matches upstream: 0 both-narrow, 1 both-wide, 2 A-wide
    // B-narrow (Blackwell auto for this shape). The adapter launches geometry 1:
    // auto B-narrow oversubscribes the down grid on H=4096 (see runtime.py).
    bool wa = geometry != 0, wb = geometry == 1;
    return {
        wa ? (void*) ns::exl3_moe_coop_a_kernel<4, CB_MCG, true>
           : (void*) ns::exl3_moe_coop_a_kernel<4, CB_MCG, false>,
        wb ? (void*) ns::exl3_moe_coop_b_kernel<4, CB_MCG, true>
           : (void*) ns::exl3_moe_coop_b_kernel<4, CB_MCG, false>,
        ns::smem_a_bytes<4>(H), ns::smem_b_bytes<4>(), wa, wb
    };
}

extern "C" int COOP_SYMBOL(glm53_coop_info_)(int bits, int geometry, int* info) {
    if (!info || bits != 4 || geometry < 0 || geometry > 2) return int(cudaErrorInvalidValue);
    int device;
    cudaError_t err = cudaGetDevice(&device); if (err != cudaSuccess) return int(err);
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device); if (err != cudaSuccess) return int(err);
    if (prop.major != 12 || prop.minor != 1) return int(cudaErrorInvalidDevice);
    Selected s = select(geometry);
    void* funcs[3] = {s.a, s.b, (void*) ns::exl3_moe_coop_rot_kernel};
    int smems[3] = {s.sa, s.sb, 0};
    for (int n = 0; n < 3; n++) {
        if (smems[n] > 48 * 1024) {
            err = cudaFuncSetAttribute(funcs[n], cudaFuncAttributeMaxDynamicSharedMemorySize, smems[n]);
            if (err != cudaSuccess) return int(err);
        }
        cudaFuncAttributes attr;
        err = cudaFuncGetAttributes(&attr, funcs[n]); if (err != cudaSuccess) return int(err);
        int blocks;
        err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, funcs[n], MOE_COOP_THREADS, smems[n]);
        if (err != cudaSuccess) return int(err);
        if (blocks < 1) return int(cudaErrorLaunchOutOfResources);
        info[n * 5 + 0] = MOE_COOP_THREADS; info[n * 5 + 1] = smems[n];
        info[n * 5 + 2] = attr.numRegs; info[n * 5 + 3] = int(attr.localSizeBytes); info[n * 5 + 4] = blocks;
    }
    info[15] = prop.multiProcessorCount;
    info[16] = int(exl3_moe_coop_ctr_len(SLOTS_MAX, ROWS_MAX, I, H));
    info[17] = int(sizeof(MoeCoopParams));
    prepared[geometry] = true;
    return 0;
}

// Pointer order: x, ids, weights; 9 gate/up/down tables (trellis,suh,svh);
// had_gate,had_up,gu_gate,gu_up,activation,down_partials,counters,output.
// Buffers have the compiled row/slot capacity, prepared before capture.
extern "C" int COOP_SYMBOL(glm53_coop_launch_)(void** t, int bits, int rows, int experts,
    float limit, int geometry, int gu_f32, void* stream_ptr) {
    if (!t || bits != 4 || rows < 1 || rows > ROWS_MAX || experts < 1 || experts > 96 ||
        geometry < 0 || geometry > 2 || gu_f32 != 0 || !std::isfinite(limit) || limit != 10.0f ||
        !prepared[geometry]) return int(cudaErrorInvalidValue);
    for (int n = 0; n < 20; n++) if (!t[n]) return int(cudaErrorInvalidValue);
    MoeCoopParams p = {};
    p.x = (half*) t[0]; p.sel = (int64_t*) t[1]; p.rw = (half*) t[2];
    p.g_trellis = (int64_t*) t[3]; p.g_suh = (int64_t*) t[4]; p.g_svh = (int64_t*) t[5];
    p.u_trellis = (int64_t*) t[6]; p.u_suh = (int64_t*) t[7]; p.u_svh = (int64_t*) t[8];
    p.d_trellis = (int64_t*) t[9]; p.d_suh = (int64_t*) t[10]; p.d_svh = (int64_t*) t[11];
    p.had_g = (half*) t[12]; p.had_u = (half*) t[13]; p.gu_g = t[14]; p.gu_u = t[15];
    p.act_out = (half*) t[16]; p.d_out = (float*) t[17]; p.ctr_a = (int*) t[18]; p.out = (float*) t[19];
    p.x_stride = H; p.bsz = rows; p.topk = TOPK; p.H = H; p.Hi = H; p.I = I; p.Ho = H; p.H_out = H;
    p.min_expert = 0; p.max_expert = experts; p.n_local = experts;
    p.act = MOE_COOP_ACT_SILU; p.act_limit = limit; p.gated = true;
    p.a_global = rows > 1; p.gu_f32 = bool(gu_f32); p.ksplit_a = p.ksplit_b = 1;
    p.slots_max = SLOTS_MAX; p.rows_max = ROWS_MAX; p.out_stride = H;
    p.ctr_a_len = SLOTS_MAX * (I / 128); p.ctr_b_len = ROWS_MAX * (H / 128);
    p.ctr_b = p.ctr_a + p.ctr_a_len; p.runs = p.ctr_b + p.ctr_b_len;
    Selected s = select(geometry);
    cudaStream_t stream = (cudaStream_t) stream_ptr;
    void* args[] = {&p};
    if (p.a_global) {
        int items = rows * TOPK * (H / 128) * 2;
        ns::exl3_moe_coop_rot_kernel<<<CEIL_DIVIDE(items, MOE_COOP_THREADS / 32), MOE_COOP_THREADS, 0, stream>>>(p);
        auto err = cudaGetLastError(); if (err != cudaSuccess) return int(err);
    }
    int ga = rows * TOPK * 2 * (I / (s.wa ? 128 : MOE_COOP_COLS));
    int gb = rows * TOPK * (H / (s.wb ? 128 : MOE_COOP_COLS));
    auto err = cudaLaunchKernel(s.a, dim3(ga), dim3(MOE_COOP_THREADS), args, s.sa, stream);
    if (err != cudaSuccess) return int(err);
    err = cudaLaunchKernel(s.b, dim3(gb), dim3(MOE_COOP_THREADS), args, s.sb, stream);
    if (err != cudaSuccess) return int(err);
    return int(cudaGetLastError());
}
