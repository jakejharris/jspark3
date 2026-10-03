// Change only ownership of independent value rows. The four-key sums, XOR
// reduction tree and chronological recurrence are identical to kda.cu.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

// Deliberately compile the original prep/output kernels in this separate,
// default-off extension. Keeping one source avoids copying their cast/norm
// semantics; the original extension and all flag-off source stay unchanged.
#include "kda.cu"

namespace {

template <int VALUES, int WARPS, int TR>
__global__ void __launch_bounds__(WARPS * 32) value_step(
        int H, const float* __restrict__ state_in, const float* __restrict__ q_in,
        const float* __restrict__ keys, const __nv_bfloat16* __restrict__ values,
        const float* __restrict__ decay, const float* __restrict__ betas, int rows,
        __nv_bfloat16* __restrict__ y_out, float* __restrict__ state_out) {
    constexpr int NT = WARPS * 32, VR = WARPS * VALUES;
    const int h = blockIdx.x, v0 = blockIdx.y * VR;
    const int lw = threadIdx.x >> 5, lane = threadIdx.x & 31;
    __shared__ __align__(16) float kt[TR][DK], qt[TR][DK], gt[TR][DK];
    __shared__ float vt[TR][VR], bt[TR];
    float s[VALUES][4];
    const size_t sbase = (size_t)h * DV * DK;
#pragma unroll
    for (int j = 0; j < VALUES; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i)
            s[j][i] = state_in[sbase + (size_t)(v0 + lw * VALUES + j) * DK + lane * 4 + i];
    for (int r0 = 0; r0 < rows; r0 += TR) {
        const int n = min(rows - r0, TR);
        __syncthreads();
        for (int x = threadIdx.x; x < n * DK; x += NT) {
            const int rr = x / DK, c = x % DK;
            const size_t at = ((size_t)(r0 + rr) * H + h) * DK + c;
            kt[rr][c] = keys[at];
            qt[rr][c] = q_in[at];
            gt[rr][c] = decay[at];
        }
        for (int x = threadIdx.x; x < n * VR; x += NT) {
            const int rr = x / VR, c = x % VR;
            vt[rr][c] = __bfloat162float(values[((size_t)(r0 + rr) * H + h) * DV + v0 + c]);
        }
        if (threadIdx.x < n) bt[threadIdx.x] = betas[(r0 + threadIdx.x) * H + h];
        __syncthreads();
        for (int rr = 0; rr < n; ++rr) {
            const float4 k4 = reinterpret_cast<const float4*>(kt[rr])[lane];
            const float4 q4 = reinterpret_cast<const float4*>(qt[rr])[lane];
            const float4 g4 = reinterpret_cast<const float4*>(gt[rr])[lane];
            const float kk[4] = {k4.x, k4.y, k4.z, k4.w};
            const float qq[4] = {q4.x, q4.y, q4.z, q4.w};
            const float gg[4] = {g4.x, g4.y, g4.z, g4.w};
#pragma unroll
            for (int j = 0; j < VALUES; ++j) {
                float kv = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    s[j][i] = s[j][i] * gg[i];
                    kv = kv + s[j][i] * kk[i];
                }
                kv = warp_sum(kv);
                const float delta = (vt[rr][lw * VALUES + j] - kv) * bt[rr];
#pragma unroll
                for (int i = 0; i < 4; ++i) s[j][i] = s[j][i] + kk[i] * delta;
                float o = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i) o = o + s[j][i] * qq[i];
                o = warp_sum(o);
                if (lane == 0)
                    y_out[((size_t)(r0 + rr) * H + h) * DV + v0 + lw * VALUES + j] = __float2bfloat16_rn(o);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < VALUES; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i)
            state_out[sbase + (size_t)(v0 + lw * VALUES + j) * DK + lane * 4 + i] = s[j][i];
}

template <int VALUES, int WARPS, int TR>
void launch_values(int H, const at::Tensor& state, const at::Tensor& q, const at::Tensor& k,
                   const at::Tensor& v, const at::Tensor& g, const at::Tensor& b, int rows,
                   at::Tensor& y, at::Tensor& next) {
    value_step<VALUES, WARPS, TR><<<dim3(H, DV / (VALUES * WARPS)), WARPS * 32, 0,
                                  at::cuda::getCurrentCUDAStream()>>>(
        H, ptr<float>(state), ptr<float>(q), ptr<float>(k), ptr<__nv_bfloat16>(v),
        ptr<float>(g), ptr<float>(b), rows, ptr<__nv_bfloat16>(y), ptr<float>(next));
}

int check_step(const at::Tensor& state, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
               const at::Tensor& g, const at::Tensor& b, int64_t rows, const at::Tensor& y,
               const at::Tensor& next, int64_t variant) {
    const at::Tensor* floats[] = {&state, &q, &k, &g, &b, &next};
    for (const auto* t : floats)
        TORCH_CHECK(t->is_cuda() && t->device() == state.device() && t->scalar_type() == at::kFloat &&
                    t->is_contiguous(), "contiguous FP32 tensors on one GPU required");
    for (const auto* t : {&v, &y})
        TORCH_CHECK(t->is_cuda() && t->device() == state.device() && t->scalar_type() == at::kBFloat16 &&
                    t->is_contiguous(), "contiguous BF16 tensors on one GPU required");
    TORCH_CHECK(state.dim() == 3 && state.size(1) == DV && state.size(2) == DK && next.sizes() == state.sizes(),
                "state must be [heads,128,128]");
    const int H = state.size(0);
    TORCH_CHECK(H > 0 && rows > 0 && q.numel() >= rows * H * DK && k.numel() >= rows * H * DK &&
                g.numel() >= rows * H * DK && v.numel() >= rows * H * DV && y.numel() >= rows * H * DV &&
                b.numel() >= rows * H, "scratch too small");
    TORCH_CHECK(variant >= 0 && variant <= 6, "unknown KDA value layout");
    return H;
}

void step(const at::Tensor& state, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
          const at::Tensor& g, const at::Tensor& b, int64_t rows, at::Tensor y, at::Tensor next, int64_t variant) {
    const int H = check_step(state, q, k, v, g, b, rows, y, next, variant);
    const c10::cuda::CUDAGuard guard(state.device());
    switch (variant) {
    case 0:  // unchanged recurrence for the isolated component benchmark
        step_kernel<4, 16><<<dim3(H, DV / 16), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
            H, ptr<float>(state), ptr<float>(q), ptr<float>(k), ptr<__nv_bfloat16>(v), ptr<float>(g), ptr<float>(b),
            rows, ptr<__nv_bfloat16>(y), ptr<float>(next));
        break;
    case 1: launch_values<2, 8, 8>(H, state, q, k, v, g, b, rows, y, next); break;
    case 2: launch_values<1, 8, 8>(H, state, q, k, v, g, b, rows, y, next); break;
    case 3: launch_values<2, 4, 8>(H, state, q, k, v, g, b, rows, y, next); break;
    case 4: launch_values<4, 4, 8>(H, state, q, k, v, g, b, rows, y, next); break;
    case 5: launch_values<4, 8, 8>(H, state, q, k, v, g, b, rows, y, next); break;
    case 6: launch_values<2, 8, 16>(H, state, q, k, v, g, b, rows, y, next); break;
    default: TORCH_CHECK(false, "unknown KDA value layout");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void chain_wide(const at::Tensor& P, int64_t p_stride, int64_t b_off, const at::Tensor& A, int64_t a_stride,
                const at::Tensor& G, int64_t g_stride, const at::Tensor& cs, const at::Tensor& cw,
                const at::Tensor& state, const at::Tensor& a_log, const at::Tensor& dt_bias,
                const at::Tensor& norm_w, double eps, double lower, int64_t rows, at::Tensor out,
                at::Tensor next, at::Tensor k, at::Tensor v, at::Tensor g, at::Tensor b,
                at::Tensor q, at::Tensor y, int64_t variant) {
    const c10::cuda::CUDAGuard guard(P.device());
    const int H = a_log.numel(), C = 3 * H * DK;
    for (const auto* t : {&P, &A, &G, &cs, &cw, &norm_w, static_cast<const at::Tensor*>(&out)})
        TORCH_CHECK(t->is_cuda() && t->device() == P.device() && t->scalar_type() == at::kBFloat16,
                    "BF16 inputs/output on one GPU required");
    for (const auto* t : {&a_log, &dt_bias})
        TORCH_CHECK(t->is_cuda() && t->device() == P.device() && t->scalar_type() == at::kFloat &&
                    t->is_contiguous(), "FP32 decay parameters on one GPU required");
    TORCH_CHECK(H > 0 && rows > 0 && P.dim() == 2 && P.size(0) >= rows && P.stride(0) == p_stride &&
                P.stride(1) == 1 && b_off >= C && P.size(1) >= b_off + H && A.dim() == 2 && G.dim() == 2 &&
                A.size(0) >= rows && G.size(0) >= rows && A.size(1) >= H * DK && G.size(1) >= H * DV &&
                A.stride(0) == a_stride && G.stride(0) == g_stride && A.stride(1) == 1 && G.stride(1) == 1,
                "invalid projection/gate rows or strides");
    TORCH_CHECK(cs.is_contiguous() && cs.numel() == 3 * C && cw.is_contiguous() && cw.numel() == C * 4 &&
                norm_w.is_contiguous() && norm_w.numel() == DV && dt_bias.numel() == H * DK &&
                out.is_contiguous() && out.numel() >= rows * H * DV, "invalid conv/norm/output shape");
    TORCH_CHECK(state.device() == P.device() && check_step(state, q, k, v, g, b, rows, y, next, variant) == H,
                "state/input head count or device differs");
    const dim3 grid(rows, H);
    auto stream = at::cuda::getCurrentCUDAStream();
    prep_kernel<<<grid, 512, 0, stream>>>(
        H, ptr<__nv_bfloat16>(P), p_stride, b_off, ptr<__nv_bfloat16>(A), a_stride, ptr<__nv_bfloat16>(cs),
        ptr<__nv_bfloat16>(cw), ptr<float>(a_log), ptr<float>(dt_bias), lower,
        ptr<float>(q), ptr<float>(k), ptr<__nv_bfloat16>(v), ptr<float>(g), ptr<float>(b));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    step(state, q, k, v, g, b, rows, y, next, variant);
    out_kernel<<<grid, DV, 0, stream>>>(H, ptr<__nv_bfloat16>(y), ptr<__nv_bfloat16>(G), g_stride,
                                      ptr<__nv_bfloat16>(norm_w), eps, ptr<__nv_bfloat16>(out));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("chain_wide", &chain_wide);
    m.def("step", &step);
}
