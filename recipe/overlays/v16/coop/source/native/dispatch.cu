// ABI 2 selects a physical capacity before any CUDA launch. No retry path.
#include <cuda_runtime.h>
#define DECLARE(N) \
extern "C" int glm53_coop_info_##N(int, int, int*); \
extern "C" int glm53_coop_launch_##N(void**, int, int, int, float, int, int, void*);
DECLARE(32)
DECLARE(64)
extern "C" int glm53_coop_abi() { return 2; }
extern "C" int glm53_coop_info(int bits, int geometry, int capacity, int* info) {
    if (capacity == 32) return glm53_coop_info_32(bits, geometry, info);
    if (capacity == 64) return glm53_coop_info_64(bits, geometry, info);
    return int(cudaErrorInvalidValue);
}
extern "C" int glm53_coop_launch(void** t, int bits, int rows, int experts,
    float limit, int geometry, int gu_f32, void* stream) {
    if (rows < 1 || rows > 64) return int(cudaErrorInvalidValue);
    return rows <= 32
        ? glm53_coop_launch_32(t, bits, rows, experts, limit, geometry, gu_f32, stream)
        : glm53_coop_launch_64(t, bits, rows, experts, limit, geometry, gu_f32, stream);
}
