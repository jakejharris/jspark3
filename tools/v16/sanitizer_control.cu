// SPDX-License-Identifier: Apache-2.0
// Maintenance-only detector control; never linked into the candidate bundle.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

__global__ void stock_control(int* p) { *p = 7; }
__global__ void exl3_moe_coop_detector_control(int* p, int index) {
    p[index] = 19;
}

int main(int argc, char** argv) {
    if (argc != 2 || (std::strcmp(argv[1], "clean") && std::strcmp(argv[1], "oob"))) return 2;
    const bool bad = !std::strcmp(argv[1], "oob");
    int* p = nullptr;
    if (cudaMalloc(&p, sizeof(int)) != cudaSuccess) return 3;
    stock_control<<<1, 1>>>(p);
    if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    // Exactly one int was allocated. Index 1 is deliberately out of bounds.
    exl3_moe_coop_detector_control<<<1, 1>>>(p, bad ? 1 : 0);
    const auto status = cudaDeviceSynchronize();
    if (!bad) {
        int value = 0;
        if (status != cudaSuccess || cudaMemcpy(&value, p, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess
            || value != 19 || cudaFree(p) != cudaSuccess) return 5;
    }
    // In the negative case return success even if the device context was killed:
    // only the sanitizer, not the application's exit status, can fail the step.
    std::printf("CONTROL_COMPLETE %s sync=%d\n", bad ? "oob" : "clean", int(status));
    return 0;
}
