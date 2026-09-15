#include "vector_add.h"

#include <cuda_runtime.h>
#include "../src/cuda_support.h"
#include <cstddef>

__global__ void vector_add_kernel(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void launch_vector_add(
    const float* d_a,
    const float* d_b,
    float* d_c,
    int n,
    int threads_per_block
) {
    if (!d_a || !d_b || !d_c || n <= 0 || threads_per_block <= 0 || threads_per_block > 1024)
        throw std::invalid_argument("Invalid vector-add launch arguments");
    const int blocks =
        n / threads_per_block + (n % threads_per_block != 0);

    vector_add_kernel<<<blocks, threads_per_block>>>(
        d_a,
        d_b,
        d_c,
        n
    );
    FORGE_CUDA_CHECK(cudaGetLastError());
}
