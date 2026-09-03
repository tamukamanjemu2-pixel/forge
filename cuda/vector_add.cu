#include "vector_add.h"

#include <cuda_runtime.h>

__global__ void vector_add_kernel(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

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
    const int blocks =
        (n + threads_per_block - 1) / threads_per_block;

    vector_add_kernel<<<blocks, threads_per_block>>>(
        d_a,
        d_b,
        d_c,
        n
    );
}