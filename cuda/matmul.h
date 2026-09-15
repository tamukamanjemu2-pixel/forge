#pragma once

void launch_matmul(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k
);

#include <cuda_runtime.h>
// Internal explicit-stream entry point; the original six-argument API remains.
void launch_matmul(const float* a, const float* b, float* c,
                   int m, int n, int k, cudaStream_t stream);
