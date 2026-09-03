#include "matmul.h"

#include <cuda_runtime.h>

namespace {

    __global__ void matmul_kernel(
        const float* a,
        const float* b,
        float* c,
        int m,
        int n,
        int k
    ) {
        const int row =
            blockIdx.y * blockDim.y + threadIdx.y;

        const int col =
            blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n) {
            return;
        }

        float accumulator = 0.0f;

        for (int i = 0; i < k; ++i) {
            accumulator +=
                a[row * k + i] *
                b[i * n + col];
        }

        c[row * n + col] = accumulator;
    }

} // namespace

void launch_matmul(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k
) {
    constexpr int tile = 16;

    const dim3 block(tile, tile);

    const dim3 grid(
        (n + tile - 1) / tile,
        (m + tile - 1) / tile
    );

    matmul_kernel<<<grid, block>>>(
        a,
        b,
        c,
        m,
        n,
        k
    );
}