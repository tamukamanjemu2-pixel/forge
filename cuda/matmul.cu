#include "matmul.h"

#include <cuda_runtime.h>
#include "../src/cuda_support.h"
#include <cstddef>

namespace {

    __global__ void matmul_kernel(
        const float* a,
        const float* b,
        float* c,
        int m,
        int n,
        int k
    ) {
        const std::size_t row =
            blockIdx.y * blockDim.y + threadIdx.y;

        const std::size_t col =
            blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n) {
            return;
        }

        float accumulator = 0.0f;

        for (int i = 0; i < k; ++i) {
            accumulator +=
                a[row * k + i] *
                b[static_cast<std::size_t>(i) * n + col];
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
    int k,
    cudaStream_t stream
) {
    if (!a || !b || !c || m <= 0 || n <= 0 || k <= 0)
        throw std::invalid_argument("Invalid MatMul launch arguments");
    constexpr int tile = 16;

    const dim3 block(tile, tile);

    const dim3 grid(
        n / tile + (n % tile != 0),
        m / tile + (m % tile != 0)
    );

    int device = 0, max_x = 0, max_y = 0;
    forge::detail::cuda_check(cudaGetDevice(&device), "MatMul cudaGetDevice");
    forge::detail::cuda_check(cudaDeviceGetAttribute(&max_x, cudaDevAttrMaxGridDimX, device), "MatMul grid X limit");
    forge::detail::cuda_check(cudaDeviceGetAttribute(&max_y, cudaDevAttrMaxGridDimY, device), "MatMul grid Y limit");
    if (grid.x > static_cast<unsigned>(max_x) || grid.y > static_cast<unsigned>(max_y))
        throw std::overflow_error("MatMul dimensions exceed device grid limits");

    matmul_kernel<<<grid, block, 0, stream>>>(
        a,
        b,
        c,
        m,
        n,
        k
    );
    forge::detail::cuda_check(cudaGetLastError(), "MatMul kernel launch");
}

void launch_matmul(const float* a, const float* b, float* c, int m, int n, int k) {
    launch_matmul(a, b, c, m, n, k, nullptr);
}
