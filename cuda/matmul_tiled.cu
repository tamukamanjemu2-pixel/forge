#include "matmul.h"
#include "../src/cuda_support.h"
#include <cstddef>

namespace {
constexpr unsigned tile = 16;
__global__ void tiled_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float as[tile][tile];
    __shared__ float bs[tile][tile];
    const unsigned tx = threadIdx.x, ty = threadIdx.y;
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * tile + ty;
    const std::size_t col = static_cast<std::size_t>(blockIdx.x) * tile + tx;
    float accumulator = 0.0f;
    for (std::size_t base = 0; base < static_cast<std::size_t>(k); base += tile) {
        as[ty][tx] = row < static_cast<std::size_t>(m) && base + tx < static_cast<std::size_t>(k)
            ? a[row * k + base + tx] : 0.0f;
        bs[ty][tx] = col < static_cast<std::size_t>(n) && base + ty < static_cast<std::size_t>(k)
            ? b[(base + ty) * n + col] : 0.0f;
        __syncthreads();
        for (unsigned i = 0; i < tile; ++i) accumulator += as[ty][i] * bs[i][tx];
        __syncthreads();
    }
    if (row < static_cast<std::size_t>(m) && col < static_cast<std::size_t>(n)) c[row * n + col] = accumulator;
}
}

void launch_matmul_tiled(const float* a, const float* b, float* c, int m, int n, int k, cudaStream_t stream) {
    if (!a || !b || !c || m <= 0 || n <= 0 || k <= 0)
        throw std::invalid_argument("Invalid tiled MatMul launch arguments");
    const dim3 block(tile, tile);
    const dim3 grid(n / tile + (n % tile != 0), m / tile + (m % tile != 0));
    int device = 0, max_x = 0, max_y = 0;
    FORGE_CUDA_CHECK(cudaGetDevice(&device));
    FORGE_CUDA_CHECK(cudaDeviceGetAttribute(&max_x, cudaDevAttrMaxGridDimX, device));
    FORGE_CUDA_CHECK(cudaDeviceGetAttribute(&max_y, cudaDevAttrMaxGridDimY, device));
    if (grid.x > static_cast<unsigned>(max_x) || grid.y > static_cast<unsigned>(max_y))
        throw std::overflow_error("Tiled MatMul dimensions exceed device grid limits");
    tiled_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    FORGE_CUDA_CHECK(cudaGetLastError());
}
