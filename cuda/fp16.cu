#include "fp16.h"
#include "../src/cuda_support.h"
#include <cuda_fp16.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>

namespace {
constexpr unsigned tile = 16, threads = 256;
template<bool Tiled, bool Activate>
__global__ void gemm(const __half* a, const __half* b, __half* c, int m, int n, int k) {
    const auto row = static_cast<std::size_t>(blockIdx.y) * tile + threadIdx.y;
    const auto col = static_cast<std::size_t>(blockIdx.x) * tile + threadIdx.x;
    float sum = 0.0f;
    if constexpr (Tiled) {
        __shared__ __half at[tile][tile], bt[tile][tile];
        for (std::size_t base = 0; base < static_cast<std::size_t>(k); base += tile) {
            at[threadIdx.y][threadIdx.x] = row < static_cast<std::size_t>(m) && base + threadIdx.x < static_cast<std::size_t>(k)
                ? a[row * k + base + threadIdx.x] : __float2half_rn(0.0f);
            bt[threadIdx.y][threadIdx.x] = col < static_cast<std::size_t>(n) && base + threadIdx.y < static_cast<std::size_t>(k)
                ? b[(base + threadIdx.y) * n + col] : __float2half_rn(0.0f);
            __syncthreads();
            for (unsigned i = 0; i < tile; ++i)
                sum += __half2float(at[threadIdx.y][i]) * __half2float(bt[i][threadIdx.x]);
            __syncthreads();
        }
    } else {
        if (row >= static_cast<std::size_t>(m) || col >= static_cast<std::size_t>(n)) return;
        for (int i = 0; i < k; ++i)
            sum += __half2float(a[row * k + i]) * __half2float(b[static_cast<std::size_t>(i) * n + col]);
    }
    if (row < static_cast<std::size_t>(m) && col < static_cast<std::size_t>(n))
        c[row * n + col] = __float2half_rn(Activate && sum < 0 ? 0.0f : sum);
}
__global__ void relu_kernel(const __half* input, __half* output, std::size_t count) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const float value = __half2float(input[i]);
        output[i] = value < 0 ? __float2half_rn(0.0f) : input[i];
    }
}
__global__ void softmax_kernel(const __half* input, __half* output, std::size_t rows, std::size_t columns) {
    __shared__ float reduction[threads];
    const unsigned lane = threadIdx.x;
    for (std::size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const auto base = row * columns;
        float maximum = -CUDART_INF_F;
        for (std::size_t col = lane; col < columns; col += threads)
            maximum = fmaxf(maximum, __half2float(input[base + col]));
        reduction[lane] = maximum;
        __syncthreads();
        for (unsigned stride = threads / 2; stride; stride /= 2) {
            if (lane < stride) reduction[lane] = fmaxf(reduction[lane], reduction[lane + stride]);
            __syncthreads();
        }
        maximum = reduction[0];
        __syncthreads();
        float sum = 0;
        for (std::size_t col = lane; col < columns; col += threads)
            sum += expf(__half2float(input[base + col]) - maximum);
        reduction[lane] = sum;
        __syncthreads();
        for (unsigned stride = threads / 2; stride; stride /= 2) {
            if (lane < stride) reduction[lane] += reduction[lane + stride];
            __syncthreads();
        }
        const float denominator = reduction[0];
        // Recompute exponentials rather than rounding intermediate values to half.
        for (std::size_t col = lane; col < columns; col += threads)
            output[base + col] = __float2half_rn(expf(__half2float(input[base + col]) - maximum) / denominator);
        __syncthreads();
    }
}
}
void launch_matmul_fp16(const void* a, const void* b, void* c, int m, int n, int k,
                        cudaStream_t stream, bool tiled, bool relu) {
    if (!a || !b || !c || m <= 0 || n <= 0 || k <= 0) throw std::invalid_argument("Invalid FP16 GEMM launch");
    const dim3 block(tile, tile), grid(n / tile + (n % tile != 0), m / tile + (m % tile != 0));
    int device = 0, max_x = 0, max_y = 0;
    FORGE_CUDA_CHECK(cudaGetDevice(&device));
    FORGE_CUDA_CHECK(cudaDeviceGetAttribute(&max_x, cudaDevAttrMaxGridDimX, device));
    FORGE_CUDA_CHECK(cudaDeviceGetAttribute(&max_y, cudaDevAttrMaxGridDimY, device));
    if (grid.x > static_cast<unsigned>(max_x) || grid.y > static_cast<unsigned>(max_y))
        throw std::overflow_error("FP16 GEMM exceeds grid limits");
    auto* aa = static_cast<const __half*>(a);
    auto* bb = static_cast<const __half*>(b);
    auto* cc = static_cast<__half*>(c);
    if (tiled) {
        if (relu) gemm<true, true><<<grid, block, 0, stream>>>(aa, bb, cc, m, n, k);
        else gemm<true, false><<<grid, block, 0, stream>>>(aa, bb, cc, m, n, k);
    } else {
        if (relu) gemm<false, true><<<grid, block, 0, stream>>>(aa, bb, cc, m, n, k);
        else gemm<false, false><<<grid, block, 0, stream>>>(aa, bb, cc, m, n, k);
    }
    FORGE_CUDA_CHECK(cudaGetLastError());
}
void launch_relu_fp16(const void* input, void* output, std::size_t count, cudaStream_t stream) {
    if (!input || !output || !count) throw std::invalid_argument("Invalid FP16 ReLU launch");
    auto blocks = static_cast<unsigned>(std::min<std::size_t>(1 + (count - 1) / threads, 65535));
    relu_kernel<<<blocks, threads, 0, stream>>>(static_cast<const __half*>(input), static_cast<__half*>(output), count);
    FORGE_CUDA_CHECK(cudaGetLastError());
}
void launch_softmax_fp16(const void* input, void* output, std::size_t rows, std::size_t columns, cudaStream_t stream) {
    if (!input || !output || !rows || !columns) throw std::invalid_argument("Invalid FP16 Softmax launch");
    auto blocks = static_cast<unsigned>(std::min<std::size_t>(rows, 65535));
    softmax_kernel<<<blocks, threads, 0, stream>>>(static_cast<const __half*>(input), static_cast<__half*>(output), rows, columns);
    FORGE_CUDA_CHECK(cudaGetLastError());
}
