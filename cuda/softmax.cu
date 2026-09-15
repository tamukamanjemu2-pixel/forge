#include "activations.h"
#include "../src/cuda_support.h"
#include <algorithm>
#include <cmath>
#include <math_constants.h>
namespace {
constexpr unsigned threads = 256;
__global__ void softmax_kernel(const float* input, float* output, std::size_t rows, std::size_t columns) {
    __shared__ float reduction[threads];
    const unsigned lane = threadIdx.x;
    for (std::size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const std::size_t base = row * columns;
        float maximum = -CUDART_INF_F;
        for (std::size_t col = lane; col < columns; col += threads)
            maximum = fmaxf(maximum, input[base + col]);
        reduction[lane] = maximum;
        __syncthreads();
        for (unsigned stride = threads / 2; stride; stride /= 2) {
            if (lane < stride) reduction[lane] = fmaxf(reduction[lane], reduction[lane + stride]);
            __syncthreads();
        }
        maximum = reduction[0];
        // All threads must consume the maximum before reusing shared storage.
        __syncthreads();
        float sum = 0.0f;
        for (std::size_t col = lane; col < columns; col += threads) {
            const float value = expf(input[base + col] - maximum);
            output[base + col] = value;
            sum += value;
        }
        reduction[lane] = sum;
        __syncthreads();
        for (unsigned stride = threads / 2; stride; stride /= 2) {
            if (lane < stride) reduction[lane] += reduction[lane + stride];
            __syncthreads();
        }
        const float denominator = reduction[0];
        for (std::size_t col = lane; col < columns; col += threads)
            output[base + col] /= denominator;
        // The next row must not overwrite shared state before all readers finish.
        __syncthreads();
    }
}
}
void launch_softmax(const float* input, float* output, std::size_t rows, std::size_t columns, cudaStream_t stream) {
    if (!input || !output || !rows || !columns) throw std::invalid_argument("Invalid Softmax launch");
    const auto blocks = static_cast<unsigned>(std::min<std::size_t>(rows, 65535));
    softmax_kernel<<<blocks, threads, 0, stream>>>(input, output, rows, columns);
    FORGE_CUDA_CHECK(cudaGetLastError());
}
