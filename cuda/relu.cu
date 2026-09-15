#include "activations.h"
#include "../src/cuda_support.h"
#include <algorithm>
namespace {
__global__ void relu_kernel(const float* input, float* output, std::size_t count) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const float value = input[i];
        output[i] = value < 0.0f ? 0.0f : value;
    }
}
}
void launch_relu(const float* input, float* output, std::size_t count, cudaStream_t stream) {
    if (!input || !output || !count) throw std::invalid_argument("Invalid ReLU launch");
    const auto blocks = static_cast<unsigned>(std::min<std::size_t>(1 + (count - 1) / 256, 65535));
    relu_kernel<<<blocks, 256, 0, stream>>>(input, output, count);
    FORGE_CUDA_CHECK(cudaGetLastError());
}
