#pragma once
#include <cuda_runtime.h>
#include <cstddef>
void launch_relu(const float* input, float* output, std::size_t elements, cudaStream_t stream);
void launch_softmax(const float* input, float* output, std::size_t rows, std::size_t columns, cudaStream_t stream);
