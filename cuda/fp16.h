#pragma once
#include <cuda_runtime.h>
#include <cstddef>
// Storage contains IEEE binary16 values; accumulation/reduction uses float.
void launch_matmul_fp16(const void* a, const void* b, void* c, int m, int n, int k,
                        cudaStream_t stream, bool tiled, bool relu);
void launch_relu_fp16(const void* input, void* output, std::size_t count, cudaStream_t stream);
void launch_softmax_fp16(const void* input, void* output, std::size_t rows, std::size_t columns, cudaStream_t stream);
