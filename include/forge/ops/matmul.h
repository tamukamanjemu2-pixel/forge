#pragma once

#include "forge/tensor.h"

namespace forge {

    class Stream;
    enum class MatMulKernel { Naive, Tiled };
    // Enqueue on the tensor device's default stream; does not synchronize.

    void matmul(
        const Tensor& a,
        const Tensor& b,
        Tensor& output
    );

    // Enqueue on an explicit stream owned by the same device as all tensors.
    void matmul(const Tensor& a, const Tensor& b, Tensor& output, const Stream& stream);

    // Explicit selection. The legacy overloads continue to use Naive.
    void matmul(const Tensor& a, const Tensor& b, Tensor& output, MatMulKernel kernel);
    void matmul(const Tensor& a, const Tensor& b, Tensor& output, const Stream& stream, MatMulKernel kernel);

    // Single-launch GEMM with a ReLU epilogue; same validation/stream contract.
    void matmul_relu(const Tensor& a, const Tensor& b, Tensor& output, MatMulKernel kernel = MatMulKernel::Naive);
    void matmul_relu(const Tensor& a, const Tensor& b, Tensor& output, const Stream& stream, MatMulKernel kernel = MatMulKernel::Naive);

} // namespace forge
