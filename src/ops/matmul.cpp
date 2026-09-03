#include "forge/ops/matmul.h"

#include "matmul.h"

#include <limits>
#include <stdexcept>

namespace forge {

void matmul(
    const Tensor& a,
    const Tensor& b,
    Tensor& output
) {
    if (
        a.ndim() != 2 ||
        b.ndim() != 2 ||
        output.ndim() != 2
    ) {
        throw std::invalid_argument(
            "MatMul requires 2D tensors"
        );
    }

    if (
        a.dtype() != DataType::Float32 ||
        b.dtype() != DataType::Float32 ||
        output.dtype() != DataType::Float32
    ) {
        throw std::invalid_argument(
            "MatMul currently supports Float32 only"
        );
    }

    if (
        a.device().type != DeviceType::CUDA ||
        b.device().type != DeviceType::CUDA ||
        output.device().type != DeviceType::CUDA
    ) {
        throw std::invalid_argument(
            "MatMul currently requires CUDA tensors"
        );
    }

    if (
        a.device() != b.device() ||
        a.device() != output.device()
    ) {
        throw std::invalid_argument(
            "MatMul tensors must be on the same device"
        );
    }

    if (
        !a.is_contiguous() ||
        !b.is_contiguous() ||
        !output.is_contiguous()
    ) {
        throw std::invalid_argument(
            "MatMul currently requires contiguous tensors"
        );
    }

    const auto m = a.shape()[0];
    const auto k = a.shape()[1];

    if (b.shape()[0] != k) {
        throw std::invalid_argument(
            "MatMul inner dimensions do not match"
        );
    }

    const auto n = b.shape()[1];

    if (
        output.shape()[0] != m ||
        output.shape()[1] != n
    ) {
        throw std::invalid_argument(
            "MatMul output shape is incorrect"
        );
    }

    if (
        a.data() == nullptr ||
        b.data() == nullptr ||
        output.data() == nullptr
    ) {
        throw std::invalid_argument(
            "MatMul received a tensor without storage"
        );
    }

    constexpr auto max_int =
        static_cast<std::int64_t>(
            std::numeric_limits<int>::max()
        );

    if (
        m > max_int ||
        n > max_int ||
        k > max_int
    ) {
        throw std::overflow_error(
            "MatMul dimensions exceed CUDA kernel limits"
        );
    }

    launch_matmul(
        static_cast<const float*>(a.data()),
        static_cast<const float*>(b.data()),
        static_cast<float*>(output.data()),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k)
    );
}

} // namespace forge