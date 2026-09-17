#include "forge/ops/matmul.h"

#include "forge/stream.h"
#ifdef FORGE_HAS_CUDA
#include "matmul.h"
#include "fp16.h"
#include "../cuda_support.h"
#endif

#include <limits>
#include <stdexcept>

namespace forge {

namespace {
void validate_matmul(
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
        (a.dtype() != DataType::Float32 && a.dtype() != DataType::Float16) ||
        b.dtype() != a.dtype() || output.dtype() != a.dtype()
    ) {
        throw std::invalid_argument(
            "MatMul requires matching Float32 or Float16 tensors"
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

    const auto overlaps = [](const Tensor& x, const Tensor& y) {
        const auto xp = reinterpret_cast<std::uintptr_t>(x.data());
        const auto yp = reinterpret_cast<std::uintptr_t>(y.data());
        return xp <= yp ? yp - xp < x.nbytes() : xp - yp < y.nbytes();
    };
    if (overlaps(a, output) || overlaps(b, output))
        throw std::invalid_argument("MatMul output must not overlap either input");
}

void dispatch(const Tensor& a, const Tensor& b, Tensor& output, const Stream* stream, MatMulKernel kernel, bool activate = false) {
    if (kernel != MatMulKernel::Naive && kernel != MatMulKernel::Tiled)
        throw std::invalid_argument("Unknown MatMul kernel");
    validate_matmul(a, b, output);
    if (stream && stream->device() != a.device())
        throw std::invalid_argument("MatMul stream and tensors must be on the same device");
#ifdef FORGE_HAS_CUDA
    detail::DeviceScope scope(a.device().index);
    if (a.dtype() == DataType::Float16) {
        launch_matmul_fp16(a.data(), b.data(), output.data(), static_cast<int>(a.shape()[0]),
                          static_cast<int>(b.shape()[1]), static_cast<int>(a.shape()[1]),
                          stream ? static_cast<cudaStream_t>(stream->native_handle()) : nullptr,
                          kernel == MatMulKernel::Tiled, activate);
        return;
    }
    using Launcher = void (*)(const float*, const float*, float*, int, int, int, cudaStream_t);
    const Launcher launch = activate
        ? (kernel == MatMulKernel::Tiled ? launch_matmul_tiled_relu : launch_matmul_relu)
        : (kernel == MatMulKernel::Tiled ? launch_matmul_tiled : static_cast<Launcher>(launch_matmul));
    launch(static_cast<const float*>(a.data()), static_cast<const float*>(b.data()),
                  static_cast<float*>(output.data()), static_cast<int>(a.shape()[0]),
                  static_cast<int>(b.shape()[1]), static_cast<int>(a.shape()[1]),
                  stream ? static_cast<cudaStream_t>(stream->native_handle()) : nullptr);
#else
    (void)activate;
    throw std::runtime_error("CUDA MatMul is unavailable: build with FORGE_ENABLE_CUDA=ON");
#endif
}
} // namespace

void matmul(const Tensor& a, const Tensor& b, Tensor& output) {
    dispatch(a, b, output, nullptr, MatMulKernel::Naive);
}
void matmul(const Tensor& a, const Tensor& b, Tensor& output, const Stream& stream) {
    dispatch(a, b, output, &stream, MatMulKernel::Naive);
}
void matmul(const Tensor& a, const Tensor& b, Tensor& output, MatMulKernel kernel) {
    dispatch(a, b, output, nullptr, kernel);
}
void matmul(const Tensor& a, const Tensor& b, Tensor& output, const Stream& stream, MatMulKernel kernel) {
    dispatch(a, b, output, &stream, kernel);
}
void matmul_relu(const Tensor& a, const Tensor& b, Tensor& output, MatMulKernel kernel) {
    dispatch(a, b, output, nullptr, kernel, true);
}
void matmul_relu(const Tensor& a, const Tensor& b, Tensor& output, const Stream& stream, MatMulKernel kernel) {
    dispatch(a, b, output, &stream, kernel, true);
}
} // namespace forge
